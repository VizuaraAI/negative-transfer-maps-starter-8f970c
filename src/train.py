"""Pretraining entry point for one cell of the capacity x mixture grid.

One invocation of this script is one run, and one run is one row in
`experiments/log_template.md`. The three things it is careful about are the
three things a reviewer will attack:

1. tokens-per-domain is tracked exactly and logged, so "matched tokens on
   domain i" is a measured fact rather than an assumption;
2. the learning rate is rescaled with width, so a struggling small model is
   capacity-limited and not just mistuned;
3. the eval suite is built once from a fixed seed, so every run in the grid is
   scored on identical held-out batches.

Usage:
    python -m src.train --config configs/default.yaml \
        --capacity tiny --mixture code --tokens-per-domain 2000000
    python -m src.train --config configs/default.yaml \
        --capacity tiny --mixture code,legal
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .config import ExperimentConfig, load_config
from .data import DOMAINS, MixtureStream, get_tokenizer
from .evaluate import evaluate_all, make_eval_suite
from .model import build_model


def parse_mixture(spec: str) -> dict[str, float]:
    """Parse "code,legal" or "code:0.8,legal:0.2" into normalised weights."""
    weights: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, w = part.partition(":")
        key = key.strip()
        if key not in DOMAINS:
            raise ValueError(f"unknown domain {key!r}; known: {sorted(DOMAINS)}")
        weights[key] = float(w) if w else 1.0
    if not weights:
        raise ValueError("empty mixture")
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32  # bf16 autocast on CPU is slow and not worth it here
    return {"float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16}[name]


def lr_at_step(step: int, total_steps: int, peak_lr: float, cfg) -> float:
    """Linear warmup then cosine decay to min_lr_ratio * peak."""
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


def train(cfg: ExperimentConfig) -> Path:
    device = resolve_device(cfg.train.device)
    dtype = resolve_dtype(cfg.train.dtype, device)
    torch.manual_seed(cfg.data.seed)

    run_dir = Path(cfg.train.out_dir) / cfg.run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    tokenizer = get_tokenizer(cfg.data.tokenizer)
    model = build_model(cfg.model).to(device)
    if cfg.train.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    n_nonembed = model.n_params(non_embedding=True)
    peak_lr = cfg.optim.lr_for_width(cfg.model.n_embd)
    tokens_per_step = cfg.train.batch_tokens
    total_steps = max(1, cfg.data.total_tokens() // tokens_per_step)
    accum = max(1, tokens_per_step // (cfg.train.micro_batch_size * cfg.model.block_size))

    print(f"run        {cfg.run_id()}")
    print(f"device     {device} ({dtype})")
    print(f"capacity   {cfg.model.capacity}: {cfg.model.n_layer}L x {cfg.model.n_embd}d, "
          f"{n_nonembed:,} non-embedding params")
    print(f"mixture    {cfg.data.mixture}")
    print(f"budget     {cfg.data.total_tokens():,} tokens "
          f"({cfg.data.tokens_per_domain:,}/domain, mode={cfg.data.budget_mode})")
    print(f"schedule   {total_steps} steps x {tokens_per_step:,} tok, "
          f"peak lr {peak_lr:.2e}, accum {accum}")

    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.optim.weight_decay), lr=peak_lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
    )
    stream = MixtureStream(
        cfg.data.mixture, tokenizer, cfg.model.block_size, seed=cfg.data.seed,
        streaming=cfg.data.streaming, cache_dir=cfg.data.cache_dir,
    )
    eval_suite = make_eval_suite(cfg, tokenizer, domains=sorted(cfg.data.mixture))

    history: list[dict] = []
    log_path = run_dir / "train_log.jsonl"
    log_file = log_path.open("w")
    start = time.time()

    model.train()
    for step in range(total_steps):
        lr = lr_at_step(step, total_steps, peak_lr, cfg.optim)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(accum):
            x, y, _ = stream.next_batch(cfg.train.micro_batch_size, device=device)
            if device == "cpu":
                _, loss = model(x, y)
            else:
                with torch.autocast(device_type=device, dtype=dtype):
                    _, loss = model(x, y)
            (loss / accum).backward()
            running += float(loss) / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()

        if step % cfg.train.log_every == 0 or step == total_steps - 1:
            record = {
                "step": step, "lr": lr, "train_loss": running,
                "tokens_seen": dict(stream.tokens_seen),
                "min_tokens_per_domain": stream.min_tokens_seen(),
                "elapsed_s": round(time.time() - start, 1),
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(f"  step {step:6d}/{total_steps}  loss {running:.4f}  lr {lr:.2e}")

        if cfg.train.eval_every and (
            (step + 1) % cfg.train.eval_every == 0 or step == total_steps - 1
        ):
            per_domain = evaluate_all(model, eval_suite, device)
            entry = {
                "step": step,
                "tokens_seen": dict(stream.tokens_seen),
                "per_domain": {k: v["loss"] for k, v in per_domain.items()},
            }
            history.append(entry)
            log_file.write(json.dumps({"eval": entry}) + "\n")
            log_file.flush()
            pretty = "  ".join(f"{k}={v['loss']:.4f}" for k, v in per_domain.items())
            print(f"  eval  step {step:6d}  {pretty}")

    log_file.close()

    if cfg.train.save_checkpoint:
        to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(to_save.state_dict(), run_dir / "model.pt")

    # The eval curve is what `transfer.py` reads: it lets the delta be taken at
    # a matched tokens-on-domain-i point rather than only at the end of training.
    (run_dir / "result.json").write_text(json.dumps({
        "run_id": cfg.run_id(),
        "capacity": cfg.model.capacity,
        "nonembed_params": n_nonembed,
        "mixture": cfg.data.mixture,
        "seed": cfg.data.seed,
        "budget_mode": cfg.data.budget_mode,
        "tokens_per_domain_target": cfg.data.tokens_per_domain,
        "tokens_seen": dict(stream.tokens_seen),
        "eval_curve": history,
        "final_per_domain": history[-1]["per_domain"] if history else {},
        "wall_clock_s": round(time.time() - start, 1),
    }, indent=2))

    print(f"done in {time.time() - start:.1f}s -> {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one grid cell.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--capacity", default=None, help="tiny|small|base|medium|large")
    parser.add_argument("--mixture", default=None, help='e.g. "code" or "code,legal"')
    parser.add_argument("--tokens-per-domain", type=int, default=None)
    parser.add_argument("--budget-mode", default=None, choices=["per_domain", "total"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        **{
            "model.capacity": args.capacity,
            "data.mixture": parse_mixture(args.mixture) if args.mixture else None,
            "data.tokens_per_domain": args.tokens_per_domain,
            "data.budget_mode": args.budget_mode,
            "data.seed": args.seed,
            "train.device": args.device,
            "train.out_dir": args.out_dir,
            "train.run_name": args.run_name,
            "train.eval_every": args.eval_every,
        },
    )
    if args.no_checkpoint:
        cfg.train.save_checkpoint = False
    train(cfg)


if __name__ == "__main__":
    main()
