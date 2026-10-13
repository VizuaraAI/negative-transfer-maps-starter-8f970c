"""Per-domain held-out evaluation.

Every run is scored on all eight domains, not just the ones it trained on. The
off-mixture columns are free information: they tell you about generalisation,
and they are what turns a 2-domain run into a row of the transfer map rather
than a single number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ExperimentConfig, load_config
from .data import DOMAIN_ORDER, build_eval_batches, get_tokenizer
from .model import TinyLM, build_model


@torch.no_grad()
def eval_domain(
    model: TinyLM, batches: list[tuple[torch.Tensor, torch.Tensor]], device: str,
) -> dict[str, float]:
    """Token-weighted mean cross-entropy (nats/token) over a fixed batch list."""
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y, reduction="sum")
        total_loss += float(loss)
        total_tokens += y.numel()
    if was_training:
        model.train()
    if total_tokens == 0:
        return {"loss": float("nan"), "ppl": float("nan"), "tokens": 0}
    mean = total_loss / total_tokens
    return {"loss": mean, "ppl": float(torch.exp(torch.tensor(mean))), "tokens": total_tokens}


def make_eval_suite(cfg: ExperimentConfig, tokenizer, domains=None) -> dict[str, list]:
    """Build the fixed eval batches for each domain, once, before training."""
    domains = domains or DOMAIN_ORDER
    n_blocks = max(
        cfg.train.micro_batch_size,
        cfg.data.eval_tokens_per_domain // cfg.model.block_size,
    )
    suite = {}
    for key in domains:
        suite[key] = build_eval_batches(
            key, tokenizer, cfg.model.block_size, n_blocks,
            batch_size=cfg.train.micro_batch_size,
            streaming=cfg.data.streaming, cache_dir=cfg.data.cache_dir,
        )
    return suite


def evaluate_all(
    model: TinyLM, suite: dict[str, list], device: str,
) -> dict[str, dict[str, float]]:
    return {key: eval_domain(model, batches, device) for key, batches in suite.items()}


def load_run(run_dir: str | Path, device: str = "cpu") -> tuple[TinyLM, ExperimentConfig]:
    run_dir = Path(run_dir)
    cfg_raw = json.loads((run_dir / "config.json").read_text())
    from .config import DataConfig, ModelConfig, OptimConfig, TrainConfig

    cfg = ExperimentConfig(
        model=ModelConfig(**cfg_raw["model"]),
        optim=OptimConfig(**cfg_raw["optim"]),
        data=DataConfig(**cfg_raw["data"]),
        train=TrainConfig(**cfg_raw["train"]),
    )
    model = build_model(cfg.model).to(device)
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a finished run on all domains.")
    parser.add_argument("run_dir", help="path to a runs/<run_id> directory")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="defaults to <run_dir>/final_eval.json")
    args = parser.parse_args()

    model, run_cfg = load_run(args.run_dir, device=args.device)
    cfg = load_config(args.config)
    cfg.model = run_cfg.model
    tokenizer = get_tokenizer(cfg.data.tokenizer)

    suite = make_eval_suite(cfg, tokenizer)
    results = evaluate_all(model, suite, args.device)

    out = Path(args.out or Path(args.run_dir) / "final_eval.json")
    out.write_text(json.dumps({"run_id": run_cfg.run_id(), "per_domain": results}, indent=2))
    for key in DOMAIN_ORDER:
        if key in results:
            print(f"  {key:10s} loss {results[key]['loss']:.4f}  ppl {results[key]['ppl']:.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
