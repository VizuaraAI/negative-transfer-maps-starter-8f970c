"""Mechanistic diagnostics that run on checkpoints you already have.

These are the "why" to the transfer map's "what". None of them require extra
training, which matters because the main grid is already expensive. Three
independent signals, each cheap:

1. `gradient_conflict`  : cos(g_i, g_j) between per-domain gradients. The
   classic multi-task interference measure (Yu et al., 2020; Wang et al., 2021).
2. `cka`                : linear CKA between per-domain activations, layer by
   layer. High overlap in a narrow model is the superposition story.
3. `polysemanticity`    : fraction of MLP hidden units that respond strongly to
   both domains. The most direct read on shared capacity.

All three should become WEAKER predictors as width grows if the capacity story
is right, which is itself a testable prediction and a good ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import build_eval_batches, get_tokenizer
from .evaluate import load_run
from .model import TinyLM


def _flat_grad(model: TinyLM, batches, device: str, max_batches: int = 8) -> torch.Tensor:
    """Accumulate a mean gradient over a few batches and flatten it."""
    model.zero_grad(set_to_none=True)
    used = 0
    for x, y in batches[:max_batches]:
        _, loss = model(x.to(device), y.to(device))
        (loss / min(max_batches, len(batches))).backward()
        used += 1
    if used == 0:
        raise ValueError("no batches supplied")
    grads = [
        p.grad.detach().reshape(-1) if p.grad is not None
        else torch.zeros(p.numel(), device=device)
        for p in model.parameters()
    ]
    model.zero_grad(set_to_none=True)
    return torch.cat(grads)


def gradient_conflict(
    model: TinyLM, suite: dict[str, list], device: str, max_batches: int = 8,
) -> dict[str, float]:
    """Pairwise cosine between per-domain gradients at the current weights.

    Negative values mean the two domains pull the shared parameters in opposing
    directions, the mechanism this project is claiming is capacity-dependent.
    """
    model.train()  # gradients, but dropout is 0.0 by default so this is harmless
    grads = {k: _flat_grad(model, b, device, max_batches) for k, b in suite.items()}
    out = {}
    keys = sorted(grads)
    for n, i in enumerate(keys):
        for j in keys[n + 1 :]:
            out[f"{i}|{j}"] = float(
                torch.nn.functional.cosine_similarity(grads[i], grads[j], dim=0)
            )
    return out


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two activation matrices (n_samples x n_features)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xy = float(np.linalg.norm(X.T @ Y, ord="fro") ** 2)
    xx = float(np.linalg.norm(X.T @ X, ord="fro"))
    yy = float(np.linalg.norm(Y.T @ Y, ord="fro"))
    if xx == 0.0 or yy == 0.0:
        return float("nan")
    return xy / (xx * yy)


@torch.no_grad()
def _collect_activations(
    model: TinyLM, batches, device: str, max_batches: int = 4, max_tokens: int = 4096,
) -> list[np.ndarray]:
    """Residual-stream activations per layer, flattened over tokens."""
    model.eval()
    per_layer: list[list[np.ndarray]] = []
    for x, _ in batches[:max_batches]:
        states = model.hidden_states(x.to(device))
        if not per_layer:
            per_layer = [[] for _ in states]
        for layer, h in enumerate(states):
            flat = h.reshape(-1, h.size(-1)).float().cpu().numpy()
            per_layer[layer].append(flat)
    return [np.concatenate(chunks, axis=0)[:max_tokens] for chunks in per_layer]


def representation_overlap(
    model: TinyLM, suite: dict[str, list], device: str, max_batches: int = 4,
) -> dict[str, list[float]]:
    """Per-layer linear CKA between every pair of domains' representations."""
    acts = {k: _collect_activations(model, b, device, max_batches)
            for k, b in suite.items()}
    out: dict[str, list[float]] = {}
    keys = sorted(acts)
    for n, i in enumerate(keys):
        for j in keys[n + 1 :]:
            n_layers = min(len(acts[i]), len(acts[j]))
            out[f"{i}|{j}"] = [
                linear_cka(acts[i][layer], acts[j][layer]) for layer in range(n_layers)
            ]
    return out


@torch.no_grad()
def polysemanticity(
    model: TinyLM, suite: dict[str, list], device: str, quantile: float = 0.95,
    max_batches: int = 4,
) -> dict[str, dict[str, float]]:
    """Fraction of MLP hidden units that fire strongly on BOTH domains.

    A unit is "active" for a domain if its mean post-GELU activation on that
    domain exceeds the `quantile` of its own cross-domain activation profile.
    Shared-active units are the slots that two domains are literally fighting
    over, so this number should fall as width grows.
    """
    model.eval()
    profiles: dict[str, list[np.ndarray]] = {}
    for key, batches in suite.items():
        sums: list[torch.Tensor] | None = None
        count = 0
        for x, _ in batches[:max_batches]:
            h = model.tok_emb(x.to(device)) + model.pos_emb(
                torch.arange(x.size(1), device=device)
            )
            layer_means = []
            for block in model.blocks:
                h = h + block.attn(block.ln1(h))
                hidden = block.mlp.hidden(block.ln2(h))
                layer_means.append(hidden.abs().mean(dim=(0, 1)).float().cpu())
                h = h + block.mlp(block.ln2(h))
            if sums is None:
                sums = layer_means
            else:
                sums = [s + m for s, m in zip(sums, layer_means)]
            count += 1
        profiles[key] = [(s / max(1, count)).numpy() for s in (sums or [])]

    out: dict[str, dict[str, float]] = {}
    keys = sorted(profiles)
    for n, i in enumerate(keys):
        for j in keys[n + 1 :]:
            shared, active_i, active_j, total = 0, 0, 0, 0
            for layer in range(min(len(profiles[i]), len(profiles[j]))):
                a, b = profiles[i][layer], profiles[j][layer]
                thresh_a = np.quantile(a, quantile)
                thresh_b = np.quantile(b, quantile)
                mask_a, mask_b = a >= thresh_a, b >= thresh_b
                shared += int(np.sum(mask_a & mask_b))
                active_i += int(mask_a.sum())
                active_j += int(mask_b.sum())
                total += a.size
            union = active_i + active_j - shared
            out[f"{i}|{j}"] = {
                "shared_units": shared,
                "jaccard": shared / union if union else float("nan"),
                "shared_frac_of_width": shared / total if total else float("nan"),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mechanistic diagnostics on a finished run."
    )
    parser.add_argument("run_dir")
    parser.add_argument("--domains", default=None,
                        help="comma-separated; defaults to the run's own mixture")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-blocks", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model, cfg = load_run(args.run_dir, device=args.device)
    domains = (args.domains.split(",") if args.domains else sorted(cfg.data.mixture))
    tokenizer = get_tokenizer(cfg.data.tokenizer)

    suite = {
        key: build_eval_batches(
            key, tokenizer, cfg.model.block_size, args.eval_blocks,
            batch_size=cfg.train.micro_batch_size,
            streaming=cfg.data.streaming, cache_dir=cfg.data.cache_dir,
        )
        for key in domains
    }

    report = {
        "run_id": cfg.run_id(),
        "capacity": cfg.model.capacity,
        "nonembed_params": model.n_params(non_embedding=True),
        "width": cfg.model.n_embd,
        "domains": domains,
        "gradient_conflict": gradient_conflict(model, suite, args.device),
        "cka_per_layer": representation_overlap(model, suite, args.device),
        "polysemanticity": polysemanticity(model, suite, args.device),
    }

    out = Path(args.out or Path(args.run_dir) / "diagnostics.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["gradient_conflict"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
