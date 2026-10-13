"""Figures for the paper, generated straight from `transfer_report.json`.

These produce the real versions of the two placeholder figures in the paper
draft, so once the grid finishes you can drop the output into `paper/figures/`
and the draft becomes a real paper with no other edits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ILLUSTRATIVE_NOTE = "Values are illustrative placeholders."


def _fmt_params(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return f"{n / 1e3:.0f}K"


def plot_transfer_maps(report: dict, out_path: str | Path, vmax: float | None = None) -> Path:
    """Panel of K x K heatmaps, one per capacity, on a shared diverging scale."""
    domains = report["domains"]
    caps = report["capacity_axis"]["order"]
    mats = [np.array(report["capacities"][c]["matrix"], dtype=float) for c in caps]

    if vmax is None:
        finite = np.concatenate([m[np.isfinite(m)].ravel() for m in mats]) if mats else np.array([])
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 0.05
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, len(caps), figsize=(3.1 * len(caps), 3.6), squeeze=False)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.88")  # holes and the diagonal render grey, never as zero

    for ax, cap, mat in zip(axes[0], caps, mats):
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=-vmax, vmax=vmax)
        n_params = report["capacities"][cap]["nonembed_params"]
        ax.set_title(f"N = {_fmt_params(n_params)}", fontsize=10)
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains, rotation=90, fontsize=7)
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels(domains, fontsize=7)
        ax.set_xlabel("donor domain $j$", fontsize=8)
        if ax is axes[0][0]:
            ax.set_ylabel("victim domain $i$", fontsize=8)

    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label(r"$\Delta L_i$ (nats/token)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Negative transfer map vs capacity (red = damaging)", fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_decay(report: dict, out_path: str | Path, top_pairs: int = 5) -> Path:
    """Two panels: mean interference vs N, and per-pair critical capacity."""
    caps = report["capacity_axis"]["order"]
    params = np.array(report["capacity_axis"]["nonembed_params"], dtype=float)
    mean_pos = np.array(
        [report["capacities"][c]["summary"].get("mean_positive_delta", np.nan) for c in caps],
        dtype=float,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    ax1.plot(params, mean_pos, "o-", label="all pairs", color="#3b4cc0")
    pairs = report.get("per_pair", {})
    ranked = sorted(
        pairs.items(),
        key=lambda kv: -(kv[1]["series"][0] if kv[1]["series"]
                         and np.isfinite(kv[1]["series"][0]) else -np.inf),
    )
    for name, info in ranked[:2]:
        ax1.plot(params, info["series"], "s--", alpha=0.65, label=name)

    ax1.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax1.set_xscale("log")
    ax1.set_xlabel("non-embedding parameters $N$")
    ax1.set_ylabel(r"mean positive $\Delta$ (nats/token)")
    ax1.set_title("Interference decays with capacity", fontsize=10)
    ax1.legend(fontsize=7, frameon=False)

    fit = report.get("global_decay_fit", {})
    if "exponent" in fit:
        ax1.annotate(
            f"slope {fit['exponent']:.2f}  ($R^2$ = {fit['r2']:.2f})",
            xy=(0.04, 0.06), xycoords="axes fraction", fontsize=7, color="0.3",
        )

    crits = [(name, info["critical_capacity"]) for name, info in pairs.items()
             if info.get("critical_capacity")]
    crits.sort(key=lambda kv: -kv[1])
    crits = crits[:top_pairs]
    if crits:
        names = [c[0] for c in crits]
        vals = np.array([c[1] for c in crits], dtype=float)
        colors = plt.get_cmap("viridis")(np.linspace(0.25, 0.85, len(vals)))
        ax2.barh(range(len(vals)), vals, color=colors)
        ax2.set_yticks(range(len(vals)))
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xscale("log")
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, "no pair crossed the threshold", ha="center",
                 va="center", fontsize=9, color="0.4")
    ax2.set_xlabel("critical capacity $N^*$ (params)")
    ax2.set_title("Per-pair critical capacity", fontsize=10)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the transfer report.")
    parser.add_argument("--report", default="results/transfer_report.json")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    out_dir = Path(args.out_dir)
    p1 = plot_transfer_maps(report, out_dir / "fig1_transfer_maps.png")
    p2 = plot_decay(report, out_dir / "fig2_decay.png")
    print(f"wrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
