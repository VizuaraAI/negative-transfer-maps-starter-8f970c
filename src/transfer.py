"""Turn a directory of finished runs into the negative transfer map.

This is the analytical heart of the project. Everything else produces runs; this
module produces the result.

Definitions used throughout:

    Delta(i <- j) = L_i(pair i+j) - L_i(single i)

    L_i(...) is held-out cross-entropy on domain i in nats/token. Delta is
    POSITIVE when adding domain j makes domain i worse, i.e. negative transfer.
    The matrix is asymmetric: code can damage legal without legal damaging code.

The one methodological commitment that makes the number mean something is
`match_tokens`: when True, the pair run is read off its eval curve at the step
where it had seen the same number of tokens-on-domain-i as the singleton run saw
in total. Without that, you are measuring "the pair run saw half as much domain
i", which is a data-budget effect and not interference.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import DOMAIN_ORDER


@dataclass
class RunRecord:
    run_id: str
    capacity: str
    nonembed_params: int
    mixture: dict[str, float]
    seed: int
    tokens_seen: dict[str, int]
    eval_curve: list[dict]
    final_per_domain: dict[str, float]

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted(self.mixture))

    @property
    def is_singleton(self) -> bool:
        return len(self.mixture) == 1

    @property
    def is_pair(self) -> bool:
        return len(self.mixture) == 2

    def loss_at_tokens(self, domain: str, target_tokens: int) -> float | None:
        """Loss on `domain` at the eval point closest to `target_tokens` seen.

        Returns None if the run never reached within 10% of the target, which
        keeps a truncated run from silently contaminating the map.
        """
        best, best_gap = None, None
        for entry in self.eval_curve:
            if domain not in entry.get("per_domain", {}):
                continue
            seen = entry.get("tokens_seen", {}).get(domain, 0)
            gap = abs(seen - target_tokens)
            if best_gap is None or gap < best_gap:
                best, best_gap = entry["per_domain"][domain], gap
        if best is None or best_gap is None:
            return None
        if best_gap > 0.10 * max(1, target_tokens):
            return None
        return float(best)


def load_runs(runs_dir: str | Path) -> list[RunRecord]:
    records = []
    for result_path in sorted(Path(runs_dir).glob("*/result.json")):
        raw = json.loads(result_path.read_text())
        records.append(RunRecord(
            run_id=raw["run_id"],
            capacity=raw["capacity"],
            nonembed_params=raw["nonembed_params"],
            mixture=raw["mixture"],
            seed=raw.get("seed", 0),
            tokens_seen=raw.get("tokens_seen", {}),
            eval_curve=raw.get("eval_curve", []),
            final_per_domain=raw.get("final_per_domain", {}),
        ))
    return records


def transfer_matrix(
    runs: list[RunRecord], capacity: str, domains: list[str] | None = None,
    match_tokens: bool = True,
) -> tuple[np.ndarray, list[str], dict]:
    """Build the K x K Delta matrix for one capacity scale.

    Returns (matrix, domain_order, provenance). matrix[i, j] is Delta(i <- j);
    the diagonal is NaN. Cells with missing runs are NaN, never zero: a hole in
    the grid must look like a hole, not like "no interference".
    """
    domains = domains or DOMAIN_ORDER
    idx = {d: n for n, d in enumerate(domains)}
    K = len(domains)
    matrix = np.full((K, K), np.nan)
    provenance: dict[str, list] = {"missing_singletons": [], "missing_pairs": [], "unmatched": []}

    at_capacity = [r for r in runs if r.capacity == capacity]
    singles: dict[str, list[RunRecord]] = {}
    pairs: dict[tuple[str, ...], list[RunRecord]] = {}
    for run in at_capacity:
        if run.is_singleton:
            singles.setdefault(run.domains[0], []).append(run)
        elif run.is_pair:
            pairs.setdefault(run.domains, []).append(run)

    def mean_over_seeds(values: list[float | None]) -> float | None:
        clean = [v for v in values if v is not None]
        return float(np.mean(clean)) if clean else None

    for i, j in itertools.permutations(domains, 2):
        single_runs = singles.get(i)
        pair_runs = pairs.get(tuple(sorted((i, j))))
        if not single_runs:
            provenance["missing_singletons"].append(i)
            continue
        if not pair_runs:
            provenance["missing_pairs"].append(f"{i}+{j}")
            continue

        baseline = mean_over_seeds([r.final_per_domain.get(i) for r in single_runs])
        if baseline is None:
            continue

        if match_tokens:
            target = int(np.mean([r.tokens_seen.get(i, 0) for r in single_runs]))
            paired = mean_over_seeds([r.loss_at_tokens(i, target) for r in pair_runs])
            if paired is None:
                provenance["unmatched"].append(f"{i}<-{j}")
                continue
        else:
            paired = mean_over_seeds([r.final_per_domain.get(i) for r in pair_runs])
            if paired is None:
                continue

        matrix[idx[i], idx[j]] = paired - baseline

    provenance["missing_singletons"] = sorted(set(provenance["missing_singletons"]))
    provenance["missing_pairs"] = sorted(set(provenance["missing_pairs"]))
    return matrix, domains, provenance


def asymmetry(matrix: np.ndarray) -> np.ndarray:
    """Delta(i<-j) - Delta(j<-i). Non-zero entries are the interesting story:
    one-directional damage cannot be explained by a symmetric capacity budget
    alone, so it points at the donor domain's token statistics."""
    return matrix - matrix.T


def summarize_capacity(matrix: np.ndarray) -> dict[str, float]:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return {"n_cells": 0}
    positive = finite[finite > 0]
    return {
        "n_cells": int(finite.size),
        "mean_delta": float(finite.mean()),
        "mean_positive_delta": float(positive.mean()) if positive.size else 0.0,
        "frac_damaging": float(positive.size / finite.size),
        "max_delta": float(finite.max()),
        "min_delta": float(finite.min()),
        "mean_abs_delta": float(np.abs(finite).mean()),
    }


def fit_decay(params: list[float], values: list[float]) -> dict[str, float]:
    """Fit log(interference) = a * log(N) + b and report the exponent.

    A clearly negative exponent is the quantitative version of "the damage
    disappears with scale"; an exponent near zero is the interesting negative
    result, and it is just as publishable.
    """
    pts = [(p, v) for p, v in zip(params, values) if p > 0 and v > 0 and np.isfinite(v)]
    if len(pts) < 3:
        return {"n_points": len(pts)}
    x = np.log(np.array([p for p, _ in pts]))
    y = np.log(np.array([v for _, v in pts]))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n_points": len(pts),
        "exponent": float(slope),
        "log_intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def critical_capacity(
    params: list[float], values: list[float], threshold: float = 0.005,
) -> float | None:
    """Smallest N at which interference is below `threshold` nats/token.

    Interpolated in log-N between the bracketing measured points, so the answer
    is not quantised to the five capacities actually trained.
    """
    pts = sorted((p, v) for p, v in zip(params, values) if np.isfinite(v) and p > 0)
    if not pts:
        return None
    for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
        if v0 > threshold >= v1:
            if v0 == v1:
                return float(p1)
            frac = (v0 - threshold) / (v0 - v1)
            return float(np.exp(np.log(p0) + frac * (np.log(p1) - np.log(p0))))
    if pts[0][1] <= threshold:
        return float(pts[0][0])
    return None  # never crosses: interference survives the whole ladder


def build_report(
    runs_dir: str | Path, domains: list[str] | None = None,
    match_tokens: bool = True, threshold: float = 0.005,
) -> dict:
    runs = load_runs(runs_dir)
    if not runs:
        raise SystemExit(f"no runs with result.json found under {runs_dir}")

    domains = domains or DOMAIN_ORDER
    capacities = sorted({r.capacity for r in runs},
                        key=lambda c: np.mean([r.nonembed_params for r in runs if r.capacity == c]))

    report: dict = {"domains": domains, "match_tokens": match_tokens,
                    "threshold": threshold, "capacities": {}, "n_runs": len(runs)}
    for cap in capacities:
        matrix, order, prov = transfer_matrix(runs, cap, domains, match_tokens)
        n_params = float(np.mean([r.nonembed_params for r in runs if r.capacity == cap]))
        report["capacities"][cap] = {
            "nonembed_params": n_params,
            "matrix": matrix.tolist(),
            "asymmetry": asymmetry(matrix).tolist(),
            "summary": summarize_capacity(matrix),
            "provenance": prov,
        }

    # Per-pair decay across the capacity ladder.
    caps = list(report["capacities"])
    params = [report["capacities"][c]["nonembed_params"] for c in caps]
    mats = [np.array(report["capacities"][c]["matrix"]) for c in caps]

    report["mean_positive_delta_by_capacity"] = {
        c: report["capacities"][c]["summary"].get("mean_positive_delta") for c in caps
    }
    report["global_decay_fit"] = fit_decay(
        params, [report["capacities"][c]["summary"].get("mean_positive_delta", np.nan) for c in caps]
    )

    per_pair = {}
    for a, b in itertools.combinations(range(len(domains)), 2):
        # Symmetrised pair interference: the worse of the two directions is what
        # a practitioner choosing a mixture actually cares about.
        series = [float(np.nanmax([m[a, b], m[b, a]])) if np.isfinite([m[a, b], m[b, a]]).any()
                  else np.nan for m in mats]
        key = f"{domains[a]}+{domains[b]}"
        per_pair[key] = {
            "series": series,
            "fit": fit_decay(params, series),
            "critical_capacity": critical_capacity(params, series, threshold),
        }
    report["per_pair"] = per_pair
    report["capacity_axis"] = {"order": caps, "nonembed_params": params}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the negative transfer map.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="results/transfer_report.json")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="nats/token below which a pair counts as non-competing")
    parser.add_argument("--no-match-tokens", action="store_true",
                        help="compare at equal TOTAL tokens instead of equal tokens-on-i")
    args = parser.parse_args()

    report = build_report(args.runs_dir, match_tokens=not args.no_match_tokens,
                          threshold=args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"{report['n_runs']} runs, {len(report['capacities'])} capacities\n")
    print(f"{'capacity':10s} {'params':>14s} {'mean +Delta':>12s} {'frac damaging':>14s}")
    for cap in report["capacity_axis"]["order"]:
        s = report["capacities"][cap]["summary"]
        print(f"{cap:10s} {report['capacities'][cap]['nonembed_params']:14,.0f} "
              f"{s.get('mean_positive_delta', float('nan')):12.4f} "
              f"{s.get('frac_damaging', float('nan')):14.2f}")
    fit = report["global_decay_fit"]
    if "exponent" in fit:
        print(f"\ndecay exponent {fit['exponent']:.3f} (R^2 = {fit['r2']:.3f})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
