#!/usr/bin/env bash
# Run the mechanistic diagnostics over every finished PAIR run.
#
# These need no extra training: they read the checkpoints the sweep already
# wrote. The output is what lets the paper say WHY the interference happens,
# not just that it does.
#
# Usage: bash scripts/run_diagnostics.sh [runs_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS_DIR="${1:-runs}"
count=0

for run in "$RUNS_DIR"/*/; do
  [[ -f "$run/model.pt" ]] || continue
  # Only pair runs have two domains to compare against each other.
  name=$(basename "$run")
  [[ "$name" == *"+"* ]] || continue
  if [[ -f "$run/diagnostics.json" ]]; then
    echo "skip (done) $name"
    continue
  fi
  echo "=== $name ==="
  python -m src.diagnostics "$run"
  count=$((count + 1))
done

echo
echo "diagnostics written for $count run(s)"
echo "next: correlate the per-pair gradient-conflict and polysemanticity numbers"
echo "      against the Delta values in results/transfer_report.json"
