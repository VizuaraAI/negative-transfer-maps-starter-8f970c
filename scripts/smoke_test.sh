#!/usr/bin/env bash
# End-to-end pipeline check. Costs a few minutes on a CPU and proves that data
# streaming, training, evaluation, the transfer-map assembly and the plotting
# all work before you spend real compute.
#
# Usage: bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=configs/smoke.yaml
OUT=runs_smoke

echo "== 1/4 domain registry =="
python -m src.data

echo
echo "== 2/4 capacity ladder =="
python -m src.model

echo
echo "== 3/4 three runs: two singletons and the pair =="
python -m src.train --config "$CFG" --mixture wiki       --out-dir "$OUT"
python -m src.train --config "$CFG" --mixture code       --out-dir "$OUT"
python -m src.train --config "$CFG" --mixture wiki,code  --out-dir "$OUT"

echo
echo "== 4/4 transfer map + figures =="
python -m src.transfer --runs-dir "$OUT" --out results_smoke/transfer_report.json
python -m src.plotting --report results_smoke/transfer_report.json \
                       --out-dir results_smoke/figures

echo
echo "smoke test passed. Numbers are meaningless at this budget; the pipeline works."
