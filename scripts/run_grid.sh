#!/usr/bin/env bash
# The main sweep: for each capacity, train every singleton and every pair.
#
# Run count per capacity = K singletons + K*(K-1)/2 pairs = 8 + 28 = 36.
# Across five capacities that is 180 runs, which is why you should start with
# --domains web,wiki,code,legal (4 domains = 10 runs per capacity) and only
# widen once the pipeline and the effect are both confirmed.
#
# Usage:
#   bash scripts/run_grid.sh tiny
#   bash scripts/run_grid.sh tiny,small,base --domains web,wiki,code,legal
#   bash scripts/run_grid.sh tiny --seeds 0,1,2 --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."

CAPACITIES="${1:-tiny}"
shift || true

DOMAINS="web,wiki,code,math,legal,biomed,dialog,multiling"
SEEDS="0"
CONFIG="configs/default.yaml"
OUT_DIR="runs"
TOKENS=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domains) DOMAINS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --tokens-per-domain) TOKENS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -r -a CAPS <<< "$CAPACITIES"
IFS=',' read -r -a DOMS <<< "$DOMAINS"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

n_dom=${#DOMS[@]}
n_pairs=$(( n_dom * (n_dom - 1) / 2 ))
total=$(( (n_dom + n_pairs) * ${#CAPS[@]} * ${#SEED_LIST[@]} ))
echo "capacities : ${CAPS[*]}"
echo "domains    : ${DOMS[*]} ($n_dom singletons, $n_pairs pairs)"
echo "seeds      : ${SEED_LIST[*]}"
echo "total runs : $total"
echo

run_one() {
  local capacity="$1" mixture="$2" seed="$3"
  local cmd=(python -m src.train --config "$CONFIG" --capacity "$capacity"
             --mixture "$mixture" --seed "$seed" --out-dir "$OUT_DIR")
  [[ -n "$TOKENS" ]] && cmd+=(--tokens-per-domain "$TOKENS")

  # Skip work that is already on disk so an interrupted sweep can be resumed.
  local mix_id
  mix_id=$(echo "$mixture" | tr ',' '\n' | sort | paste -sd'+' -)
  local run_id="${capacity}__${mix_id}__s${seed}"
  if [[ -f "$OUT_DIR/$run_id/result.json" ]]; then
    echo "  skip (done)  $run_id"
    return 0
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  would run    ${cmd[*]}"
    return 0
  fi
  echo "  running      $run_id"
  "${cmd[@]}"
}

for capacity in "${CAPS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    echo "--- capacity=$capacity seed=$seed : singletons ---"
    for d in "${DOMS[@]}"; do
      run_one "$capacity" "$d" "$seed"
    done
    echo "--- capacity=$capacity seed=$seed : pairs ---"
    for ((a = 0; a < n_dom; a++)); do
      for ((b = a + 1; b < n_dom; b++)); do
        run_one "$capacity" "${DOMS[a]},${DOMS[b]}" "$seed"
      done
    done
  done
done

if [[ $DRY_RUN -eq 0 ]]; then
  echo
  echo "=== assembling the map ==="
  python -m src.transfer --runs-dir "$OUT_DIR" --out results/transfer_report.json
  python -m src.plotting --report results/transfer_report.json --out-dir results/figures
fi
