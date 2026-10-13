# CLAUDE.md

Instructions for the coding agent working on this repository with Romy.

## What this project is

An empirical study measuring **pairwise negative transfer between pretraining
domains as a function of model capacity**. The deliverable is a workshop paper.
The core object is an asymmetric `K x K` matrix per capacity scale:

```
Delta(i <- j) = L_i(trained on i and j) - L_i(trained on i alone)
```

evaluated at a **matched number of tokens seen on domain i**. `Delta > 0` means
domain `j` damaged domain `i`.

The hypothesis: interference is large at 4M non-embedding parameters and
vanishes by 410M. The headline number is the exponent `alpha` in
`log(interference) = alpha * log(N) + c`, plus a per-pair critical capacity
`N*`.

**A null result is a real result here.** If `alpha` is indistinguishable from
zero, that refutes the capacity account and validates small-proxy mixture
tuning. Never tune the analysis toward the "exciting" direction, and never
describe a null outcome as a failure.

## Layout

| path | what it does |
|---|---|
| `src/config.py` | capacity presets, mixture specs, token budgets, YAML loading |
| `src/data.py` | the 8-domain registry, streaming loaders, `MixtureStream` token accounting |
| `src/model.py` | plain decoder-only transformer, width/depth knob, `n_nonembed_params` |
| `src/train.py` | trains one cell of the capacity x mixture grid, writes `result.json` |
| `src/evaluate.py` | scores a checkpoint on all 8 domains |
| `src/transfer.py` | assembles runs into the map, fits decay, computes `N*` |
| `src/diagnostics.py` | gradient conflict, linear CKA, polysemanticity on existing weights |
| `src/plotting.py` | the paper's two figures from `transfer_report.json` |
| `experiments/log_template.md` | experiment log; its tables map onto the paper's tables |

Data flow: `train.py` writes `runs/<run_id>/result.json` → `transfer.py` reads
all of them → `results/transfer_report.json` → `plotting.py` draws the figures.
`run_id` is `<capacity>__<domains joined by +>__s<seed>`.

## Conventions

- **Python 3.10+, standard library plus the pinned deps.** No new heavyweight
  dependency without asking. If something needs `scikit-learn` for one
  function, write the twenty lines instead.
- **Config over flags.** Anything that must stay identical across the grid lives
  in `configs/default.yaml`. Command-line flags override only what is supposed
  to vary: capacity, mixture, seed.
- **No silent defaults in the science path.** If a measurement cannot be made,
  return `None`/`NaN` and record why in a `provenance` field. Never substitute
  `0.0` for a missing measurement: zero means "measured, no effect".
- **Comments explain why, not what.** The existing comments flag confounds and
  justify methodological choices. Match that; do not add `# loop over domains`.
- **Determinism.** Seeds are explicit and logged. The eval suite is built once
  from a fixed seed that differs from any training seed, and is byte-identical
  across every run in the grid.
- Keep modules small enough to read in one sitting. If `transfer.py` grows past
  roughly 400 lines, split the analysis out rather than nesting more helpers.

## Before you change anything

```bash
python -m src.data --check     # 8/8 domains reachable
bash scripts/smoke_test.sh     # full pipeline at a toy budget, a few minutes
```

Run the smoke test again **after** any change to data loading, the model, or the
training loop. It is the only thing standing between a typo and a wasted
48-hour sweep.

## Three invariants that must not be broken

These are the claims the paper rests on. If a change would violate one, stop and
say so rather than working around it.

1. **Token matching.** Pair runs train for twice the total budget so that
   tokens-seen-on-domain-`i` matches the singleton. `transfer.py` reads the pair
   loss off the eval curve at that matched point and returns `None` if no eval
   point is within 10% of it. Do not "fix" that `None` by falling back to the
   final loss: that silently reintroduces the data-budget confound.
2. **Learning rate scaling.** LR is rescaled `1/width` across the ladder. A
   mistuned small model is indistinguishable from a capacity-limited one, which
   is the confound most likely to sink the paper. Never hardcode a single LR
   across capacities.
3. **Identical eval sets.** Every run is scored on byte-identical held-out
   batches. If evaluation ever becomes seed- or run-dependent, the deltas stop
   being comparable and the whole map is void.

## Where the project is going

Rough order. The experiment log in `experiments/log_template.md` tracks actual
progress.

1. Pipeline works end to end; 4 domains at `tiny`. The first real artifact is a
   `4 x 4` map with a visible asymmetry.
2. Same 4 domains across `tiny, small, base`. This is the first time the central
   question gets an answer, even a noisy one.
3. Widen to 8 domains; add seeds at the two smallest capacities for error bars.
4. Push to `medium` and `large` to pin down `N*`.
5. Ablations: LR sweep, GradNorm, width vs depth, compute-matched budget.
6. Diagnostics and the regression of `Delta` on gradient conflict / CKA /
   shared units.
7. The proxy test: pick a mixture from the 4M map, pick another from the 160M
   map, train both at a held-out larger capacity, see whether they differ.

## Working with Romy

- Prefer the smallest experiment that could change the answer. A 4-domain map at
  two capacities beats an 8-domain map at one.
- When a run fails, the log row stays with status `failed` and a reason. Rows are
  never deleted; a sweep whose failures vanish is a cherry-pick.
- Before writing new analysis code, check whether `transfer.py` already computes
  it. Most "new" questions are a different slice of the same report JSON.
- If asked to make results look better, ask what the error bars are first.
