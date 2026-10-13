# Negative Transfer Maps Under Capacity Pressure

Starter code for Romy Wadhwa's research project: measuring **which pretraining
domains damage each other when model capacity is scarce**, and whether that
damage disappears as models get bigger.

## The question

When you train a language model on a mixture of domains (code, legal text,
math, dialogue, ...), some of those domains fight each other for the model's
limited representational capacity. The claim this project tests:

> Domain interference is a **capacity** phenomenon. It should be large and
> structured in a 4M-parameter model and nearly absent in a 410M-parameter one.

If that is true, it has a sharp practical consequence. The data-mixture
literature almost always tunes mixture weights on a small proxy model and then
reuses them at a much larger scale. If interference vanishes with scale, those
weights are partly fitting a bottleneck the deployed model does not have.

The measurement is a **negative transfer map**: an asymmetric `K x K` matrix
where

```
Delta(i <- j) = L_i(trained on i and j) - L_i(trained on i alone)
```

`Delta > 0` means adding domain `j` **hurt** domain `i`. The matrix is
asymmetric on purpose: code can damage legal without legal damaging code, and
symmetric mixture-weight methods cannot express that.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate

# Install torch FIRST, from the index that matches your hardware:
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU

pip install -r requirements.txt
```

No datasets are committed here. All eight domains stream from the Hugging Face
Hub on demand. Set `HF_TOKEN` in your environment if you hit rate limits.

## The first three things to run

**1. Confirm every domain still resolves on the Hub** (about a minute). Hub
datasets get renamed, gated and archived, and you want to find that out now and
not 30 runs into a sweep:

```bash
python -m src.data --check
```

You should see `8/8 domains reachable`.

**2. Look at the capacity ladder** you will be sweeping over:

```bash
python -m src.model
```

```
preset    layers   width    non-embed        total
tiny           6     240    4,166,400   16,362,240
small          8     384   14,196,480   33,709,824
base          12     560   45,246,880   73,703,840
medium        16     912  159,886,368  206,230,560
large         24    1200  415,096,800  476,076,000
```

Non-embedding parameters are the capacity axis, because embedding parameters do
not take part in the shared computation where domains compete.

**3. Run the whole pipeline end to end at a toy budget** (a few minutes on a
laptop CPU). This trains two singletons and one pair, assembles the transfer
map, and draws the figures:

```bash
bash scripts/smoke_test.sh
```

The numbers it produces are meaningless. Its job is to prove that data
streaming, training, evaluation, map assembly and plotting all work before you
spend real compute. **Run it again every time you touch the data or training
code.**

## The real sweep

Start small. Four domains is 10 runs per capacity instead of 36:

```bash
# see what it would do, without doing it
bash scripts/run_grid.sh tiny --domains web,wiki,code,legal --dry-run

# the smallest real result: one capacity, four domains
bash scripts/run_grid.sh tiny --domains web,wiki,code,legal

# the contrast that makes the paper: the same map at three capacities
bash scripts/run_grid.sh tiny,small,base --domains web,wiki,code,legal
```

The script skips any run whose `result.json` already exists, so an interrupted
sweep resumes where it stopped. When runs finish it assembles the map:

```bash
python -m src.transfer --runs-dir runs --out results/transfer_report.json
python -m src.plotting --report results/transfer_report.json --out-dir results/figures
```

`src/plotting.py` produces the real versions of the two placeholder figures in
the paper draft. Drop them into `paper/figures/` and the draft becomes a real
paper with no other edits.

Then the mechanistic side, which needs no extra training because it reads the
checkpoints you already have:

```bash
bash scripts/run_diagnostics.sh runs
```

## Layout

```
configs/
  default.yaml       the real settings; everything that must stay identical across runs
  smoke.yaml         toy budget for the end-to-end check
src/
  config.py          capacity presets, mixtures, token budgets
  data.py            the eight-domain registry and the streaming token pipeline
  model.py           a plain decoder-only transformer with a width/depth knob
  train.py           train one cell of the capacity x mixture grid
  evaluate.py        score a checkpoint on all eight domains
  transfer.py        assemble runs into the K x K map, fit the decay, find N*
  diagnostics.py     gradient conflict, CKA, polysemanticity on existing weights
  plotting.py        the paper's two figures, straight from the report JSON
scripts/
  smoke_test.sh      run this first
  run_grid.sh        the main sweep, resumable
  run_diagnostics.sh mechanistic diagnostics over finished pair runs
experiments/
  log_template.md    experiment log whose tables map onto the paper's tables
```

## The three things that will decide whether this is publishable

Everything else is engineering. These are the claims a reviewer will attack, so
the code is built around defending them.

**1. Matched tokens per domain.** A 50/50 pair run that trains for the same
total tokens as a singleton has seen *half as much* of domain `i`. Comparing
final losses then measures a data budget difference, not interference. So pair
runs train for twice the total budget (`budget_mode: per_domain`), and
`transfer.py` reads the pair loss off its eval curve at the step where
tokens-seen-on-`i` matches the singleton. If no eval point is within 10% of the
target, that cell becomes `NaN` rather than a quietly wrong number.

**2. The learning rate confound.** A mistuned small model looks *exactly* like a
capacity-limited one. This is the single most dangerous confound in the project.
The LR is rescaled as `1/width` (muP-style) so no capacity in the ladder is
accidentally mistuned, and you should still run the explicit LR sweep ablation
at the two smallest capacities. If the interference survives the best LR at
every width, the capacity story stands. If it collapses, it was tuning all
along, and better to learn that in week 3 than in review.

**3. Holes must look like holes.** A missing run produces `NaN` in the map, never
`0.0`. Zero means "measured, no interference"; `NaN` means "not measured". The
`provenance` block in `transfer_report.json` lists every missing singleton, every
missing pair, and every cell that could not be token-matched. Read it before you
believe a map.

## A note on what counts as a result

The hypothesis is that interference decays with capacity. The fitted exponent
`alpha` in `log(interference) = alpha * log(N) + c` is the headline number.

A clearly negative `alpha` supports the capacity account. But an `alpha`
indistinguishable from zero is **not a failed project**: it would mean domain
competition is a property of the data rather than the model, that small proxies
are a sound instrument, and that the superposition account does not explain
inter-domain interference at these scales. That refutes a widely held intuition
and aligns with an existing finding in multilingual scaling. Write it up with
the same confidence either way, and decide which story you are telling only
after the error bars are in.
