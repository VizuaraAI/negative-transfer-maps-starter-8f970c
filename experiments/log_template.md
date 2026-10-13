# Experiment log

One row per run. The columns are chosen so that this table can be pivoted
directly into Table 1 of the paper and into the heatmaps of Figure 2: there is
no second bookkeeping step between "I ran something" and "it is in the paper".

Fill a row in **the same sitting** as the run. A log written a week later is
fiction.

---

## 1. Run ledger

Every training run, in order. `run_id` matches the directory name under `runs/`
(`<capacity>__<domains joined by +>__s<seed>`).

| run_id | date | capacity | non-emb. params | mixture | seed | tokens/domain | budget mode | final loss (own domains) | wall clock | status | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `tiny__wiki__s0` | 2026-10-20 | tiny | 4.1M | wiki | 0 | 50M | per_domain | 3.91 | 0h42 | done | singleton baseline |
| `tiny__code__s0` | 2026-10-20 | tiny | 4.1M | code | 0 | 50M | per_domain | 2.14 | 0h41 | done | singleton baseline |
| `tiny__code+wiki__s0` | 2026-10-21 | tiny | 4.1M | code,wiki | 0 | 50M | per_domain | 4.07 / 2.19 | 1h25 | done | first pair, both degraded |
| | | | | | | | | | | | |

Status values: `queued`, `running`, `done`, `failed`, `discarded`. If a run is
`discarded`, say why in notes and **do not delete the row**. Discarded runs that
vanish from the log are how a sweep silently becomes a cherry-pick.

---

## 2. Transfer coefficients

Derived, not typed by hand: produced by `python -m src.transfer`. This table
mirrors the matrix in `results/transfer_report.json` and feeds Figure 2a.

| capacity | victim $i$ | donor $j$ | $L_i$ single | $L_i$ pair | $\Delta(i \leftarrow j)$ | tokens matched? | seeds | $\pm$ sd |
|---|---|---|---|---|---|---|---|---|
| tiny | wiki | code | 3.91 | 4.07 | +0.16 | yes | 1 | n/a |
| tiny | code | wiki | 2.14 | 2.19 | +0.05 | yes | 1 | n/a |
| | | | | | | | | |

The two rows above are the same pair in both directions. Keep them adjacent:
the asymmetry is a result, not a rounding error.

---

## 3. Capacity summary

One row per capacity. **This table becomes Table 1 of the paper**, column for
column.

| capacity | non-emb. params | $\overline{\Delta^+}$ | max $\Delta$ | % damaging | mean abs. asymmetry | mean grad. cosine | notes |
|---|---|---|---|---|---|---|---|
| tiny | 4.1M | | | | | | |
| small | 14.2M | | | | | | |
| base | 45.2M | | | | | | |
| medium | 159.7M | | | | | | |
| large | 414.7M | | | | | | |

Fitted decay exponent $\alpha$: ______  ($R^2$ = ______)

---

## 4. Per-pair critical capacity

Feeds Figure 2c. Pairs that never cross the threshold get `none`, never a
guessed number.

| pair | $\Delta$ @ tiny | @ small | @ base | @ medium | @ large | $N^*$ | crosses? |
|---|---|---|---|---|---|---|---|
| code/legal | | | | | | | |
| web/wiki | | | | | | none | no |
| | | | | | | | |

---

## 5. Diagnostics

From `python -m src.diagnostics`. Used in the Analysis section to test whether
cheap single-model signals predict the expensive paired measurement.

| run_id | pair | grad. cosine | mean CKA | peak-layer CKA | shared-unit Jaccard | sign conflict | $\Delta$ (worse dir.) |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

---

## 6. Ablation ledger

Each ablation in the paper's Analysis section, and whether it survived.

| ablation | question it answers | config change | result | verdict |
|---|---|---|---|---|
| LR sweep @ tiny, small | is the effect just mistuning? | `optim.base_lr` grid | | |
| GradNorm | is it gradient magnitude imbalance? | per-domain grad norm | | |
| width vs depth | is it width specifically? | matched params, varied aspect ratio | | |
| compute-matched | does the budget convention flip the sign? | `budget_mode: total` | | |
| proxy test | does a small-scale mixture still help large? | mixture from 4M map @ large | | |
| three-domain additivity | are pairwise terms sufficient? | 3-domain mixtures | | |

**Verdict** is one of: `supports`, `refutes`, `inconclusive`. Write it before
you look at whether it is convenient.

---

## 7. Decision log

Dated, one line each. Every methodological choice that a reviewer could
reasonably have made differently. This is the raw material for the paper's
Methodology and Limitations sections, and the hardest thing to reconstruct
later.

| date | decision | alternative considered | why |
|---|---|---|---|
| 2026-10-18 | capacity axis = non-embedding params | total params | embedding params do not take part in the shared computation where domains compete |
| 2026-10-19 | token-matched per domain is the headline | compute-matched | compute-matched confounds interference with a halved data budget |
| | | | |

---

## 8. Open questions

Running list. Move items out as they are answered, with the run_id that
answered them.

- [ ] Does the asymmetry direction stay stable across seeds, or is it noise?
- [ ] Is `legal` damaged by everything, or is its eval set just high variance?
- [ ] At `large`, is the residual interference real or below the noise floor?
      (Needs the 3-seed error bars before it can be claimed either way.)
