# Uncertainty

`python -m ai_chip_imports.uncertainty` computes the ranges reported with the estimates: a
conditional 90% interval for each destination, and a structural envelope across alternative
readings of the trade. The [methodology page](https://epoch.ai/data/ai-chip-imports-documentation/methodology)
describes the priors and the cases; the module docstrings of `ai_chip_imports.uncertainty` give
the details.

## What it runs

- **Draws.** Each draw perturbs the non-AI reference, the imported content of exports (k), the
  content classifier's price and density thresholds, selected content readings, and Epoch's
  per-model sales and prices, then reruns the whole pipeline: netting, composite reference,
  ledger, content, estimator and allocation. Draws come from two independently scrambled Sobol
  sequences with fixed seeds, so a run is reproducible and a larger run extends a smaller one.
- **Cases.** 144 structural cases, each held fixed over the whole history: three non-AI
  references, server-mixture and selection rules, composition stresses on ambiguous parts and
  other units, the 2022 reference, a merchant-only price reference and a high-China case, each
  crossed with 0%, 50% and 100% of the Malaysia server-gap deduction. Every case runs the same
  draws. `central__central` is the published model.

Each draw allocates exactly its own sampled supply, so world totals have one interval in every
case. A draw is rejected when the allocator reports a warning or an infeasible fit; the rejected
draws are kept and counted.

## Running it

The inputs are the same as for the main pipeline (see [data.md](data.md)), with the clean books
and the 2021 reference inputs built.

```bash
python -m ai_chip_imports.uncertainty prepare                  # central universe and controls
python -m ai_chip_imports.uncertainty probe --workers 3        # checks before the run
python -m ai_chip_imports.uncertainty run --samples 64 --workers 6
python -m ai_chip_imports.uncertainty report
```

- `prepare` builds the central universe once and the deterministic results that each case's
  undisturbed draw must reproduce.
- `probe` runs seven representative cases. Each one's undisturbed draw is checked against those
  results, and one draw is computed both with and without the per-worker observation cache.
- `run` starts one process per case, `--workers` at a time. It is resumable: completed draws are
  skipped, and `status` reports progress. `--samples` must be a power of two; raising it later
  extends the same sequences.

At 64 draws per case (9,216 draws) a run takes about a day on six cores. Everything lands in
the run directory, `output/uncertainty/` by default (use `--out` to choose another). The run
refuses to continue if the package code or the input files change after `prepare`; start a new
directory instead.

## Outputs

**`country_intervals.csv`** -- one row per destination (and `World`).

| Column | Meaning |
|---|---|
| `central_h100e` | the published estimate: cumulative arrivals since 2022, H100e |
| `p05`, `median`, `p95` | the central case's draws: the conditional 90% interval |
| `envelope_low`, `envelope_high` | lowest case 5th percentile, highest case 95th percentile |

**`case_intervals.csv`** -- the same percentiles per case and destination, with:

- `unfiltered_p05`/`unfiltered_p95`: the percentiles over every draw, rejected ones included;
- `step_p05_pct`/`step_p95_pct`: how much each endpoint moves from the first half of the draws
  to all of them;
- `scramble_p05_pct`/`scramble_p95_pct`: the gap between the two independent sequences.

**`acceptance.csv`** gives draws and accepted draws per case; **`REPORT.md`** summarizes the
main destinations. `cases/<case>/` holds each draw's allocation (`draw_NNNN.csv`), its sampled
sales factors (`draw_NNNN_factors.csv`), and its metadata and parameter values (`.json`).

## Reading the ranges

- **The conditional interval** covers the parameter draws under the adopted interpretation of
  the trade. It depends on the stated priors, and it is not a validated frequentist interval.
- **The envelope** spans the tested alternatives. No probabilities are attached to them, so it
  is not a 90% interval, and it does not exhaust every possible reading of ambiguous trade.
  Country endpoints come from different cases and are not additive.
- **Sample size.** With 64 draws per case each 5% tail rests on about three draws.
  `report_status.json` records the largest endpoint step and scramble gap; read the endpoints
  as approximate when these are large.
