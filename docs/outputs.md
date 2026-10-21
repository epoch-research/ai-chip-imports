# Outputs

`python -m ai_chip_imports` writes every output to `output/` (or `AI_CHIP_IMPORTS_OUTPUT`) as
`<name>_<YYYYMMDD>.csv`, stamped with the run date. Quarters (`q`) are calendar arrival
quarters: customs records are summed by month of arrival, and Epoch's sales are moved to the
months they arrive in (see the [methodology page](https://epoch.ai/data/ai-chip-imports-documentation/methodology)). Dollar columns are US dollars on the
FOB basis. H100e is H100-equivalent compute as defined in Epoch's Chip Sales data.

Every allocator output carries the window columns `arrival_quarter`, `arrival_window_start`,
`arrival_window_end` (the arrival months) and `sales_window_start`, `sales_window_end` (the sales
months that arrive in them); most also carry `partial_sales_coverage` (the quarter draws on an
incomplete Epoch sales quarter), `shares_window_end` and `timing_basis`.

## Allocation

**`gpu_country_flows`** -- arrivals per destination x quarter x model.

| Column | Meaning |
|---|---|
| `dest`, `tier` | destination; `measured` (own customs book) or `thin` (reached through other books' exports) |
| `model`, `designer`, `pool` | Epoch model, its designer, and the allocation pool (`cn_domestic`, `cn_spec`, `row`) |
| `dest_share` | the destination's share of the model's units in the quarter |
| `units`, `h100e`, `usd` | units, H100e and sales value allocated |
| `units_lo`, `units_hi`, `h100e_lo`, `h100e_hi` | per-cell assumption envelope; **not additive across models** -- use `gpu_country_bands` for destination totals |
| `provisional` | the latest customs quarter, or one drawing on incomplete sales |
| `excess_z` | the destination's excess in units of its own pre-2023Q3 noise (reported, never enforced) |
| `share_carried_forward` | the destination's books stop before the quarter ends; it keeps its last complete quarter's share |

**`gpu_country_stocks`** -- cumulative arrivals since 2022 per destination x model x quarter
(no retirement). `units_cum`, `h100e_cum`, `usd_cum` and their `_lo`/`_hi` envelopes (each band
variant is cumulated first, then the envelope taken; not additive across models);
`excess_z_cum` and `evidence` (`weak` < 1, `moderate` 1-3, `strong` >= 3).

**`gpu_country_bands`** -- destination totals per quarter, with coherent bands: each band
variant's destination total at Epoch's 5th/95th percentiles, then the minimum and maximum across
variants. Flows (`units`, `h100e` and `_lo`/`_hi`) and cumulative stocks (`units_cum`,
`h100e_cum` and `_lo`/`_hi`). Rows with `dest = World` reproduce Epoch's own totals and
percentiles. These are assumption envelopes, not probability intervals; the conditional intervals
and the structural envelope come from the uncertainty run ([uncertainty.md](uncertainty.md)).

**`gpu_country_shares`** -- every share set behind the allocation: `variant`, `in_ci` (part of
the band), `q`, `dest`, `D_usd` (chip dollars) and `share`. The `headline|smoothshares` variant
is a timing sensitivity outside the band.

**`gpu_country_eu_members`** -- an indicative split of the EU's arrivals among members by their
gross extra-EU imports: `member`, `share`, `units`, `h100e`, cumulative columns, and `nl_flag`
(the Netherlands, a distribution hub, is an upper bound).

**`gpu_country_origin_mix`** -- a diagnostic: for each measured destination, the share of its
chip dollars by origin bucket (`taiwan`, `mexico_assembly`, `vietnam_assembly`,
`thailand_assembly`, `united_states`, `china_origin`, `other`) over the AI window.

**`gpu_country_imputation`** -- per destination x quarter, the chip dollars and the part that
depends on imputed exporter content (`imputation_dependent_dollars`, `fraction`). A measure of
dependence on imputation, not an uncertainty band.

## Estimator

**`gpu_finalprice_quarterly`** -- per calendar quarter: `epoch_sellin` (Epoch's sales of the
Taiwan-fabricated designers, by sales quarter), `approach2_corrected` (customs chip dollars, by
arrival quarter) and `epoch_arrivals_interpolated` (Epoch's sales moved to arrival quarters).

**`gpu_epoch_match`** -- per arrival quarter of the AI window: `epoch_arrivals`,
`matched_customs` and `match_pct` (customs as a percentage of Epoch).

**`gpu_finalprice_band`** -- the median quarterly ratio of customs to Epoch for each band
combination: `baseline` (`indexed`, `grown+5%`, or a Malaysia mirror endpoint), `growth`,
`content` (`corridor` = measured, `mfg` = flat per-code constants, `high085` = 0.85 on systems),
`median_ratio` (same quarter) and `median_ratio_lag` (arrival quarters).

**`gpu_finalprice_dest`** -- per destination x code x quarter: counted `excess` and chip
dollars (`chiponly_corr`).

**`gpu_content_cells`** -- the evidence behind the content readings: one row per origin slice
(`source = qty`), native tariff line (`native`) or classifier cell (`cell`), with value,
quantity, excess, unit value or density (`uv`, `uv_basis`), the reference price, the cell's
content (`c_grp`), the line's raw and smoothed content, and `measured_cell` (whether the quarter
is served by a measured reading).

**`gpu_content_monthly`** -- the monthly content readings in use (`dest`, `code_hs6`,
`period`, `content`).

**`gpu_exporter_content`**, **`gpu_exporter_imputation`** -- the classified export rows behind
the thin tier (field used, coverage, observed and imputed content with low/high donor
scenarios), and per destination x code x month the share of trade and chip dollars resting on
imputed content.

**`gpu_run_manifest_<stamp>.json`** -- provenance: git revision, parameters, the FOB curve, the
re-export shares, the per-book netting ledger, the classified content pairs, and the allocation
variants.
