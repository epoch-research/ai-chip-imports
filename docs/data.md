# Data

This repository contains code only. The pipeline reads customs records and a few reference
datasets that you obtain yourself from their publishers, under each publisher's terms. This page
lists every input, where it comes from, and where the pipeline expects it.

## Layout

All inputs live under one data root, `data/` in the repository by default
(`AI_CHIP_IMPORTS_DATA` overrides it); outputs go to `output/` (`AI_CHIP_IMPORTS_OUTPUT`).

```
data/
├── customs/<book>/raw/      source files as retrieved, never edited
├── customs/<book>/clean/    standardized CSVs built by `python -m ai_chip_imports.sources`
├── external/                reference inputs (below)
└── fx/imf_fx_monthly.csv    monthly exchange rates
```

Every raw and clean file name carries its retrieval date as `pulled<YYYYMMDD>`; the pipeline
reads the newest file of each kind.

## Customs books

| Book (`data/customs/…`) | Publisher and publication | Code detail | Flows | Import valuation |
|---|---|---|---|---|
| `us` | US Census Bureau, international trade API | HS-10 | M, X = DX + RX | customs value (FOB-type) |
| `taiwan` | Taiwan Customs Administration, trade statistics portal | CCC 11-digit | M, X | CIF (imports) |
| `china` | General Administration of Customs (GACC) statistics portal | HS-8 | M, X | CIF |
| `eu` | Eurostat COMEXT (DS-045409), 27 members | CN8 | M, X | CIF |
| `uk` | HM Revenue & Customs, Overseas Trade Statistics | CN8 | M, X | CIF |
| `japan` | Ministry of Finance trade statistics (via e-Stat) | 9-digit | M, X | CIF |
| `korea` | Korea Customs Service trade statistics | HSK-10 | M, X | CIF |
| `hk` | Census and Statistics Department, Trade-IDDS | HKHS-8 | M, X = DX + RX | CIF |
| `singapore` | Department of Statistics, SingStat Table Builder | HS-8 | M, X = DX + RX | CIF |
| `malaysia` | Department of Statistics Malaysia, METS Online; plus Malaysia's UN Comtrade submission for physical fields | HS-10 | M, X = DX + RX | CIF |
| `mexico` | INEGI foreign-trade cube; plus Mexico's UN Comtrade submission for the remaining partners and 2022 | 10-digit | M, X | FOB |
| `thailand` | Ministry of Commerce trade statistics | AHTN 11-digit | M, X | CIF |
| `indonesia` | BPS-Statistics Indonesia | HS-8 | M, X | CIF |
| `india` | Department of Commerce, TRADESTAT | ITC-HS 8-digit | M, X | CIF |
| `saudi_arabia`, `australia`, `canada` | UN Comtrade monthly submissions | HS-6 | M, X (Canada: M, X = DX + RX) | Saudi Arabia, Australia CIF; Canada FOB |

Only the four counted codes and bare chips (`847149`, `847150`, `847180`, `847330`, `854231`)
are used by the estimator; the parsers also keep a few neighbouring codes the books were
collected with.

**Raw files.** The parsers read raw files in the archive format of the download scripts used to
build this dataset, which are not published. For most books that format is the publisher's
responses, kept verbatim, inside a small JSON envelope recording the request and the reference
tables the responses need; for Taiwan and China, files exported from the publishers' query
portals. The docstring of each module in `src/ai_chip_imports/sources/` specifies the exact layout
it expects, so the archives can be reproduced with any downloader.

**Building clean files.**

```bash
python -m ai_chip_imports.sources hk
python -m ai_chip_imports.sources comtrade --book canada
python -m ai_chip_imports.sources singapore --vintage 20260817
```

Books that receive trailing-period refreshes merge every retrieval in `raw/`, the newest winning
per record; the others read their newest retrieval. `--vintage` rebuilds a book as of a past
retrieval.

**Clean schema** (one row per period x flow x partner x code): `reporter`, `flow` (M import, X
total export, RX re-export, DX domestic export; X = DX + RX, so never add X to either),
`partner`, `code_native`, `code_hs6`, `code_desc`, `period` (YYYY-MM), `value_usd`,
`value_local`, `currency_local`, `qty`, `qty_unit`, `weight_kg`, `source`, `pulled_date`. Fields a
book does not publish are blank. Books carry their national lines next to their HS-6 lines; the
loader uses one granularity at a time.

## Reference inputs (`data/external/`)

| Path | Content | Built from |
|---|---|---|
| `ai_chip_sales/timelines_by_chip.csv`, `chip_types.csv` | Epoch AI, [AI Chip Sales](https://epoch.ai/data/ai-chip-sales) (CC BY): units, H100-equivalents and sales value per model and quarter | download as published |
| `baseline_index/` | UN Comtrade monthly imports by partner of the basket and of laptops/desktops (847130, 847141), one file per reporter (`<reporter>_comtrade_baseidx_*`) | `ai_chip_imports.sources.comtrade.build_index_pulls` |
| `baseline_index_alt/` | the same for monitors and printers (852851, 852852, 844331, 844332; `*_altidx_*`) | idem |
| `placebo_candidates/` | the same for input/output units (847160; `*_placebo2_*`) | idem |
| `us_cif/us_cif_charges_M_*.csv` | US imports by HS-6, country and month: customs value, CIF value, air and vessel charges and weights | `ai_chip_imports.sources.us.build_cif_charges` |
| `eurostat_prodcom_2620_sold_production.csv` | Eurostat PRODCOM (DS-059358) sold production of 26.20.14 and 26.20.15 per member and year: `geo, reporter, year, prccode, prodval_eur` | Eurostat |
| `ons_prodcom_2620_sold_production.csv` | ONS UK manufacturers' sales by product (ProdCom), same products: `geo, reporter, year, prccode, prodval_gbp` | ONS |
| `baseline_2021/` | the 2021 reference (below) | `ai_chip_imports.prepare` |

Index pulls cover reporters `us`, `china`, `eu`, `hong kong`, `singapore`, `malaysia`, `korea`,
`uk`, `japan`, `indonesia`, `india` from 2018, and `australia`, `norway`, `canada`,
`switzerland`, `iceland`, `israel`, `united arab emirates`, `saudi arabia`, `south africa`,
`new zealand`, `turkey`, `thailand`, `brazil` from 2021, where they file monthly data.

The UN Comtrade parsers keep only the total rows of each response (`motCode=0`,
`partner2Code=0`, `customsCode=C00`): the API also returns transport-mode and customs-procedure
breakdowns of the same trade, and summing everything roughly doubles it.

## The 2021 reference (`data/external/baseline_2021/`)

The non-AI reference is built on 2021 trade, before the national books in `data/customs/` begin.
Raw inputs:

| Path | Content |
|---|---|
| `raw/comtrade/<reporter>_2021.json` | UN Comtrade monthly records of 2021 for the measured books (reporter names with spaces use underscores), the four counted codes, flows M, X, RX, DX: the API's `{"data": [...]}` response; `austria_2021_annual.json` holds Austria's annual records (it filed no monthly data) |
| `raw/eurostat/<CC>_<1\|2>_2021.json` | Eurostat COMEXT JSON-stat responses, one per EU member (`CC`) and flow (1 import, 2 export), all partners, 2021 |
| `raw/hmrc/UK_2021.json` | HMRC OTS records of 2021 grouped by month, flow type, CN8 and partner (`MonthId`, `FlowTypeId`, `Commodity.Cn8Code`, `Country.CountryName`, `V`, `NM`, `SU`) |
| `fx_2021.csv` | 2021 monthly rates for EUR and GBP, same columns as the main FX table |
| `uk_index_native_raw.json`, `eu_index_exports_raw.json`, `uk_comtrade_origin_dispatch_2022.json` | the inputs of the UK index reconstruction (see `ai_chip_imports.prepare.uk_index`) |

Steps, in order:

```bash
python -m ai_chip_imports.prepare.books_2021      # -> comtrade_prepared.csv.gz, eu_native_prepared.csv.gz, uk_native_prepared.csv.gz
python -m ai_chip_imports.prepare.uk_index        # -> uk_index_2021.csv
python -m ai_chip_imports.prepare.history_2021    # -> measured_history.csv.gz (reads the clean books too)
```

The 2021 books are put on the FOB basis with the freight curve fitted on a fixed US charges file
(`us_cif_charges_M_2022-2026_pulled20260911.csv`) and with re-export shares and pooled densities
fixed at the values measured when the reference was prepared (constants in
`prepare/books_2021.py`, with functions that recompute them). This keeps the reference stable as
the national books are revised.

## Exchange rates (`data/fx/imf_fx_monthly.csv`)

Columns `currency_local, period, units_per_usd`: monthly period-average rates, local currency
per US dollar -- IMF International Financial Statistics for GBP, HKD, JPY, MYR and SGD, and ECB
reference rates (monthly average) for EUR. Parsers leave `value_usd` blank where a rate is
missing, never guessed.
