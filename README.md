# AI Chip Imports

Code behind Epoch AI's [AI Chip Imports](https://epoch.ai/data/ai-chip-imports) explorer, which
estimates how much AI compute arrives in each country every quarter since 2022.

Almost every advanced AI chip is packaged in Taiwan, so nearly every chip crosses at least one
border before reaching a data center, and most countries publish monthly records of what crosses
their borders. We combine these customs records with our
[AI Chip Sales](https://epoch.ai/data/ai-chip-sales) estimates in four steps:

1. **Hardware retained in each country.** We count hardware only in the country where it stays,
   subtracting re-exports and the imported content of servers a country assembles and ships on.
2. **AI-related trade.** We estimate AI-related trade as the excess above a baseline built from
   each country's 2021 trade.
3. **Accelerator share of trade value.** We estimate how much of the remaining value is the AI
   chip itself, rather than CPUs, memory, networking or chassis, using national tariff
   descriptions and declared value per unit and per kilogram.
4. **Allocation of chips to countries.** We split our chip sales estimates, by chip model and
   quarter, across countries in proportion to their chip dollars, assuming chips arrive about two
   months after they are sold.

We collect customs data for 16 destinations with their own records; countries without one are
estimated from what other countries declare shipping to them. The
[methodology page](https://epoch.ai/data/ai-chip-imports-documentation/methodology) describes
each step, and the module docstrings document how the code implements it.

This repository contains code only. It does not include customs data, the scripts used to
download it, or any outputs: obtain the inputs from their publishers, under their terms
([docs/data.md](docs/data.md) lists them and the layout the code expects).

## Quick start

Requires Python 3.12 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

1. **Place the inputs** under `data/` as described in [docs/data.md](docs/data.md): the raw
   customs files of each book, the Epoch Chip Sales export, the UN Comtrade index pulls, the US
   freight-charge records, the production statistics and the exchange-rate table.
2. **Build the clean customs books** from the raw files:

   ```bash
   for book in us taiwan china eu uk japan korea hk singapore malaysia mexico thailand indonesia india; do
       python -m ai_chip_imports.sources $book
   done
   for book in saudi_arabia australia canada malaysia mexico; do
       python -m ai_chip_imports.sources comtrade --book $book
   done
   ```

3. **Prepare the 2021 reference inputs** (see [docs/data.md](docs/data.md)):

   ```bash
   python -m ai_chip_imports.prepare.books_2021
   python -m ai_chip_imports.prepare.uk_index
   python -m ai_chip_imports.prepare.history_2021
   ```

4. **Run the pipeline**:

   ```bash
   python -m ai_chip_imports            # estimator + allocator; python -m ai_chip_imports estimate for the first half
   ```

   Outputs land in `output/` as `<name>_<YYYYMMDD>.csv`; [docs/outputs.md](docs/outputs.md)
   describes each file. A full run takes a few minutes.

## Main assumptions

The adopted assumptions are constants in the modules that use them, each documented where it is
defined. These are the ones the results depend on most, roughly in order (the methodology page's
uncertainty section quantifies them):

| Assumption | Adopted value | Where |
|---|---|---|
| Accelerator content of ambiguous trade (boards and parts in 8473.30, other units in 8471.80) | unit-value ladder and $/kg ramps, national tariff lines first | `content.py`: `LOW_MAX`…`SERVER_LO`, `CELL_DENS`, `TIER_C` |
| Ordinary-trade baseline | 2021 trade, indexed by client IT for systems and by a non-AI-origin component index for boards | `baseline.py`, `history.py`; `AI_CHIP_IMPORTS_REFERENCE_YEAR` (2022: the earlier 2022 reference) |
| Imported content of re-exported servers (k) | 0.10 / 0.40 / 0.90 at 250 / 600 / 2,000 $/kg | `netting.py`: `K_DENS_ANCH`, `K_UV_ANCH` |
| Sales-to-arrival lag | 2 months | `timing.py`; `AI_CHIP_IMPORTS_LAG_MONTHS` |
| Malaysia server-gap deduction | 50% of the China–Malaysia server mirror gap | `netting.py`; `AI_CHIP_IMPORTS_MALAYSIA_MIRROR_SHARE` |
| China price premium on diverted chips | 1.13–2.35× by quarter from 2023Q4 | `allocator.py`: `PREMIUM_SERIES` |
| Where captive chips (TPU, Trainium) are deployed | destination countries per chip model | `assumptions/captive_geography.csv` |
| Netting memory | 2 quarters | `ledger.py`; `AI_CHIP_IMPORTS_NETTING_QUARTERS` |

The four environment variables override their defaults for sensitivity runs; the defaults are
the published method. Two more set the paths: `AI_CHIP_IMPORTS_DATA` (input root, default
`./data`) and `AI_CHIP_IMPORTS_OUTPUT` (default `./output`).

## Uncertainty

`ai_chip_imports.uncertainty` computes the ranges published with the estimates. It reruns the
whole pipeline under random draws of the parameters and of Epoch's sales, within 144 structural
cases that read the trade differently. The conditional 90% interval comes from the central
case; the structural envelope spans every case.

```bash
python -m ai_chip_imports.uncertainty prepare
python -m ai_chip_imports.uncertainty probe
python -m ai_chip_imports.uncertainty run --samples 64 --workers 6    # about a day on six cores
python -m ai_chip_imports.uncertainty report
```

[docs/uncertainty.md](docs/uncertainty.md) describes the run, its outputs and how to read them.

## Reproducibility

The pipeline is deterministic: the same inputs and package versions reproduce every output
byte for byte (`requirements.txt` pins the versions used for published runs). To check a run
against a reference run:

```bash
python -m ai_chip_imports.compare path/to/reference_output output
```

## Layout

```
src/ai_chip_imports/
├── sources/            raw customs files -> clean CSVs, one parser per publication
├── prepare/            derived 2021 reference inputs
├── books.py            loading the clean books at HS-6
├── fob.py              CIF import books -> FOB
├── netting.py          the single-counted destination universe
├── baseline.py         the measured non-AI index
├── history.py          the 2021 reference and warm-up
├── ledger.py           excess over the reference and the netting ledger
├── content.py          accelerator content from the books' own physical fields
├── exporter_content.py content for destinations without an import book
├── epoch.py            Epoch's Chip Sales data
├── timing.py           arrival-quarter conventions
├── estimator.py        chip dollars per destination, compared with Epoch
├── allocator.py        Epoch's units distributed across destinations
├── compare.py          output comparison between runs
└── uncertainty/        parameter and sales draws within structural cases
tests/                  unit tests (synthetic data)
```

Run the tests with `pip install -e ".[dev]" && pytest`.

## License

The code is released under the [MIT License](LICENSE). The data it reads remains subject to each
publisher's terms; Epoch AI's Chip Sales data is licensed CC BY.
