# AI chip imports

Code behind Epoch AI's [AI Chip Imports](https://epoch.ai/data/ai-chip-imports) data: estimates
of where AI accelerators arrive, built from national customs records and Epoch AI's
[AI Chip Sales](https://epoch.ai/data/ai-chip-sales) data.

The pipeline isolates the AI-attributable slice of customs trade in servers and accelerator
boards for sixteen measured destinations (and, through their exports, every other country),
converts it into accelerator dollars, checks the total against Epoch's sales, and distributes
Epoch's per-model sales across destinations in proportion to those dollars. The method is
documented on the [methodology page](https://epoch.ai/data/ai-chip-imports-documentation/methodology); module docstrings give the
details of each step.

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

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AI_CHIP_IMPORTS_DATA` | `./data` | input data root |
| `AI_CHIP_IMPORTS_OUTPUT` | `./output` | output directory |
| `AI_CHIP_IMPORTS_LAG_MONTHS` | `2` | months between a sale and the border crossing (0-6) |
| `AI_CHIP_IMPORTS_NETTING_QUARTERS` | `2` | memory of the excess ledger, in quarters (`inf` for none) |
| `AI_CHIP_IMPORTS_MALAYSIA_MIRROR_SHARE` | `0.5` | share of the China-Malaysia server mirror gap deducted from Malaysia |

The model's assumptions are constants in the modules that use them, each documented where it
is defined; the captive-accelerator geography is a table in
`src/ai_chip_imports/assumptions/captive_geography.csv`.

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
└── compare.py          output comparison between runs
tests/                  unit tests (synthetic data)
```

Run the tests with `pip install -e ".[dev]" && pytest`.

## License

The code is released under the [MIT License](LICENSE). The data it reads remains subject to each
publisher's terms; Epoch AI's Chip Sales data is licensed CC BY.
