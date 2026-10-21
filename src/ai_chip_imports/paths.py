"""Filesystem layout of the pipeline's inputs and outputs.

Every path derives from two roots, each overridable by an environment variable:

    AI_CHIP_IMPORTS_DATA     input data root    (default: <repo>/data)
    AI_CHIP_IMPORTS_OUTPUT   output directory   (default: <repo>/output)

Under the data root (see docs/data.md for what each input is and where it comes from):

    customs/<book>/raw/      source files as received from the publisher, never edited
    customs/<book>/clean/    standardized CSVs built from raw/ by ``ai_chip_imports.sources``
    external/                non-book inputs: index pulls, the Epoch Chip Sales export,
                             US freight charges, production statistics, the 2021 reference
    fx/imf_fx_monthly.csv    monthly period-average exchange rates (local currency per USD)

Model assumptions that are part of the method rather than data (the captive-accelerator
geography table) ship with the package under ``ai_chip_imports/assumptions/``.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parents[1]

DATA_DIR = Path(os.environ.get("AI_CHIP_IMPORTS_DATA", REPO_DIR / "data"))
OUTPUT_DIR = Path(os.environ.get("AI_CHIP_IMPORTS_OUTPUT", REPO_DIR / "output"))

CUSTOMS_DIR = DATA_DIR / "customs"
EXTERNAL_DIR = DATA_DIR / "external"
FX_CSV = DATA_DIR / "fx" / "imf_fx_monthly.csv"
ASSUMPTIONS_DIR = PACKAGE_DIR / "assumptions"

# Every output file carries the run date: <name>_<YYYYMMDD>.csv. Readers take the newest stamp.
STAMP = datetime.date.today().strftime("%Y%m%d")


def raw_dir(book: str) -> Path:
    """Directory holding a customs book's source files, e.g. ``raw_dir("hk")``."""
    return CUSTOMS_DIR / book / "raw"


def clean_dir(book: str) -> Path:
    """Directory holding a customs book's standardized CSVs."""
    return CUSTOMS_DIR / book / "clean"


def output_path(name: str, suffix: str = "csv") -> Path:
    """Dated output file for this run, e.g. ``output_path("gpu_country_flows")``."""
    return OUTPUT_DIR / f"{name}_{STAMP}.{suffix}"
