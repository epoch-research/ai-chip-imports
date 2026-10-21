"""Command line for the raw -> clean parsers; see ``ai_chip_imports.sources``."""
from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path

from .. import paths
from . import PARSERS
from ._common import load_fx_rates, write_clean


def _stamp(text: str) -> str:
    """argparse type for ``--vintage``: a ``YYYYMMDD`` retrieval stamp."""
    if not re.fullmatch(r"\d{8}", text):
        raise argparse.ArgumentTypeError(f"expected YYYYMMDD, got {text!r}")
    return text


def main(argv=None) -> None:
    """Build one parser's clean CSVs; ``argv`` defaults to the command line."""
    ap = argparse.ArgumentParser(prog="python -m ai_chip_imports.sources",
                                 description="Build clean CSVs from raw customs files.")
    ap.add_argument("parser", choices=PARSERS)
    ap.add_argument("--book", help="book directory under data/customs/ (default: the parser's "
                                   "own; required for comtrade)")
    ap.add_argument("--raw-dir", type=Path, help="override data/customs/<book>/raw")
    ap.add_argument("--clean-dir", type=Path, help="override data/customs/<book>/clean")
    ap.add_argument("--fx", type=Path, default=paths.FX_CSV, help="exchange-rate table")
    ap.add_argument("--vintage", type=_stamp, metavar="YYYYMMDD",
                    help="retrieval to build: pins that retrieval for single-retrieval books; "
                         "ignores later retrievals for books that merge refreshes")
    args = ap.parse_args(argv)

    module = importlib.import_module(f"{__package__}.{args.parser}")
    book = args.book or module.BOOK
    if book is None:
        ap.error(f"{args.parser} serves several books ({', '.join(module.BOOKS)}); "
                 f"pass --book")
    raw = args.raw_dir or paths.raw_dir(book)
    clean = args.clean_dir or paths.clean_dir(book)
    for out in module.build(raw, load_fx_rates(args.fx), vintage=args.vintage):
        write_clean(out, clean)


if __name__ == "__main__":
    main()
