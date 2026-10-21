"""Shared pieces of the raw -> clean parsers: the clean schema, raw-file selection, FX
conversion and output.

Clean schema
    Long format, one row per period x flow x partner x code:

    reporter        reporting book, lower-case key (e.g. ``hk``, ``eu``)
    flow            M import, X total export, RX re-export, DX domestic export (and RM
                    re-import, passed through from UN Comtrade where a reporter files it). Where
                    a book splits its exports, X == DX + RX: never add X to DX or RX.
    partner         partner country, English name
    code_native     code at the book's own granularity, a string (leading zeros matter). Books
                    that publish several granularities (4-digit parents, HS-6, national lines)
                    carry each as its own rows: filter on the length of code_native before
                    summing, never add rows across granularities.
    code_hs6        the first six digits of code_native. On 4-digit parent rows the books
                    differ: EU and Hong Kong leave it blank, while the US, Taiwan, Malaysia and
                    Thailand repeat the 4-digit code. The pipeline reads only rows whose
                    code_hs6 has six digits, so both conventions read the same.
    code_desc       short description
    period          YYYY-MM
    value_usd       value in US dollars (not thousands); blank where no exchange rate exists
    value_local     value as reported, in units of currency_local, for non-USD books
    currency_local  ISO currency code of value_local
    qty, qty_unit   quantity as reported, where the book carries one
    weight_kg       net weight, where the book carries one (the US reports gross shipping weight)
    source          short identifier of the publication
    pulled_date     YYYY-MM-DD the raw files were retrieved (each module states its rule)

    Fields a book does not publish are left blank, never imputed. A few books append extra
    columns after these; their modules document them.

Raw files and vintages
    Every raw file name carries the date it was retrieved as ``pulled<YYYYMMDD>`` (e.g.
    ``..._pulled20261010.json``): a retrieval, or vintage, is identified by that stamp. Two
    selection rules cover the books:

    * :func:`select_vintage` -- books whose every retrieval re-reads the whole series use the
      files of a single retrieval: the newest, or the one stamped ``vintage``.
    * :func:`raw_files` -- books that receive trailing-period refreshes merge every retrieval,
      the newest retrieval winning per record. The files are ranked newest first (same-day
      files by name, descending, so a trailing refresh such as ``..._2026-2026_pulled...``
      outranks the full retrieval ``..._2022-2026_pulled...`` of the same day); ``vintage``
      leaves out the retrievals made after that date, rebuilding the book as it stood then.
      :func:`keep_newest` applies the newest-wins rule to rows already read into one frame.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import paths

SCHEMA_COLS = [
    "reporter", "flow", "partner", "code_native", "code_hs6", "code_desc", "period",
    "value_usd", "value_local", "currency_local", "qty", "qty_unit", "weight_kg",
    "source", "pulled_date",
]

# Canonical row order of a clean file.
ROW_ORDER = ["flow", "code_native", "period", "partner"]

# Short labels of the basket's HS-6 codes and their 4-digit parents, used wherever a book's own
# description is unavailable or too long.
BASKET_DESC = {
    "847150": "ADP processing units (assembled servers, HGX/DGX boards)",
    "847330": "Parts & accessories of ADP machines (GPU cards/boards)",
    "847180": "Other ADP units",
    "847149": "ADP systems entered as such",
    "854231": "Processors & controllers (bare chips)",
    "854232": "Memory ICs",
    "854239": "Other electronic ICs",
    "851762": "Transmission/reception machines (switches)",
    "852351": "Solid-state non-volatile storage (NVMe SSD)",
    "8471": "ADP machines (4-digit parent)",
    "8473": "Parts of ADP machines (4-digit parent)",
    "8542": "Electronic integrated circuits (4-digit parent)",
}

_STAMP = re.compile(r"pulled(\d{8})")


@dataclass
class CleanBook:
    """One clean CSV produced by a parser: its rows and the file name it is written under."""

    frame: pd.DataFrame
    filename: str


def pulled_stamp(path: str | Path) -> str:
    """The ``YYYYMMDD`` retrieval stamp in a raw file name (``..._pulled20261010.json``)."""
    m = _STAMP.search(Path(path).name)
    if not m:
        raise ValueError(f"no pulled<YYYYMMDD> stamp in {path}")
    return m.group(1)


def iso_date(stamp: str) -> str:
    """``20261010`` -> ``2026-10-10``."""
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"


def _glob(raw_dir: Path, pattern: str | Iterable[str]) -> list[Path]:
    """Files in ``raw_dir`` matching one glob or any of several."""
    patterns = (pattern,) if isinstance(pattern, str) else tuple(pattern)
    return [p for pat in patterns for p in Path(raw_dir).glob(pat)]


def _describe(raw_dir: Path, pattern: str | Iterable[str]) -> str:
    """The patterns and directory searched, for error messages."""
    patterns = (pattern,) if isinstance(pattern, str) else tuple(pattern)
    return f"{' or '.join(patterns)} in {raw_dir}"


def raw_files(raw_dir: Path, pattern: str | Iterable[str],
              vintage: str | None = None) -> list[Path]:
    """Every raw file matching ``pattern`` (one glob or several), newest retrieval first.

    Files are ranked by their ``pulled<YYYYMMDD>`` stamp, descending; files retrieved the same
    day by file name, descending. With ``vintage`` (``YYYYMMDD``), files retrieved after that
    date are left out. Raises FileNotFoundError when nothing is left.
    """
    files = _glob(raw_dir, pattern)
    if vintage is not None:
        files = [p for p in files if pulled_stamp(p) <= vintage]
    if not files:
        until = f" retrieved on or before {vintage}" if vintage is not None else ""
        raise FileNotFoundError(f"no raw files{until} matching {_describe(raw_dir, pattern)}")
    return sorted(files, key=lambda p: (pulled_stamp(p), p.name), reverse=True)


def select_vintage(raw_dir: Path, pattern: str | Iterable[str],
                   vintage: str | None = None) -> list[Path]:
    """The raw files of one retrieval, in file-name order.

    That is the files sharing the newest ``pulled<YYYYMMDD>`` stamp or, with ``vintage``
    (``YYYYMMDD``), the files stamped exactly that day. Raises FileNotFoundError when there
    are none.
    """
    files = _glob(raw_dir, pattern)
    if files:
        stamp = vintage if vintage is not None else max(pulled_stamp(p) for p in files)
        files = [p for p in files if pulled_stamp(p) == stamp]
    if not files:
        pinned = f" stamped {vintage}" if vintage is not None else ""
        raise FileNotFoundError(f"no raw files{pinned} matching {_describe(raw_dir, pattern)}")
    return sorted(files)


def keep_newest(frame: pd.DataFrame, keys: list[str],
                tiebreak: Iterable[str] = ()) -> pd.DataFrame:
    """One row per ``keys``, taken from the newest retrieval.

    ``frame`` holds the rows of several raw files, appended file by file in file-name order,
    with each row's retrieval date in ``pulled_date``. Rows are ordered stably by
    ``pulled_date`` and then the ``tiebreak`` columns, and the last row of each key is kept: the
    newest retrieval wins; between files retrieved the same day, the higher tiebreak values,
    then the later file name.
    """
    order = ["pulled_date", *tiebreak]
    return frame.sort_values(order, kind="stable").drop_duplicates(keys, keep="last")


def finalize(frame: pd.DataFrame, columns: list[str] = SCHEMA_COLS,
             order: list[str] = ROW_ORDER) -> pd.DataFrame:
    """A clean frame in its written form: ``columns`` in order, codes as strings, rows sorted
    by ``order``."""
    frame = frame[list(columns)].copy()
    frame["code_native"] = frame["code_native"].astype(str)
    frame["code_hs6"] = frame["code_hs6"].astype(str)
    return frame.sort_values(list(order)).reset_index(drop=True)


def to_usd(values: Iterable, periods: Iterable[str], currency: str,
           fx_rates: dict[tuple[str, str], float]) -> list:
    """Local-currency values -> whole US dollars at each month's rate.

    ``fx_rates`` maps ``(currency, "YYYY-MM")`` to units of currency per USD (see
    :func:`load_fx_rates`). A month without a rate gives ``""``: value_usd is left blank
    rather than guessed.
    """
    rates = [fx_rates.get((currency, p)) for p in periods]
    return [int(round(v / r)) if r else "" for v, r in zip(values, rates)]


def year_span(periods: pd.Series) -> str:
    """``<first year>-<last year>`` of a column of ``YYYY-MM`` periods, for file names."""
    return f"{periods.min()[:4]}-{periods.max()[:4]}"


def load_fx_rates(path: Path = paths.FX_CSV) -> dict[tuple[str, str], float]:
    """Monthly period-average rates as ``{(currency, "YYYY-MM"): units of currency per USD}``.

    The table has columns ``currency_local, period, units_per_usd`` (IMF IFS period averages).
    A missing table yields no rates, so value_usd stays blank rather than guessed.
    """
    path = Path(path)
    if not path.exists():
        print(f"  [fx] no exchange-rate table at {path}; value_usd will be blank for "
              f"local-currency books")
        return {}
    fx = pd.read_csv(path, dtype={"period": str})
    return {(r.currency_local, r.period): float(r.units_per_usd) for r in fx.itertuples()}


def write_clean(book: CleanBook, clean_dir: Path) -> Path:
    """Write a parser's output as UTF-8 CSV under ``clean_dir``."""
    clean_dir = Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    out = clean_dir / book.filename
    book.frame.to_csv(out, index=False, encoding="utf-8")
    print(f"  wrote {out} ({len(book.frame):,} rows)")
    return out
