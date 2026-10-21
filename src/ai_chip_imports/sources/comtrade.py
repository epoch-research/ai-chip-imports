"""UN Comtrade: monthly commodity trade by reporting country (the books in ``BOOKS``).

Publication
    UN Comtrade (United Nations Statistics Division): monthly merchandise trade by HS-6
    commodity and partner, as submitted by each reporting country.

Books
    The parser serves every book whose customs data come from Comtrade rather than a national
    portal (Saudi Arabia, Australia, Canada), and the Comtrade companions of national books
    (Malaysia, Mexico) that supply partners or physical fields the national release lacks.
    Since no single book is the natural default, ``BOOK`` is None and the command line
    requires ``--book``; ``BOOKS`` lists the books. A book's raw directory may also hold the
    national book's files, which the file pattern below leaves alone. The same files, pulled
    for other code sets, feed the non-AI baseline indices: see :func:`build_index_pulls`.

Codes
    Comtrade trims national detail to six digits, so ``code_native`` is the HS-6 code and
    ``code_hs6`` repeats it. ``code_desc`` is the short basket label (blank for index codes
    without one).

Flows
    Comtrade flow codes map one to one: ``M`` import, ``X`` export, ``RX`` re-export, ``DX``
    domestic export, ``RM`` re-import. Where a reporter files the split, X == DX + RX.

Partners
    English names from Comtrade's partner-area reference. World (code 0) and any partner code
    listed as a group are dropped. Each cell is taken from its total record only (see
    :func:`is_total_row`).

Values
    US dollars (Comtrade's ``primaryValue``, which UNSD converts from the reporter's
    currency), rounded to the dollar; ``fx_rates`` is unused and ``value_local``/
    ``currency_local`` stay blank. ``primaryValue`` is the CIF value for imports where the
    reporter supplies one and FOB otherwise (Mexico and Canada compile imports FOB); exports
    are FOB.

Quantity and weight
    ``qty``/``qty_unit`` (``qtyUnitAbbr``) and ``weight_kg`` (net weight, ``netWgt``) as filed
    by the reporter. Where a reporter files none, UNSD imputes them from the value at a
    standard unit value and flags them (``isQtyEstimated``/``isNetWgtEstimated``); imputed
    figures are not data and are left blank, as are zeros. ``keep_imputed=True`` keeps them
    instead.

Raw files
    ``<reporter>_comtrade_<tag>_<flows>_hs6_<first>-<last>_pulled<YYYYMMDD>.json``, one per
    retrieval of one reporter (e.g.
    ``canada_comtrade_T1-T2_MXRXDX_hs6_2022-2026_pulled20260916.json``): a JSON envelope
    holding

    * ``rows``: the data records of Comtrade's monthly HS responses, verbatim (one object per
      record, Comtrade's own field names: ``period``, ``flowCode``, ``partnerCode``,
      ``partner2Code``, ``cmdCode``, ``customsCode``, ``motCode``, ``primaryValue``, ``qty``,
      ``qtyUnitAbbr``, ``isQtyEstimated``, ``netWgt``, ``isNetWgtEstimated``, ...);
    * ``partner_names``: ``{partner code: English name}`` from Comtrade's partner-area
      reference table, and ``groups``: the partner codes to treat as aggregates (World = 0);
    * ``reporter`` (lower-case name, e.g. ``saudi arabia``), ``reporter_code`` (UN M49),
      ``flows`` (e.g. ``M,X``), ``codes``, ``tag`` (the code set: ``T1-T2`` for the basket)
      and ``pulled_date``.

Vintages
    Every retrieval re-reads the whole series, so ``build`` reads the files of the newest
    retrieval only (or of the one stamped ``vintage``), one clean file per raw file. The
    retrieval is chosen by the ``pulled<YYYYMMDD>`` stamp, never by file-name order: a book can
    hold pulls of different flow sets (``..._MX_...`` and ``..._MXRXDX_...``) whose names do
    not sort by date.

Output
    ``<reporter>_comtrade_<tag>_<flows>_<first>-<last>_pulled<YYYYMMDD>.csv`` (spaces in the
    reporter name become underscores, commas in the flows are dropped; the years are those
    of the data). ``pulled_date`` is the raw file's ``pulled<YYYYMMDD>`` stamp.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .. import paths
from ._common import (
    BASKET_DESC,
    SCHEMA_COLS,
    CleanBook,
    finalize,
    iso_date,
    pulled_stamp,
    raw_files,
    select_vintage,
    year_span,
)

# No default book: the command line requires --book, one of BOOKS.
BOOK = None
BOOKS = ("saudi_arabia", "australia", "canada", "malaysia", "mexico")

SOURCE = "comtrade"
RAW_PATTERN = "*_comtrade_*_pulled*.json"

# The index pulls behind the non-AI baseline: raw JSON and clean CSV side by side.
INDEX_DIRS = tuple(paths.EXTERNAL_DIR / name
                   for name in ("baseline_index", "baseline_index_alt", "placebo_candidates"))

# The basket labels, plus the two codes of the non-AI baseline index that have one. Other
# index codes (monitors, printers, input/output units) are written with a blank description.
CODE_DESC = {
    **BASKET_DESC,
    "847130": "Portable ADP machines <=10 kg (laptops)",
    "847141": "Other ADP machines with CPU + input/output in one housing (desktops)",
}

# Comtrade flowCode -> schema flow; other flows are skipped.
FLOW_MAP = {"M": "M", "X": "X", "RX": "RX", "RM": "RM", "DX": "DX"}

# The series a raw file belongs to: reporter and code set (``canada_comtrade_baseidx``).
_SERIES = re.compile(r"^(?P<series>.+?_comtrade_[^_]+)_")


def is_total_row(row: dict) -> bool:
    """True for the record that carries a cell's total; False for its breakdown rows.

    Comtrade publishes each period x flow x partner x commodity cell as a total record plus
    breakdowns of the same trade by mode of transport (``motCode``), second partner
    (``partner2Code``, e.g. the country of consignment) and customs procedure
    (``customsCode``). The breakdowns add up to the total, so summing every record counts the
    same trade about three times. The total is the record with all three dimensions at their
    "all" value: motCode 0 (all modes), partner2Code 0 (World) and customsCode C00 (all
    procedures).
    """
    return (row.get("motCode") in (0, "0") and row.get("partner2Code") in (0, "0")
            and str(row.get("customsCode")) == "C00")


def standardize(rows: list[dict], reporter: str, partner_names: dict[int, str],
                groups: set[int], pulled_date: str, keep_imputed: bool = False) -> pd.DataFrame:
    """Map Comtrade records to the clean schema, keeping total records with a country partner.

    UNSD-imputed quantities and weights are blanked unless ``keep_imputed``.
    """
    partner_names = partner_names or {}
    groups = groups or {0}
    out = []
    for d in rows:
        if not is_total_row(d):
            continue
        flow = FLOW_MAP.get(str(d.get("flowCode")))
        if not flow:
            continue
        try:
            pcode = int(d.get("partnerCode"))
        except (TypeError, ValueError):
            continue
        if pcode in groups:
            continue
        code = str(d.get("cmdCode"))
        period = str(d.get("period"))
        value = d.get("primaryValue")
        qty_ok = (d.get("qty") not in (None, 0)
                  and (keep_imputed or not d.get("isQtyEstimated")))
        wgt_ok = (d.get("netWgt") not in (None, 0)
                  and (keep_imputed or not d.get("isNetWgtEstimated")))
        out.append({
            "reporter": reporter, "flow": flow,
            "partner": partner_names.get(pcode, str(pcode)),
            "code_native": code, "code_hs6": code[:6],
            "code_desc": CODE_DESC.get(code, ""), "period": f"{period[:4]}-{period[4:6]}",
            "value_usd": "" if value is None else int(round(float(value))),
            "value_local": "", "currency_local": "",
            "qty": d.get("qty") if qty_ok else "",
            "qty_unit": (d.get("qtyUnitAbbr") or "") if qty_ok else "",
            "weight_kg": d.get("netWgt") if wgt_ok else "",
            "source": SOURCE, "pulled_date": pulled_date,
        })
    if not out:
        return pd.DataFrame(columns=SCHEMA_COLS)
    return finalize(pd.DataFrame(out))


def build_pull(raw_json: Path, keep_imputed: bool = False) -> CleanBook:
    """One archived Comtrade retrieval -> its clean CSV (frame and file name)."""
    raw_json = Path(raw_json)
    with open(raw_json, encoding="utf-8") as fh:
        raw = json.load(fh)
    stamp = pulled_stamp(raw_json)
    reporter = raw["reporter"]
    df = standardize(raw["rows"], reporter,
                     {int(k): v for k, v in raw["partner_names"].items()},
                     set(raw["groups"]), iso_date(stamp), keep_imputed)
    # An empty pull is named after the default start year.
    coverage = year_span(df["period"]) if not df.empty else "2022-2022"
    flows = raw.get("flows", "M,X").replace(",", "")
    tag = raw.get("tag", "T1-T2")
    name = f"{reporter.replace(' ', '_')}_comtrade_{tag}_{flows}_{coverage}_pulled{stamp}.csv"
    return CleanBook(df, name)


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """Clean CSVs for every Comtrade file of the newest retrieval, or of the one stamped
    ``vintage``. ``fx_rates`` is unused: Comtrade values are in US dollars."""
    books = [build_pull(p) for p in select_vintage(raw_dir, RAW_PATTERN, vintage)]
    for book in books:
        print(f"  comtrade: {len(book.frame):,} rows -> {book.filename}")
    return books


def build_index_pulls(index_dir: Path, keep_imputed: bool = False) -> list[CleanBook]:
    """Clean CSVs for an index-pull directory, from the newest pull of each reporter and
    code set.

    The baseline-index directories (``INDEX_DIRS``) hold the index pulls, raw JSON and clean
    CSV side by side; write the result back into ``index_dir``. The default rule blanks
    Comtrade-imputed quantities and weights, as in the books. Older archived index pulls were
    cleaned keeping the imputed figures (``keep_imputed=True`` rebuilds those); the pipeline
    reads only ``value_usd`` from index pulls, so the choice never affects results.
    """
    latest: dict[str, Path] = {}
    for path in raw_files(index_dir, RAW_PATTERN):         # newest first
        m = _SERIES.match(path.name)
        latest.setdefault(m.group("series") if m else path.name, path)
    return [build_pull(p, keep_imputed) for _, p in sorted(latest.items())]
