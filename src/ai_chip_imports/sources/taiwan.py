"""Taiwan: Ministry of Finance customs trade statistics (book ``taiwan``).

Publication
    Customs Administration, Ministry of Finance (Taiwan), monthly trade statistics by
    commodity and country (area), exported as ``.xls`` from the publisher's trade-statistics
    query portal (English interface). One export holds one flow, all partner countries,
    monthly rows for every requested commodity code.

Codes
    Taiwan's CCC (Customs Commodity Code, the national HS extension). The book mixes three
    granularities, each as published: the 4-digit parents 8471/8473/8542 (context only),
    the HS-6 basket, and the 11-digit CCC lines (10 digits plus a computed check digit).
    ``code_native`` is the code as published; ``code_hs6`` is its first six characters,
    which on the 4-digit parent rows is the 4-digit code itself. The granularities overlap:
    filter by code length, never sum across them. HS-6 and 4-digit rows carry the short
    basket label; 11-digit rows keep the publisher's official description.

Flows
    "Imports" -> M, "Exports" -> X (total exports; no domestic/re-export split is
    published in these tables).

Partners
    Country (area) names as published, except a few comma-inverted formal names mapped to
    the short English names the other books use.

Values
    ``Value(USD$ 1,000)`` is in thousands of US dollars and is multiplied by 1,000 into
    ``value_usd``. A file exported in NT dollars only (``Value(NTD$ 1,000)``) fills
    ``value_local`` in TWD, likewise times 1,000, and leaves ``value_usd`` blank: no exchange
    rate is applied, so ``fx_rates`` is unused. Imports are valued CIF, exports FOB.

Quantity and weight
    ``Weight(KGM)`` is net weight in kg; when only ``Weight(TNE)`` is present it is converted
    from tonnes. ``Quantity`` and ``Unit of Quantity`` (SET, PCE, ...) are published for
    11-digit codes only; codes without a natural count publish "-" for both, which is left
    blank.

Raw files
    The portal's exports saved as downloaded (BIFF ``.xls``, read with xlrd), any number of
    them, each named with a ``_pulled<YYYYMMDD>`` retrieval stamp (files without one are not
    read), e.g. ``taiwan_portal_T1-T2_exports_hs6_2024-2025_pulled20260708.xls``. Content
    (flow, granularity, value and weight fields) is read from each file's headers and rows,
    not its name; columns are matched by header text because their order and the optional
    weight/quantity/NTD columns vary with the query. ``Time`` reads ``YYYY/M``; the latest
    month carries a ``(preliminary)`` suffix and is revised in later releases. Preliminary
    months are parsed like any other and listed in the build summary. Exports of other portal
    tables that may sit alongside are reported and skipped.

Vintages
    Exports usually come as yearly files that overlap by a month, and later retrievals revise
    recent months, so ``build`` merges every file: where the same flow x code x partner x
    month (the merge key) appears more than once, the newest retrieval wins; between files
    retrieved the same day, the row carrying weight wins, then the later file name.
    ``vintage="YYYYMMDD"`` leaves out the files retrieved after that date.

Output
    Two books, split by month:
      ``taiwan_T1-T2_<flows>_<first year>-<last year>_pulled<stamp>.csv``
          the monthly series from January 2022 onward;
      ``taiwan_history_T1_<flows>_<year(s)>_pulled<stamp>.csv``
          earlier months: the 2021 reference year, retrieved separately for exports of the
          four computer codes (847149, 847150, 847180, 847330; ``T1`` in the file names) and
          kept apart so the main series keeps its 2022 start.
    ``<stamp>`` is the newest retrieval date among each book's rows; ``<flows>`` concatenates
    the flows present (``MX``, ``X``, ...). ``pulled_date`` on each row is the retrieval
    stamp of the file the row came from.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from ._common import (
    BASKET_DESC,
    SCHEMA_COLS,
    CleanBook,
    finalize,
    iso_date,
    keep_newest,
    pulled_stamp,
    raw_files,
)

BOOK = "taiwan"
REPORTER = "taiwan"
SOURCE = "mof_portal"
RAW_PATTERN = "*_pulled*.xls"

# Months before this one form the separate reference-year (history) book.
HISTORY_BEFORE = "2022-01"

# The portal lists a few countries under comma-inverted formal names; map them to the short
# English names used across the other books. Everything else passes through as published.
PARTNER_MAP = {
    "Arab Emirates, United": "United Arab Emirates",
    "Congo, Democratic Republic of": "Democratic Republic of the Congo",
    "Korea, Republic of": "South Korea",
    "Moldova, Republic of": "Moldova",
    "Palestine, State of": "Palestine",
    "Taiwan, ROC": "Taiwan",
    "Tanzania, United Republic of": "Tanzania",
    "Viet Nam": "Vietnam",
    "Virgin Islands, British": "British Virgin Islands",
}

# A clean row's identity, the key on which retrievals are merged.
_KEY = ["flow", "code_native", "partner", "period"]


def _find_col(cols, *keywords):
    """First column whose header contains any of ``keywords`` (case-insensitive), else None."""
    for kw in keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                return c
    return None


def parse_period(raw_time) -> tuple[str, bool]:
    """``"2026/5(preliminary)"`` -> ``("2026-05", True)``; ``("", flag)`` if unparsable."""
    s = str(raw_time)
    prelim = "prelim" in s.lower()
    m = re.match(r"\s*(\d{4})\s*/\s*(\d{1,2})", s)
    if not m:
        return "", prelim
    return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}", prelim


def read_raw_file(path: Path) -> tuple[list[dict], set[str]]:
    """Clean-schema rows of one portal export, and the months it marks preliminary.

    Raises ValueError when the sheet lacks the flow, time, code, partner or value columns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # xlrd style chatter on the portal's .xls files
        df = pd.read_excel(path)
    cols = list(df.columns)
    c_flow = _find_col(cols, "Imports/Exports", "import")
    c_time = _find_col(cols, "Time")
    c_code = _find_col(cols, "Commodity Code", "commodity")
    c_desc = _find_col(cols, "Description")
    c_ctry = _find_col(cols, "Country", "Area")
    c_usd = _find_col(cols, "USD")
    c_ntd = _find_col(cols, "NTD", "NT$")
    c_kgm = _find_col(cols, "KGM", "KG)", "(KG")
    c_tne = _find_col(cols, "TNE", "Metric Ton", "TON")
    # "Quantity" and "Unit of Quantity" both contain "quantity", so match each explicitly.
    c_qty = next((c for c in cols
                  if "quantity" in str(c).lower() and "unit" not in str(c).lower()), None)
    c_unit = _find_col(cols, "Unit of Quantity", "Unit")

    missing = [n for n, c in [("flow", c_flow), ("time", c_time), ("code", c_code),
                              ("country", c_ctry)] if c is None]
    if missing or (c_usd is None and c_ntd is None):
        raise ValueError(f"unrecognized layout (missing {missing or 'value column'}): {cols}")

    pulled = iso_date(pulled_stamp(path))
    rows, prelim_periods = [], set()

    for rec in df.to_dict("records"):
        flow_raw = str(rec.get(c_flow, "")).strip().lower()
        if flow_raw.startswith("import"):
            flow = "M"
        elif flow_raw.startswith("export"):
            flow = "X"
        else:
            continue

        period, is_prelim = parse_period(rec.get(c_time))
        if not period:
            continue
        if is_prelim:
            prelim_periods.add(period)

        # Codes can arrive as numbers (847150.0); keep them as digit strings.
        code = str(rec.get(c_code, "")).split(".")[0].strip()
        if not code or code.lower() == "nan":
            continue
        code_hs6 = code[:6]

        if len(code) > 6:  # national 11-digit line
            desc = ((str(rec.get(c_desc, "")).strip() if c_desc else "")
                    or BASKET_DESC.get(code_hs6, ""))
        else:
            desc = BASKET_DESC.get(code, BASKET_DESC.get(code_hs6, ""))

        value_usd, value_local, currency_local = "", "", ""
        if c_usd is not None:
            v = pd.to_numeric(rec.get(c_usd), errors="coerce")
            value_usd = round(float(v) * 1000) if pd.notna(v) else ""  # thousands -> dollars
        elif c_ntd is not None:
            v = pd.to_numeric(rec.get(c_ntd), errors="coerce")
            if pd.notna(v):
                value_local, currency_local = round(float(v) * 1000), "TWD"

        weight_kg = ""
        if c_kgm is not None:
            w = pd.to_numeric(rec.get(c_kgm), errors="coerce")
            weight_kg = float(w) if pd.notna(w) else ""
        elif c_tne is not None:
            w = pd.to_numeric(rec.get(c_tne), errors="coerce")
            weight_kg = float(w) * 1000 if pd.notna(w) else ""

        qty, qty_unit = "", ""
        if c_qty is not None:
            q = pd.to_numeric(rec.get(c_qty), errors="coerce")  # "-" -> NaN -> blank
            qty = float(q) if pd.notna(q) else ""
        if c_unit is not None:
            u = str(rec.get(c_unit, "")).strip()
            qty_unit = "" if u in ("", "-", "nan") else u

        partner = str(rec.get(c_ctry, "")).strip()
        partner = PARTNER_MAP.get(partner, partner)

        # A row with no value, weight or quantity carries no information.
        if (value_usd in ("", 0) and value_local in ("", 0) and weight_kg in ("", 0.0)
                and qty in ("", 0.0)):
            continue

        rows.append({
            "reporter": REPORTER,
            "flow": flow,
            "partner": partner,
            "code_native": code,
            "code_hs6": code_hs6,
            "code_desc": desc,
            "period": period,
            "value_usd": value_usd,
            "value_local": value_local,
            "currency_local": currency_local,
            "qty": qty,
            "qty_unit": qty_unit,
            "weight_kg": weight_kg,
            "source": SOURCE,
            "pulled_date": pulled,
        })
    return rows, prelim_periods


def merge_pulls(rows: list[dict]) -> pd.DataFrame:
    """Merge rows from overlapping files into one row per flow x code x partner x month.

    ``rows`` come file by file in file-name order. The newest retrieval wins; on a same-day
    tie the weight-bearing row wins (a re-export of the same query with the weight fields
    added), otherwise the file later in name order.
    """
    df = pd.DataFrame(rows, columns=SCHEMA_COLS)
    df["_w"] = (df["weight_kg"] != "").astype(int)
    return finalize(keep_newest(df, _KEY, tiebreak=["_w"]))


def _flow_token(df: pd.DataFrame) -> str:
    """The flows present, concatenated (``MX``), for file names."""
    return "".join(sorted(df["flow"].unique()))


def _stamp(df: pd.DataFrame) -> str:
    """The newest retrieval stamp among a book's rows, for file names."""
    return df["pulled_date"].max().replace("-", "")


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """Build the Taiwan books from every portal export in ``raw_dir`` (all retrievals merged).

    Returns the monthly book (2022 onward) and, when earlier months are present, the
    reference-year history book. With ``vintage``, files retrieved after that date are
    ignored. ``fx_rates`` is unused: values are published in USD.
    """
    files = sorted(raw_files(raw_dir, RAW_PATTERN, vintage))   # file-name order
    rows, prelim = [], set()
    for p in files:
        # Exports of other portal tables may sit alongside the basket files; they are
        # reported and left out rather than aborting the build.
        try:
            file_rows, file_prelim = read_raw_file(p)
        except Exception as e:
            print(f"  skip {p.name}: {type(e).__name__}: {e}")
            continue
        rows.extend(file_rows)
        prelim |= file_prelim
    if not rows:
        raise ValueError(f"no usable rows in the raw files under {raw_dir}")

    df = merge_pulls(rows)
    books = []
    main = df[df["period"] >= HISTORY_BEFORE].reset_index(drop=True)
    if not main.empty:
        years = main["period"].str[:4]
        name = (f"taiwan_T1-T2_{_flow_token(main)}_{years.min()}-{years.max()}"
                f"_pulled{_stamp(main)}.csv")
        books.append(CleanBook(main, name))
    history = df[df["period"] < HISTORY_BEFORE].reset_index(drop=True)
    if not history.empty:
        first, last = history["period"].str[:4].agg(["min", "max"])
        span = first if first == last else f"{first}-{last}"
        name = f"taiwan_history_T1_{_flow_token(history)}_{span}_pulled{_stamp(history)}.csv"
        books.append(CleanBook(history, name))

    if prelim:
        print(f"  taiwan: preliminary months (revised in later releases): {sorted(prelim)}")
    return books
