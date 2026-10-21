"""Singapore: Enterprise Singapore merchandise trade, via SingStat (book ``singapore``).

Publication
    The SingStat Table Builder "Detailed Trade Dataset": table T010001 (trade values) and its
    twin T010002 (trade volumes), monthly by commodity and market, compiled by Enterprise
    Singapore. Monthly from January 2022.

Codes
    HS 8-digit (AHTN 2022). Every published record carries its full hierarchy
    (``hs2Products`` .. ``hs8Products`` as ``"<code> - <label>"``), with an aggregate record at
    each level beside the 8-digit detail. Two levels are kept as separate rows: the 8-digit
    lines under the basket's nine HS-6 codes (the set listed in the raw file, so nomenclature
    changes carry through) and the nine HS-6 aggregates, so the two levels can be reconciled.
    There are no 4-digit rows. 8-digit rows carry the publisher's label; HS-6 rows the short
    basket label.

Flows
    IMPORTS -> M, EXPORTS -> X (total), DOMESTIC EXPORTS -> DX, RE-EXPORTS -> RX, so
    X == DX + RX up to the publisher's independent rounding of each flow to S$ thousands.

Partners
    "Markets". Regional groupings (WORLD, ASEAN, APEC, EUROPEAN UNION, ...) are dropped so the
    book is country-level only and nothing is double counted. Names are normalized to the
    spellings the other books use (unlisted names are title-cased).

Values
    Published in S$ thousands; ``value_local`` is SGD (x 1,000). ``value_usd`` divides by the
    IMF monthly period-average SGD per USD and is blank for months without a rate. Imports
    are valued c.i.f., exports f.o.b. Months after the table's last published period
    (``end_period``) are clipped, and empty or zero values are dropped.

Quantity and weight
    T010002 is published for 8-digit lines only; its quantities are joined onto the 8-digit
    value rows of the same code, market, flow and month. The unit is the suffix of the
    8-digit label: ``NMB`` (number of pieces), ``MIL`` (thousands of pieces) or ``KGM``
    (kilograms); ``qty`` is stored as published. HS-6 rows carry no quantity, since their
    lines mix units. The publication has no weight field, so ``weight_kg`` is blank.

Raw files
    ``singapore_stb_T1-T2_T010001_<YYYY>-<YYYY>_pulled<YYYYMMDD>.json`` (values) and
    ``singapore_stb_T1-T2_T010002_<YYYY>-<YYYY>_pulled<YYYYMMDD>.json`` (volumes) of the same
    retrieval. Each is a JSON envelope holding ``end_period`` (the table's last published
    month, e.g. ``"2026 Jul"``) and ``passes``, a list of ``{"search": <HS-6 code>, "rows":
    [...]}`` whose rows are the publisher's records verbatim (``market``, ``tradeType``,
    ``period`` as ``"2026 Jul"``, and ``value`` or ``quantity`` as strings); the values file
    also holds ``hs8_labels``, the publisher's 8-digit labels for the basket. The volumes file
    is optional: without it the book has no quantities.

Vintages
    Every retrieval re-reads the whole series, so ``build`` reads the value/volume pair of the
    newest retrieval only; ``vintage="YYYYMMDD"`` pins the retrieval stamped that day. Each
    code's records must cover every month from the retrieval's first month through
    ``end_period``: an incomplete retrieval (one that lost months for some codes) fails
    loudly instead of producing a book with silent holes, and an earlier, complete retrieval
    must be pinned instead.

Output
    ``singapore_T1-T2_MXRXDX_<first year>-<last year>_pulled<YYYYMMDD>.csv``. ``pulled_date``
    is the retrieval's ``pulled<YYYYMMDD>`` stamp.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ._common import (
    BASKET_DESC,
    SCHEMA_COLS,
    CleanBook,
    finalize,
    iso_date,
    pulled_stamp,
    select_vintage,
    to_usd,
)

BOOK = "singapore"
REPORTER = "singapore"
SOURCE = "singstat_tablebuilder"
CURRENCY = "SGD"
VALUE_PATTERN = "singapore_stb_T1-T2_T010001_*_pulled*.json"
VOLUME_PATTERN = "singapore_stb_T1-T2_T010002_*_pulled*.json"

FLOW_MAP = {"IMPORTS": "M", "EXPORTS": "X", "DOMESTIC EXPORTS": "DX", "RE-EXPORTS": "RX"}

# The basket's HS-6 codes, whose aggregate records are kept beside the 8-digit lines.
CODES_HS6 = {code for code in BASKET_DESC if len(code) == 6}

# Markets that are regional groupings rather than countries. They overlap each other and the
# countries they contain, so keeping any of them would double count. ("CARRIBBEAN" is the
# publisher's spelling.)
MARKET_GROUPINGS = {
    "WORLD", "APEC", "ASEAN", "CARRIBBEAN", "CENTRAL ASIA", "CLMV", "EAST ASIA + TAIWAN",
    "EUROPEAN UNION", "GCC", "LATIN AMERICA", "MIDDLE EAST", "NAFTA", "SOUTH ASIA",
    "OC AFRICA", "OC ASIA", "COUNTRIES NOT LISTED",
}

# Market names (upper case, as published) that differ from the canonical English names used
# across books. Any other name is title-cased. The Lao name uses a typographic apostrophe.
PARTNER_MAP = {
    "MAINLAND CHINA": "China",
    "TAIWAN": "Taiwan",
    "HONG KONG": "Hong Kong",
    "MACAO": "Macau",
    "REPUBLIC OF KOREA": "South Korea",
    "UNITED STATES OF AMERICA": "United States",
    "UNITED STATES VIRGIN ISLANDS": "US Virgin Islands",
    "UNITED KINGDOM": "United Kingdom",
    "VIET NAM": "Vietnam",
    "RUSSIAN FEDERATION": "Russia",
    "LAO PEOPLE’S DEMOCRATIC REPUBLIC": "Laos",
    "BRUNEI DARUSSALAM": "Brunei",
    "IRAN (ISLAMIC REPUBLIC OF)": "Iran",
    "DEMOCRATIC REPUBLIC OF THE CONGO": "Democratic Republic of the Congo",
    "COTE D'IVOIRE": "Cote d'Ivoire",
    "CZECHIA": "Czech Republic",
    "TÜRKIYE": "Turkey",
    "MACEDONIA": "North Macedonia",
    "CABO VERDE": "Cape Verde",
    "VATICAN (HOLY SEE)": "Vatican City",
    "TIMOR-LESTE": "Timor-Leste",
    "UNITED ARAB EMIRATES": "United Arab Emirates",
}

MONTH_NUM = {abbr: i for i, abbr in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}

_MISSING = ("-", "na", "NA", "nan")


def parse_period(raw) -> str:
    """``"2026 Apr"`` -> ``"2026-04"``; ``""`` if unparseable."""
    parts = str(raw).split()
    if len(parts) != 2 or parts[1].title() not in MONTH_NUM:
        return ""
    return f"{int(parts[0]):04d}-{MONTH_NUM[parts[1].title()]:02d}"


def parse_qty_unit(label) -> str:
    """Unit token from an 8-digit label suffix: ``"... MEMORIES (MIL)"`` -> ``"MIL"``."""
    m = re.search(r"\(([A-Z]{2,4})\)\s*$", str(label))
    return m.group(1) if m else ""


def normalize_partner(market) -> str:
    """Published market name -> canonical English name."""
    market = str(market).strip()
    return PARTNER_MAP.get(market.upper(), market.title())


def _code_of(field) -> str:
    """Leading code of a hierarchy field: ``"854232 - MEMORIES"`` -> ``"854232"``."""
    field = str(field or "").strip()
    return field.split(" - ")[0].strip() if field else ""


def _frontier(end_period: str) -> tuple[int, int]:
    """``"2026 Jul"`` -> ``(2026, 7)``."""
    year, mon = end_period.split()
    return int(year), MONTH_NUM[mon.title()]


def _value_sgd(v) -> int | None:
    """S$-thousands string -> SGD; None for empty, zero or non-numeric values."""
    s = str(v).strip()
    if not s or s in _MISSING:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return int(round(f * 1000)) if f != 0 else None


def _country_record(row, frontier) -> tuple[str, str, str] | None:
    """(flow, market, period) of a country-level record within the frontier, else None."""
    flow = FLOW_MAP.get(str(row.get("tradeType", "")).strip().upper())
    if not flow:
        return None
    market = str(row.get("market", "")).strip()
    if market.upper() in MARKET_GROUPINGS:
        return None
    period = parse_period(row.get("period"))
    if not period or (int(period[:4]), int(period[5:7])) > frontier:
        return None
    return flow, market, period


def value_rows(passes, hs8_labels: dict[str, str], end_period: str) -> list[dict]:
    """T010001 records -> one row per (flow, market, code, month), 8- and 6-digit levels.

    Each pass holds a 6-digit code's aggregate records beside its 8-digit detail (and, since
    the publisher matches labels as well as codes, occasionally records of unrelated codes),
    so every record is matched on its exact code at its own level; repeats collapse on the key.
    """
    frontier = _frontier(end_period)
    out = {}
    for p in passes:
        for row in p["rows"]:
            code8 = _code_of(row.get("hs8Products"))
            code6 = _code_of(row.get("hs6Products"))
            if code8:
                if code8 not in hs8_labels:
                    continue
                code_native, code_hs6 = code8, code8[:6]
                desc = hs8_labels[code8].split(" - ", 1)[-1].strip()
            elif code6 in CODES_HS6:
                code_native, code_hs6, desc = code6, code6, BASKET_DESC[code6]
            else:
                continue   # another HS-6 code, or an aggregate above HS-6
            rec = _country_record(row, frontier)
            if rec is None:
                continue
            flow, market, period = rec
            val = _value_sgd(row.get("value"))
            if val is None:
                continue
            out[(flow, market, code_native, period)] = {
                "flow": flow, "partner": normalize_partner(market), "market": market,
                "code_native": code_native, "code_hs6": code_hs6, "code_desc": desc,
                "period": period, "value_local": val,
            }
    return list(out.values())


def volume_index(passes, hs8_labels: dict[str, str], end_period: str) -> dict:
    """T010002 records -> ``{(code8, MARKET, flow, period): (quantity, unit)}``."""
    frontier = _frontier(end_period)
    out = {}
    for p in passes:
        for row in p["rows"]:
            code8 = _code_of(row.get("hs8Products"))
            if code8 not in hs8_labels:
                continue
            rec = _country_record(row, frontier)
            if rec is None:
                continue
            flow, market, period = rec
            q = str(row.get("quantity", "")).strip()
            if not q or q in _MISSING:
                continue
            try:
                qi = int(round(float(q)))
            except ValueError:
                continue
            if qi == 0:
                continue
            out[(code8, market.upper(), flow, period)] = (qi, parse_qty_unit(hs8_labels[code8]))
    return out


def standardize(value_passes, volume_passes, hs8_labels: dict[str, str], end_period: str,
                pulled_date: str, fx_rates: dict) -> pd.DataFrame:
    """Value and volume records -> the clean frame."""
    rows = value_rows(value_passes, hs8_labels, end_period)
    if not rows:
        return pd.DataFrame(columns=SCHEMA_COLS)
    vol = volume_index(volume_passes, hs8_labels, end_period)

    df = pd.DataFrame(rows)
    df["reporter"] = REPORTER
    df["source"] = SOURCE
    df["currency_local"] = CURRENCY
    df["pulled_date"] = pulled_date
    df["weight_kg"] = ""

    qty, unit = [], []
    for r in rows:
        q, u = (vol.get((r["code_native"], r["market"].upper(), r["flow"], r["period"]),
                        (None, None))
                if len(r["code_native"]) == 8 else (None, None))
        qty.append(q if q is not None else "")
        unit.append(u if u else "")
    df["qty"] = qty
    df["qty_unit"] = unit

    df["value_usd"] = to_usd(df["value_local"], df["period"], CURRENCY, fx_rates)
    return finalize(df)


def _month_range(first: str, last: str) -> list[str]:
    """Every ``YYYY-MM`` from ``first`` through ``last``."""
    lo = int(first[:4]) * 12 + int(first[5:7]) - 1
    hi = int(last[:4]) * 12 + int(last[5:7]) - 1
    return [f"{i // 12:04d}-{i % 12 + 1:02d}" for i in range(lo, hi + 1)]


def check_coverage(passes, end_period: str, table: str) -> None:
    """Raise if any code's records skip a month between the retrieval's first month and
    ``end_period``: the signature of a retrieval that silently lost months."""
    have = {p["search"]: {parse_period(r.get("period")) for r in p["rows"]} - {""}
            for p in passes}
    seen = set().union(*have.values())
    if not seen:
        raise ValueError(f"{table}: no records")
    expected = _month_range(min(seen), parse_period(end_period))
    holes = {code: [m for m in expected if m not in months] for code, months in have.items()}
    holes = {code: miss for code, miss in holes.items() if miss}
    if holes:
        detail = "; ".join(f"{code}: {len(miss)} month(s) {miss[0]}..{miss[-1]}"
                           for code, miss in holes.items())
        raise ValueError(f"{table} retrieval is incomplete through {end_period} ({detail}); "
                         f"build a complete earlier retrieval with vintage='YYYYMMDD' "
                         f"(--vintage on the command line)")


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """Build the Singapore clean book from the newest retrieval, or the one stamped
    ``vintage``.

    Raises ValueError if the retrieval is missing months (see ``check_coverage``).
    """
    value_path = select_vintage(raw_dir, VALUE_PATTERN, vintage)[-1]
    stamp = pulled_stamp(value_path)
    values = json.loads(value_path.read_text(encoding="utf-8"))
    try:
        volume_path = select_vintage(raw_dir, VOLUME_PATTERN, stamp)[-1]
    except FileNotFoundError:          # no volume table in this retrieval: no quantities
        volume_path = None
    volumes = (json.loads(volume_path.read_text(encoding="utf-8"))
               if volume_path else {"passes": []})

    end_period = values["end_period"]
    check_coverage(values["passes"], end_period, "T010001")
    if volume_path:
        check_coverage(volumes["passes"], end_period, "T010002")

    df = standardize(values["passes"], volumes["passes"], values["hs8_labels"], end_period,
                     iso_date(stamp), fx_rates)
    years = sorted({parse_period(r.get("period"))[:4] for p in values["passes"]
                    for r in p["rows"]} - {""})
    name = f"singapore_T1-T2_MXRXDX_{years[0]}-{years[-1]}_pulled{stamp}.csv"
    return [CleanBook(df, name)]
