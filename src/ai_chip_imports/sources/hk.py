"""Hong Kong: Census and Statistics Department (C&SD) merchandise trade (book ``hk``).

Publication
    Hong Kong's monthly external merchandise trade statistics, as disseminated by C&SD's
    Interactive Data Dissemination Service for Trade Statistics (Trade-IDDS). Monthly, by
    commodity and by partner country, from January 2022.

Codes
    Hong Kong Harmonized System (HKHS), a single nomenclature for all flows. Three levels are
    kept as separate rows so the 8-digit lines can be reconciled against their published
    6-digit totals: the HKHS-8 national lines under the basket's nine HS-6 codes, the nine
    HKHS-6 codes themselves, and the HKHS-4 parents 8471/8473/8542 (``code_hs6`` blank on
    those). HKHS-8 rows carry the publisher's own English label; 6- and 4-digit rows the
    short basket label.

Flows
    Trade type 1 imports -> M, 2 domestic exports -> DX, 3 re-exports -> RX, 4 total exports
    -> X, so X == DX + RX. Imports are by country of origin; the three export flows are by
    country of destination (C&SD crosses re-exports with commodity detail only by
    destination).

Partners
    C&SD partner names, normalized to the spellings the other books use (unlisted names are
    title-cased).

Values
    Published in HK$ thousands; ``value_local`` is HKD (x 1,000). ``value_usd`` divides by the
    IMF monthly period-average HKD per USD and is blank for months without a rate. Imports
    are valued c.i.f., exports f.o.b.

Quantity and weight
    The quantity series (``sv == "QCm"``) is joined onto the value rows (``sv == "VCm"``) of
    the same trade type, code, partner and month, with the publisher's unit label
    (``unitEN``, e.g. ``THOUSAND`` for integrated circuits). HKHS-4 rows are value-only.
    C&SD publishes no shipping weight, so ``weight_kg`` is always blank.

Raw files
    ``hk_tradeidds_T1-T2_ttype<k>_<YYYYMM>-<YYYYMM>_pulled<YYYYMMDD>.json``, one per trade
    type k = 1..4 and retrieval. Each is a JSON envelope with ``ttype``, ``period``
    (``[first, last]`` month as ``YYYYMM``) and ``dataSet``, the publisher's records verbatim:
    one record per month x partner x code x statistic, with fields ``period``, ``co``/``cc``
    (partner code) and ``coDescEN``/``ccDescEN`` (partner name), ``codeclass``, ``code``,
    ``codeDescEN``, ``sv``, ``unitEN`` and ``figure`` (a string; empty for nil records).

Vintages
    Every retrieval re-reads the whole series, so ``build`` reads the four files of the
    newest retrieval only; ``vintage="YYYYMMDD"`` pins the retrieval stamped that day.

Output
    ``hk_T1-T2_MXRXDX_<first year>-<last year>_pulled<YYYYMMDD>.csv``. ``pulled_date`` is the
    retrieval's ``pulled<YYYYMMDD>`` stamp.
"""
from __future__ import annotations

import json
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

BOOK = "hk"
REPORTER = "hk"
SOURCE = "tradeidds"
CURRENCY = "HKD"
RAW_PATTERN = "hk_tradeidds_T1-T2_ttype*_pulled*.json"

# Trade type -> schema flow, and the partner dimension each trade type is published by.
TTYPE_FLOW = {1: "M", 2: "DX", 3: "RX", 4: "X"}
TTYPE_DIM = {1: "co", 2: "cc", 3: "cc", 4: "cc"}   # imports by origin, exports by destination

# C&SD partner names (upper case) that differ from the canonical English names used across
# books. Any other name is title-cased.
PARTNER_MAP = {
    "CHINESE MAINLAND": "China",
    "KOREA": "South Korea",
    "KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF": "North Korea",
    "TAIWAN": "Taiwan",
    "USA": "United States",
    "VIET NAM": "Vietnam",
    "TURKIYE": "Turkey",
    "CZECHIA": "Czech Republic",
    "MACAO": "Macau",
    "RUSSIA": "Russia",
}


def normalize_partner(name) -> str:
    """C&SD partner name -> canonical English name."""
    name = str(name).strip()
    return PARTNER_MAP.get(name.upper(), name.title())


def _period(yyyymm) -> str:
    """``202605`` -> ``2026-05``."""
    s = str(yyyymm)
    return f"{s[:4]}-{s[4:6]}"


def _figure(figure) -> float | None:
    """A ``figure`` string as a number; None for the empty strings of nil records."""
    try:
        return float(str(figure).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def load_raw(files) -> tuple[dict[int, list[dict]], str, str]:
    """Read one retrieval's raw files -> ``({trade type: records}, first YYYYMM, last YYYYMM)``."""
    records, first, last = {}, None, None
    for path in sorted(files):
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        records[blob["ttype"]] = blob["dataSet"]
        first, last = blob["period"]
    return records, first, last


def standardize(records_by_ttype: dict[int, list[dict]], fx_rates: dict,
                pulled_date: str) -> pd.DataFrame:
    """C&SD records -> clean rows: one per flow x code x partner x month, quantity joined on."""
    values = {}   # (ttype, code, partner code, period) -> clean record
    qty = {}      # same key -> (quantity, unit)

    for ttype, rows in records_by_ttype.items():
        flow = TTYPE_FLOW[ttype]
        dim = TTYPE_DIM[ttype]
        for r in rows:
            code = str(r.get("code", ""))
            key = (ttype, code, r.get(dim), _period(r.get("period")))
            if r.get("sv") == "QCm":
                q = _figure(r.get("figure"))
                if q is not None and q != 0:
                    qty[key] = (int(round(q)), str(r.get("unitEN", "")).strip())
                continue
            v = _figure(r.get("figure"))
            if v is None:
                continue
            codeclass = r.get("codeclass", "")
            is_parent = codeclass == "HKHS4" or len(code) == 4
            if codeclass == "HKHS8":
                desc = str(r.get("codeDescEN", "")).strip()
            else:
                desc = BASKET_DESC.get(code if is_parent else code[:6], "")
            values[key] = {
                "reporter": REPORTER,
                "flow": flow,
                "partner": normalize_partner(r.get(dim + "DescEN", "")),
                "code_native": code,
                "code_hs6": "" if is_parent else code[:6],
                "code_desc": desc,
                "period": key[3],
                "value_local": int(round(v * 1000.0)),   # HK$ thousands -> HKD
                "currency_local": CURRENCY,
                "qty": "", "qty_unit": "", "weight_kg": "",
                "source": SOURCE,
                "pulled_date": pulled_date,
            }

    for key, rec in values.items():
        if key in qty:
            rec["qty"], rec["qty_unit"] = qty[key]

    df = pd.DataFrame(list(values.values()))
    if df.empty:
        return pd.DataFrame(columns=SCHEMA_COLS)
    df["value_usd"] = to_usd(df["value_local"], df["period"], CURRENCY, fx_rates)
    return finalize(df)


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """Build the Hong Kong clean book from the newest retrieval, or the one stamped
    ``vintage``."""
    files = select_vintage(raw_dir, RAW_PATTERN, vintage)
    stamp = pulled_stamp(files[0])
    records, first, last = load_raw(files)
    df = standardize(records, fx_rates, iso_date(stamp))
    name = f"hk_T1-T2_MXRXDX_{first[:4]}-{last[:4]}_pulled{stamp}.csv"
    return [CleanBook(df, name)]
