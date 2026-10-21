"""Malaysia: Department of Statistics Malaysia (DOSM), external trade (book ``malaysia``).

Publication
    Malaysia External Trade Statistics (METS), compiled by DOSM from Royal Malaysian Customs
    declarations and disseminated through METS Online: the "Classification" tables, monthly by
    commodity and partner country. Monthly from January 2022.

Codes
    Malaysia's 10-digit national tariff lines (AHTN-based; one nomenclature for all flows).
    Three levels are kept as separate rows: the 10-digit lines under the basket's nine HS-6
    codes, the nine HS-6 subheadings and the HS-4 headings 8471/8473/8542 (``code_hs6``
    repeats the 4-digit code on those). 10-digit rows carry the line's official description
    from the publisher's product list (the data tables repeat only the HS-6 heading text at
    that level; where the list gives several variants of a code's description, the longest,
    most complete one is used); HS-6 and HS-4 rows the short basket label.

Flows
    Imports -> M, Exports -> X (total), Domestic Exports -> DX, Re-Exports -> RX, so
    X == DX + RX.

Partners
    The publisher's partner list keeps historical name variants of the same country (e.g.
    "UNITED STATES" and "UNITED STATES OF AMERICA"); variants mapping to one canonical name
    are summed. Unlisted names are title-cased.

Values
    Plain Malaysian ringgit (not thousands): ``value_local`` in MYR. ``value_usd`` divides by
    the IMF monthly period-average MYR per USD and is blank for months without a rate.
    Imports are valued c.i.f., exports f.o.b. The tables report 0, not blank, for months not
    yet published, so months after the publication frontier recorded in the raw file are
    dropped, as are zero cells.

Quantity and weight
    METS publishes values only: ``qty``, ``qty_unit`` and ``weight_kg`` are blank.

Raw files
    ``malaysia_mets_T1-T2_MXRXDX_hs4-6-10_<YYYY>-<YYYY>_pulled<YYYYMMDD>.json``, one per
    retrieval: a JSON envelope holding ``frontier`` (``{"year", "month"}``, the latest
    published month), ``code_entries_hs10`` (the publisher's product list verbatim:
    ``{"codeValue", "codeName"}`` records, all nomenclature vintages) and ``passes``, one per
    code level and year: ``{"hsType", "year", "columns", "items"}``, the publisher's wide
    table verbatim. Each item is one code x partner row (code, description, partner country
    name, then one column per month x flow headed like ``"Jan_2024_RE-EXPORTS Value (MYR)"``).

Vintages
    Every retrieval re-reads the whole series, so ``build`` reads the newest file alone;
    ``vintage="YYYYMMDD"`` pins the file stamped that day.

Output
    ``malaysia_T1-T2_MXRXDX_<first year>-<last year>_pulled<YYYYMMDD>.csv``. ``pulled_date`` is
    the raw file's ``pulled<YYYYMMDD>`` stamp.
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

BOOK = "malaysia"
REPORTER = "malaysia"
SOURCE = "mets_api"
CURRENCY = "MYR"
RAW_PATTERN = "malaysia_mets_T1-T2_MXRXDX_hs4-6-10_*_pulled*.json"

FLOW_MAP = {"IMPORTS": "M", "EXPORTS": "X", "DOMESTIC EXPORTS": "DX", "RE-EXPORTS": "RX"}

# Partner names (upper case, as published) that differ from the canonical English names used
# across books, including the historical variants of one country. Any other name is
# title-cased.
PARTNER_MAP = {
    "UNITED STATES": "United States",
    "UNITED STATES OF AMERICA": "United States",
    "TAIWAN": "Taiwan",
    "TAIWAN, PROVINCE OF CHINA": "Taiwan",
    "CHINA": "China",
    "CHINA, PEOPLE'S REPUBLIC OF": "China",
    "HONG KONG": "Hong Kong",
    "HONG KONG, SAR": "Hong Kong",
    "MACAO": "Macau",
    "MACAU": "Macau",
    "KOREA, REPUBLIC OF": "South Korea",
    "KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF": "North Korea",
    "VIET NAM": "Vietnam",
    "VIETNAM, SOCIALIST REP. OF": "Vietnam",
    "RUSSIAN FEDERATION": "Russia",
    "LAO, PEOPLE'S DEMOCRATIC REPUBLIC": "Laos",
    "MOLDOVA, REPUBLIC OF": "Moldova",
    "CONGO, THE DEMOCRATIC REPUBLIC OF THE": "Democratic Republic of the Congo",
    "TANZANIA, UNITED REPUBLIC OF": "Tanzania",
    "BRUNEI DARUSSALAM": "Brunei",
    "UNITED ARAB EMIRATES": "United Arab Emirates",
    "SLOVAK REPUBLIC": "Slovakia",
    "SLOVAKIA": "Slovakia",
    "CZECH REPUBLIC": "Czech Republic",
    "UNITED KINGDOM": "United Kingdom",
}

MONTH_NUM = {abbr: i for i, abbr in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}

_VALUE_SUFFIX = " Value (MYR)"


def normalize_partner(name) -> str:
    """Published partner name -> canonical English name."""
    name = str(name).strip()
    return PARTNER_MAP.get(name.upper(), name.title())


def value_columns(columns) -> dict[int, tuple[str, str]]:
    """Column index -> (``YYYY-MM``, flow) for the value columns of a wide table, whose
    headers read like ``"Jan_2024_DOMESTIC EXPORTS Value (MYR)"``."""
    out = {}
    for i, col in enumerate(columns):
        c = str(col)
        if _VALUE_SUFFIX not in c:
            continue
        try:
            mon, year, flow_tok = c.split(_VALUE_SUFFIX)[0].split("_", 2)
        except ValueError:
            continue
        flow = FLOW_MAP.get(flow_tok.strip().upper())
        m = MONTH_NUM.get(mon.strip().title())
        if flow and m:
            out[i] = (f"{int(year):04d}-{m:02d}", flow)
    return out


def hs10_descriptions(entries) -> dict[str, str]:
    """``{10-digit code: description}`` from the product list, keeping the longest of a code's
    variants (the list mixes nomenclature vintages, the longest text is the most complete)."""
    best = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        cv, nm = e.get("codeValue"), (e.get("codeName") or "").strip()
        if isinstance(cv, str) and len(cv) == 10 and nm and len(nm) > len(best.get(cv, "")):
            best[cv] = nm
    return best


def standardize(passes, frontier: tuple[int, int], hs10_desc: dict[str, str],
                pulled_date: str, fx_rates: dict) -> pd.DataFrame:
    """Wide METS tables -> the clean frame: melted, clipped at the publication frontier and
    with partner-name variants summed."""
    rows = []
    for p in passes:
        columns, items = p["columns"], p["items"]
        if not columns or not items:
            continue
        try:
            partner_idx = columns.index("partner_country_name")
        except ValueError:
            continue   # a table without a partner breakdown has no place in the book
        valcols = value_columns(columns)
        for row in items:
            code = str(row[0]).strip()
            if not code or code.lower() == "nan":
                continue
            code_hs6 = code[:6]
            row_desc = str(row[1]).strip() if len(row) > 1 else ""
            native_desc = hs10_desc.get(code) if len(code) == 10 else BASKET_DESC.get(code)
            desc = native_desc or row_desc or BASKET_DESC.get(code_hs6, "")
            partner = normalize_partner(row[partner_idx])
            for i, (period, flow) in valcols.items():
                if i >= len(row):
                    continue
                v = row[i]
                if not isinstance(v, (int, float)) or v == 0:
                    continue
                if (int(period[:4]), int(period[5:7])) > frontier:
                    continue   # not yet published: the cell cannot be trusted
                rows.append({
                    "flow": flow, "partner": partner, "code_native": code,
                    "code_hs6": code_hs6, "code_desc": desc, "period": period,
                    "value_local": float(v),
                })

    if not rows:
        return pd.DataFrame(columns=SCHEMA_COLS)

    df = pd.DataFrame(rows)
    df = (df.groupby(["flow", "partner", "code_native", "code_hs6", "period"], as_index=False)
            .agg(value_local=("value_local", "sum"), code_desc=("code_desc", "first")))
    df["value_local"] = df["value_local"].round().astype("int64")

    df["reporter"] = REPORTER
    df["source"] = SOURCE
    df["currency_local"] = CURRENCY
    df["value_usd"] = to_usd(df["value_local"], df["period"], CURRENCY, fx_rates)
    df["qty"] = ""
    df["qty_unit"] = ""
    df["weight_kg"] = ""
    df["pulled_date"] = pulled_date
    return finalize(df)


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """Build the Malaysia clean book from the newest raw file, or the one stamped
    ``vintage``."""
    path = select_vintage(raw_dir, RAW_PATTERN, vintage)[-1]
    stamp = pulled_stamp(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    frontier = (raw["frontier"]["year"], raw["frontier"]["month"])
    hs10_desc = hs10_descriptions(raw.get("code_entries_hs10", []))
    df = standardize(raw["passes"], frontier, hs10_desc, iso_date(stamp), fx_rates)
    years = [p["year"] for p in raw["passes"]]
    name = f"malaysia_T1-T2_MXRXDX_{min(years)}-{max(years)}_pulled{stamp}.csv"
    return [CleanBook(df, name)]
