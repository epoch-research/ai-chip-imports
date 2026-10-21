"""United Kingdom: HMRC Overseas Trade in Goods Statistics (OTS) (book ``uk``).

Publication
    HM Revenue & Customs' monthly OTS fact table, published on uktradeinfo, at its native grain
    of month x flow type x CN8 commodity x partner country x port. The raw records are already
    summed over ports, so there is exactly one record per (month, flow type, CN8, partner).

Codes
    Native CN8 (8-digit Combined Nomenclature) only: OTS has no HS-6 rows, so the sum of the
    CN8 rows under a subheading is its HS-6 total by construction. ``code_desc`` is HMRC's CN8
    long description. The commodity code list spans nomenclature vintages back to 2006, but
    only one vintage carries trade per subheading from 2022 on, so retired lines simply have
    no rows.

Flows
    OTS has four flow types, an EU / non-EU split of two flows: 1 EU imports, 2 EU exports,
    3 non-EU imports, 4 non-EU exports. They map to M = {1, 3} and X = {2, 4}; the split is
    kept in the extra column ``uk_eu_flag`` (``EU`` / ``NON_EU``) so it never reads as a
    separate flow. OTS has no re-export concept, so RX and DX never occur.

Partners
    HMRC country names, with a few normalized to the spellings the other books use. Variants
    that normalize to the same name are summed. Residual buckets ("Other Asia and Oceania",
    "Confidential Country", "Stores & Provis.") are genuine partners in OTS and are kept;
    there are no world or EU aggregate partners to drop.

Values
    ``Value`` is in pounds sterling (not thousands) -> ``value_local`` (GBP). ``value_usd``
    divides it by the monthly period-average GBP-per-USD rate from the exchange-rate table;
    months without a rate leave it blank. Imports are valued CIF, exports FOB.

Quantity and weight
    ``NetMass`` (kg) -> ``weight_kg``; ``SuppUnit`` (supplementary units, number of items) ->
    ``qty`` with ``qty_unit`` ``NO`` where non-zero. Both are written as decimals and are 0.0
    where HMRC reports nothing.

Raw files
    ``uk_ots_T1-T2_MX_cn8_<YYYY>-<YYYY>_pulled<YYYYMMDD>.json``: a JSON envelope holding the
    OTS records exactly as HMRC returned them, with the metadata needed to read them:

        pulled_date   retrieval date
        frontier      {"year", "month"}: latest month HMRC had published; later months dropped
        years         calendar years covered
        codelist      HMRC Commodity records for the basket (Cn8Code, Hs6Code,
                      Cn8LongDescription, ...)
        rows          OTS records summed over ports: MonthId (YYYYMM), FlowTypeId,
                      Commodity.Cn8Code, Country.CountryName, V (value), NM (net mass),
                      SU (supplementary units)

Vintages
    A refresh may re-retrieve only the trailing years, so ``build`` merges every retrieval in
    the directory. Files are ranked newest first (by ``pulled<YYYYMMDD>`` stamp; same-day files
    by name, descending, so a same-day trailing refresh outranks the full retrieval) and each
    record is taken from the highest-ranked file carrying it; the merge key is (month, flow
    type, CN8, partner). The publication frontier and code list come from the top-ranked
    file. ``vintage="YYYYMMDD"`` leaves out the retrievals made after that date.

Output
    ``uk_T1-T2_MX_<first year>-<last year>_pulled<YYYYMMDD>.csv``, spanning every year any
    retrieval read covers, with the extra column ``uk_eu_flag``. ``pulled_date`` on every row
    and the output stamp are the top-ranked retrieval's date.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ._common import SCHEMA_COLS as _BASE_COLS
from ._common import CleanBook, finalize, iso_date, pulled_stamp, raw_files, to_usd

BOOK = "uk"
RAW_PATTERN = "uk_ots_*_pulled*.json"

REPORTER = "uk"
SOURCE = "hmrc_ots_odata"
CURRENCY = "GBP"

# Standard clean schema plus the EU / non-EU flag that OTS's flow types carry.
SCHEMA_COLS = _BASE_COLS + ["uk_eu_flag"]

# OTS FlowTypeId -> (schema flow, EU / non-EU flag).
FLOW_MAP = {1: ("M", "EU"), 2: ("X", "EU"), 3: ("M", "NON_EU"), 4: ("X", "NON_EU")}

# HMRC names -> the English names the other books use (HMRC names are otherwise kept as is).
PARTNER_MAP = {
    "USA": "United States",
    "United States of America": "United States",
    "China": "China",
    "Hong Kong": "Hong Kong",
    "Taiwan": "Taiwan",
    "Korea, South": "South Korea",
    "South Korea": "South Korea",
    "Korea (South)": "South Korea",
    "Korea, North": "North Korea",
    "Vietnam": "Vietnam",
    "Viet Nam": "Vietnam",
    "Russia": "Russia",
    "Russian Federation": "Russia",
    "Czechia": "Czech Republic",
    "Czech Republic": "Czech Republic",
    "Slovakia": "Slovakia",
    "Slovak Republic": "Slovakia",
    "Netherlands": "Netherlands",
    "United Arab Emirates": "United Arab Emirates",
    "Irish Republic": "Ireland",
    "Ireland": "Ireland",
}

_GROUP_KEYS = ["flow", "uk_eu_flag", "partner", "code_native", "code_hs6", "period"]


def normalize_partner(name) -> str:
    """HMRC country name -> the cross-book English name."""
    name = str(name or "").strip()
    return PARTNER_MAP.get(name, name)


def standardize(records: list[dict], codelist: list[dict], frontier: tuple[int, int],
                pulled_date: str, fx_rates: dict[tuple[str, str], float]) -> pd.DataFrame:
    """OTS records -> the clean frame.

    Records past ``frontier`` (year, month) are dropped. A (month, flow type, CN8, partner)
    record that appears more than once is taken from its first occurrence only, so overlapping
    files never double-count. Records with no value, mass or units are skipped.
    """
    fy, fm = frontier
    desc_by_cn8 = {c["Cn8Code"]: (c.get("Cn8LongDescription") or "").strip() for c in codelist}
    hs6_by_cn8 = {c["Cn8Code"]: c.get("Hs6Code") for c in codelist}

    rows, seen = [], set()
    for r in records:
        try:
            month_id = int(r["MonthId"])
            flow, eu_flag = FLOW_MAP[int(r["FlowTypeId"])]
        except (KeyError, TypeError, ValueError):
            continue
        cn8 = str((r.get("Commodity") or {}).get("Cn8Code") or "").strip()
        partner_raw = (r.get("Country") or {}).get("CountryName")
        if not cn8 or not partner_raw:
            continue
        y, m = month_id // 100, month_id % 100
        if not (1 <= m <= 12) or (y, m) > (fy, fm):
            continue
        ident = (month_id, int(r["FlowTypeId"]), cn8, str(partner_raw))
        if ident in seen:
            continue
        seen.add(ident)
        value = float(r.get("V") or 0)
        mass = r.get("NM")
        units = r.get("SU")
        if value == 0 and not mass and not units:
            continue
        rows.append({
            "flow": flow, "uk_eu_flag": eu_flag,
            "partner": normalize_partner(partner_raw),
            "code_native": cn8,
            "code_hs6": hs6_by_cn8.get(cn8) or cn8[:6],
            "code_desc": desc_by_cn8.get(cn8, ""),
            "period": f"{y:04d}-{m:02d}",
            "value_local": value,
            "weight_kg": float(mass) if mass not in (None, "") else None,
            "qty": float(units) if units not in (None, "") else None,
        })

    if not rows:
        return pd.DataFrame(columns=SCHEMA_COLS)

    # Summing over the group keys merges partner-name variants that normalize together.
    df = (pd.DataFrame(rows)
            .groupby(_GROUP_KEYS, as_index=False)
            .agg(value_local=("value_local", "sum"), weight_kg=("weight_kg", "sum"),
                 qty=("qty", "sum"), code_desc=("code_desc", "first")))
    df["value_local"] = df["value_local"].round().astype("int64")

    df["reporter"] = REPORTER
    df["source"] = SOURCE
    df["currency_local"] = CURRENCY
    df["qty_unit"] = df["qty"].apply(lambda v: "NO" if pd.notna(v) and v else "")

    df["value_usd"] = to_usd(df["value_local"], df["period"], CURRENCY, fx_rates)
    df["pulled_date"] = pulled_date
    return finalize(df, SCHEMA_COLS)


def build(raw_dir: Path, fx_rates: dict[tuple[str, str], float],
          vintage: str | None = None) -> list[CleanBook]:
    """Clean UK book from every raw retrieval in ``raw_dir``, newest record winning.

    Records are concatenated in rank order (newest first), and ``standardize`` keeps the
    first occurrence of each (month, flow type, CN8, partner), so the newest retrieval of every
    record wins. With ``vintage``, retrievals made after that date are ignored.
    """
    files = raw_files(raw_dir, RAW_PATTERN, vintage)
    stamp = pulled_stamp(files[0])
    records, years, frontier, codelist = [], set(), None, None
    for path in files:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if frontier is None:
            frontier = (raw["frontier"]["year"], raw["frontier"]["month"])
            codelist = raw["codelist"]
        records.extend(raw["rows"])
        years.update(int(y) for y in raw["years"])
    df = standardize(records, codelist, frontier, iso_date(stamp), fx_rates)
    blank = int((df["value_usd"] == "").sum())
    if blank:
        print(f"  [uk] value_usd blank on {blank:,} rows (no exchange rate for the month)")
    return [CleanBook(df, f"uk_T1-T2_MX_{min(years)}-{max(years)}_pulled{stamp}.csv")]
