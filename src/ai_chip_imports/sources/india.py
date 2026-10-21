"""India: Department of Commerce TRADESTAT, commodity x partner x month (book ``india``).

Publication
    Department of Commerce / Directorate General of Commercial Intelligence and Statistics
    (DGCI&S), TRADESTAT "Export Import Data Bank", monthly databank (MEIDB): the
    commodity-wise, all-countries import and export tables. Each table gives one code, one
    month and one flow for every partner country.

Codes
    India's schedule is ITC-HS 8-digit. Over this basket it is nearly flat (seven of the nine
    HS-6 parents have a single 8-digit line), so the raw tables are at the HS-6 parent and
    ``code_native`` is the 6-digit code (an 8-digit table keeps its 8-digit code).
    ``code_desc`` is the official ITC-HS description from the publisher's code list, else the
    HS-6 basket label.

Flows
    Imports -> M, exports -> X. TRADESTAT has no re-export split, so DX and RX are never
    emitted.

Partners
    TRADESTAT's upper-case partner names, normalized to the spellings the other books use
    (unlisted names are title-cased). Two labels that name one partner keep one cell.

Values
    Published in millions of US dollars to two decimals, i.e. at a resolution of $10,000:
    ``value_usd`` = value x 1e6, with no exchange-rate step (``fx_rates`` is unused), and
    ``value_local``/``currency_local`` stay blank. Partners whose value rounds to 0.00 are
    absent from the value table. Imports are valued CIF, exports FOB.

Quantity and weight
    Quantity is a separate table in the publisher's statistical unit, which ITC-HS defines on
    the 8-digit line (``NOS`` pieces, ``KGS`` kilograms): an HS-6 row carries its 8-digit
    lines' unit when they all share one and a blank unit when they mix (847330 has both NOS
    and KGS lines). Value and quantity are merged per (flow, code, month, partner), so a
    partner below the value resolution appears as a quantity-only row with a blank
    ``value_usd``. TRADESTAT publishes no weight.

Raw files
    ``india_tradestat_T1-T2_MX_<YYYY>-<YYYY>_pulled<YYYYMMDD>.json``, one per retrieval: a JSON
    envelope with

        codelist   the publisher's ITC-HS code list for the basket:
                   ``{code: {"unit": ..., "desc": ...}}`` at 6 and 8 digits
        records    one entry per table cell: ``{flow, code, period (YYYY-MM), measure ("value"
                   | "qty"), partner (the publisher's upper-case name), val, rev (revision
                   mark)}``, read from the requested-month column of each table (never the
                   prior-year or fiscal-year-to-date columns, which re-cut the same data)

Vintages
    A retrieval may cover only trailing months, so ``build`` merges every retrieval in the
    directory: files are ranked newest first (by ``pulled<YYYYMMDD>`` stamp; same-day files by
    name, descending, which puts a trailing refresh (later first year) ahead of the full
    retrieval) and each table cell is taken from the highest-ranked file carrying it; the
    merge key is (flow, code, period, measure, partner). The code list comes from the
    top-ranked file. ``vintage="YYYYMMDD"`` leaves out the retrievals made after that date.

Output
    ``india_T1-T2_MX_<first YYYY-MM>_<last YYYY-MM>_pulled<YYYYMMDD>.csv``. ``pulled_date`` on
    every row and the output stamp are the top-ranked retrieval's date. Two extra columns
    follow the shared schema:

    in_revision      the revision mark on the month's column: ``F`` final, ``R`` revised
                     final. Months still marked ``A`` (awaited) are zero-filled placeholders
                     and are not in the raw files.
    partner_suspect  True on every row of a (flow, code, month) cell where one partner holds
                     at least 97% of the value and at most two partners report any.
                     TRADESTAT has months, across India's fiscal year 2023-24, in which a
                     cell's whole total is booked to the alphabetically first partner and
                     every other partner shows 0.00. The cell total is right; its partner
                     split is not. Rows are flagged, not changed; exclude them from any
                     partner-level use.
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
    raw_files,
)

BOOK = "india"
REPORTER = "india"
SOURCE = "tradestat_meidb"
RAW_PATTERN = "india_tradestat_*_pulled*.json"

COLUMNS = SCHEMA_COLS + ["in_revision", "partner_suspect"]

# TRADESTAT's upper-case partner names -> the spellings the other books use. Unlisted names are
# title-cased ("BELGIUM" -> "Belgium").
PARTNER_MAP = {
    "CHINA P RP": "China",
    "CHINA": "China",
    "U S A": "United States",
    "USA": "United States",
    "UNITED STATES": "United States",
    "TAIWAN": "Taiwan",
    "CHINESE TAIPEI": "Taiwan",
    "KOREA RP": "South Korea",
    "KOREA DP RP": "North Korea",
    "HONG KONG": "Hong Kong",
    "VIETNAM SOC REP": "Vietnam",
    "SINGAPORE": "Singapore",
    "MALAYSIA": "Malaysia",
    "THAILAND": "Thailand",
    "INDONESIA": "Indonesia",
    "JAPAN": "Japan",
    "GERMANY": "Germany",
    "NETHERLAND": "Netherlands",
    "NETHERLANDS": "Netherlands",
    "U K": "United Kingdom",
    "UNITED KINGDOM": "United Kingdom",
    "IRELAND": "Ireland",
    "CZECH REPUBLIC": "Czech Republic",
    "MEXICO": "Mexico",
    "U ARAB EMTS": "United Arab Emirates",
    "UNITED ARAB EMIRATES": "United Arab Emirates",
    "RUSSIA": "Russia",
    "PHILIPPINES": "Philippines",
    "AUSTRALIA": "Australia",
    "ISRAEL": "Israel",
    "SWITZERLAND": "Switzerland",
    "FRANCE": "France",
    "ITALY": "Italy",
    "POLAND": "Poland",
    "HUNGARY": "Hungary",
    "AUSTRIA": "Austria",
    "SWEDEN": "Sweden",
    "CANADA": "Canada",
    "BRAZIL": "Brazil",
    "TURKEY": "Turkey",
    "SAUDI ARAB": "Saudi Arabia",
}

# The collapsed-partner signature behind ``partner_suspect``: one partner holding at least this
# share of a cell's value while at most this many partners report anything. Structural rather
# than a fixed date window, so a recurrence elsewhere is caught too.
SUSPECT_TOP_SHARE = 0.97
SUSPECT_MAX_PARTNERS = 2


def normalize_partner(raw) -> str:
    """Canonical English name for a TRADESTAT partner label."""
    raw = re.sub(r"\s+", " ", str(raw)).strip()
    return PARTNER_MAP.get(raw.upper(), raw.title())


def _qty_unit(code: str, codelist: dict) -> str:
    """Statistical unit of ``code``; for an HS-6 code, its 8-digit lines' unit if they agree."""
    unit = (codelist.get(code, {}).get("unit") or "").strip()
    if unit or len(code) >= 8:
        return unit
    kids = {(codelist[k].get("unit") or "").strip() for k in codelist
            if len(k) == 8 and k.startswith(code)}
    kids.discard("")
    return kids.pop() if len(kids) == 1 else ""


def process(records: list[dict], codelist: dict, pulled_date: str) -> pd.DataFrame:
    """Table-cell records -> the clean frame, merging the value and quantity tables."""
    if not records:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(records)
    df["partner"] = df["partner"].map(normalize_partner)
    # Two publisher labels can name one partner; keep one cell per normalized name.
    df = df.drop_duplicates(subset=["flow", "code", "period", "measure", "partner"], keep="last")

    key = ["flow", "code", "period", "partner"]
    val = (df[df["measure"] == "value"]
           .groupby(key, as_index=False)
           .agg(value_musd=("val", "sum"), in_revision=("rev", "first")))
    qty = (df[df["measure"] == "qty"]
           .groupby(key, as_index=False).agg(qty=("val", "sum")))
    out = val.merge(qty, on=key, how="outer")

    out["reporter"] = REPORTER
    out["source"] = SOURCE
    out["code_native"] = out["code"].astype(str)
    out["code_hs6"] = out["code_native"].str[:6]
    out["code_desc"] = [
        (codelist.get(c, {}).get("desc") or BASKET_DESC.get(c[:6], ""))
        for c in out["code_native"]
    ]
    out["value_usd"] = (pd.to_numeric(out["value_musd"], errors="coerce") * 1e6).round(0)
    out["value_local"] = ""
    out["currency_local"] = ""
    out["qty_unit"] = [
        _qty_unit(c, codelist) if pd.notna(q) and q else ""
        for c, q in zip(out["code_native"], out["qty"])
    ]
    out["weight_kg"] = ""
    out["pulled_date"] = pulled_date
    out["in_revision"] = out["in_revision"].fillna("")

    cell = out.groupby(["flow", "code_native", "period"])["value_usd"]
    cell_sum = cell.transform("sum")
    cell_max = cell.transform("max")
    cell_n = cell.transform(lambda s: (s.fillna(0) > 0).sum())
    out["partner_suspect"] = (
        (cell_sum > 0)
        & (cell_max / cell_sum >= SUSPECT_TOP_SHARE)
        & (cell_n <= SUSPECT_MAX_PARTNERS)
    )

    out = out[out["value_usd"].notna() | out["qty"].notna()]
    return finalize(out, COLUMNS)


def load_records(files: list[Path]) -> tuple[list[dict], dict]:
    """Merged table-cell records and the code list from files ordered newest first.

    The first file to carry a (flow, code, period, measure, partner) cell supplies it.
    """
    codelist, seen, records = None, set(), []
    for path in files:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if codelist is None:
            codelist = raw.get("codelist", {})
        for r in raw["records"]:
            k = (r["flow"], r["code"], r["period"], r.get("measure"), r["partner"])
            if k in seen:
                continue
            seen.add(k)
            records.append(r)
    return records, codelist or {}


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """Build the India clean book from every raw file in ``raw_dir``, newest retrieval winning.

    With ``vintage``, retrievals made after that date are ignored. ``fx_rates`` is unused:
    the publication reports US dollars.
    """
    files = raw_files(raw_dir, RAW_PATTERN, vintage)
    stamp = pulled_stamp(files[0])
    records, codelist = load_records(files)
    df = process(records, codelist, iso_date(stamp))
    coverage = f"{df['period'].min()}_{df['period'].max()}" if not df.empty else "empty"
    return [CleanBook(df, f"india_T1-T2_MX_{coverage}_pulled{stamp}.csv")]
