"""Indonesia: Statistics Indonesia (BPS) foreign-trade statistics, HS-8 (book ``indonesia``).

Publication
    Badan Pusat Statistik (BPS, Statistics Indonesia), foreign trade data (Ekspor-Impor),
    monthly, at the full national tariff line, by partner country and port.

Codes
    BTKI 8-digit lines (Indonesia's ASEAN-harmonised tariff schedule), labelled by HS vintage.
    The 8-digit lines under each HS-6 basket parent are all lines that carried trade.
    ``code_native`` is the 8-digit code, ``code_hs6`` its first six digits, and ``code_desc``
    the publisher's description from the row label (the HS-6 basket label if it has none).

Flows
    Exports -> X, imports -> M. BPS has no re-export split, so DX and RX are never emitted.

Partners
    BPS's upper-case partner labels, normalized to the spellings the other books use
    (unlisted labels are title-cased). Labels that name the same country under different
    conventions ("KOREA, REPUBLIC OF", long forms, older spellings) map to one name and are
    summed. ``INDONESIA`` appears as a partner (returned goods) and is kept.

Values
    Published in US dollars: ``value_usd`` directly, with no exchange-rate step (``fx_rates``
    is unused), and ``value_local``/``currency_local`` stay blank. Imports are valued CIF,
    exports FOB. Rows are published by port of entry or exit; the clean schema has no port,
    so rows are summed over ports. Periods past the publication frontier recorded in the raw
    file are dropped, as are rows with neither value nor weight.

Quantity and weight
    Net weight is published in kilograms -> ``weight_kg``. The publication has no statistical
    quantity, so ``qty``/``qty_unit`` stay blank.

Raw files
    ``indonesia_bps_T1-T2_MX_hs8_<YYYY>-<YYYY>_pulled<YYYYMMDD>.json``, one per retrieval,
    covering whole calendar years: a JSON envelope with

        frontier   ``{"year", "month"}``: the latest month BPS had published at retrieval,
                   read from a chapter-level total (a sparse 8-digit line cannot show it)
        codes      ``{code: description}`` for the 8-digit lines found
        records    ``[{sumber, year, rows}]``: ``sumber`` 1 = export, 2 = import, and ``rows``
                   the data rows as BPS returns them, with ``tahun`` (year), ``bulan``
                   (``"[05] Mei"``), ``kodehs`` (``"[84715090] <description>"``), ``pod``
                   (port), ``ctr`` (partner), ``value`` (USD) and ``netweight`` (kg)

    The same source row can appear in more than one record of a file (overlapping
    extracts); such repeats, identified by flow, code, month, partner and port, are counted
    once.

Vintages
    A retrieval re-fetches whole years, and a refresh may cover only the latest year, so
    ``build`` merges every retrieval in the directory: files are ranked newest first (by
    ``pulled<YYYYMMDD>`` stamp; same-day files by name, descending, which puts a trailing
    refresh (later first year) ahead of the full retrieval) and each calendar year -- the merge
    key -- is taken whole from the highest-ranked file containing it. The frontier comes from
    the top-ranked file. ``vintage="YYYYMMDD"`` leaves out the retrievals made after that date.

Output
    ``indonesia_T1-T2_MX_<first year>-<last year>_pulled<YYYYMMDD>.csv``. ``pulled_date`` on
    every row and the output stamp are the top-ranked retrieval's date.
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

BOOK = "indonesia"
REPORTER = "indonesia"
SOURCE = "bps_webapi"
RAW_PATTERN = "indonesia_bps_*_pulled*.json"
START_YEAR = 2022            # names the output when the raw files hold no records

# BPS ``sumber`` -> flow.
FLOW_MAP = {1: "X", 2: "M"}

# BPS's upper-case partner labels -> the spellings the other books use. Unlisted labels are
# title-cased ("SINGAPORE" -> "Singapore").
PARTNER_MAP = {
    "UNITED STATES": "United States",
    "UNITED STATES OF AMERICA": "United States",
    "AMERICAN SAMOA": "American Samoa",
    "TAIWAN": "Taiwan",
    "TAIWAN, PROVINCE OF CHINA": "Taiwan",
    "CHINA": "China",
    "CHINA, PEOPLE'S REPUBLIC OF": "China",
    "HONG KONG": "Hong Kong",
    "MACAO": "Macau",
    "MACAU": "Macau",
    "KOREA, REPUBLIC OF": "South Korea",
    "KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF": "North Korea",
    "GERMANY, FED. REP. OF": "Germany",
    "GERMANY": "Germany",
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
    "NETHERLANDS": "Netherlands",
    "INDONESIA": "Indonesia",
}

# Indonesian month names as they appear in ``bulan``. The bracketed number is authoritative;
# the name is the fallback when a label has no number.
MONTH_ID = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "nopember": 11,
    "november": 11, "desember": 12,
}


def parse_period(row: dict) -> str | None:
    """``bulan`` ``'[05] Mei'`` with ``tahun`` ``'2025'`` -> ``'2025-05'``; None if unparseable."""
    bulan, tahun = str(row.get("bulan", "")), str(row.get("tahun", "")).strip()
    m = re.match(r"\s*\[(\d{1,2})\]", bulan)
    month = int(m.group(1)) if m else MONTH_ID.get(re.sub(r"[^a-z]", "", bulan.lower()), 0)
    if not (1 <= month <= 12) or not re.fullmatch(r"\d{4}", tahun):
        return None
    return f"{tahun}-{month:02d}"


def parse_code(row: dict) -> tuple[str, str]:
    """``kodehs`` ``'[84715090] Oth processing unit...'`` -> code and description,
    ``('84715090', 'Oth processing unit...')``; ``("", <label>)`` if the label has no code."""
    raw = str(row.get("kodehs", "")).strip()
    m = re.match(r"\[(\w+)\]\s*(.*)", raw)
    if not m:
        return "", raw
    return m.group(1).strip(), re.sub(r"\s+", " ", m.group(2)).strip()


def normalize_partner(raw) -> str:
    """Canonical English name for a BPS partner label."""
    raw = str(raw).strip()
    return PARTNER_MAP.get(raw.upper(), raw.title())


def process(records: list[dict], frontier: tuple[int, int], pulled_date: str) -> pd.DataFrame:
    """Raw records -> the clean frame.

    Drops periods past ``frontier`` and repeated source rows, then sums over ports and over
    partner labels that map to one name.
    """
    fy, fm = frontier
    rows, seen = [], set()
    for rec in records:
        flow = FLOW_MAP[int(rec["sumber"])]
        for r in rec["rows"]:
            code, desc = parse_code(r)
            period = parse_period(r)
            if not code or not period:
                continue
            if (int(period[:4]), int(period[5:7])) > (fy, fm):
                continue
            # A source row's identity includes its port: distinct port rows are summed below,
            # while the same row fetched twice must count once.
            ident = (flow, code, period, str(r.get("ctr", "")), str(r.get("pod", "")))
            if ident in seen:
                continue
            seen.add(ident)
            try:
                value = float(r.get("value") or 0)
                weight = float(r.get("netweight") or 0)
            except (TypeError, ValueError):
                continue
            if value == 0 and weight == 0:
                continue
            rows.append({
                "flow": flow, "partner": normalize_partner(r.get("ctr", "")),
                "code_native": code, "code_hs6": code[:6],
                "code_desc": desc or BASKET_DESC.get(code[:6], ""),
                "period": period, "value_usd": value, "weight_kg": weight,
            })

    if not rows:
        return pd.DataFrame(columns=SCHEMA_COLS)

    df = pd.DataFrame(rows)
    df = (df.groupby(["flow", "partner", "code_native", "code_hs6", "period"], as_index=False)
            .agg(value_usd=("value_usd", "sum"), weight_kg=("weight_kg", "sum"),
                 code_desc=("code_desc", "first")))
    df["value_usd"] = df["value_usd"].round(2)
    df["weight_kg"] = df["weight_kg"].round(2)

    df["reporter"] = REPORTER
    df["source"] = SOURCE
    df["value_local"] = ""
    df["currency_local"] = ""
    df["qty"] = ""
    df["qty_unit"] = ""
    df["pulled_date"] = pulled_date
    return finalize(df)


def load_records(files: list[Path]) -> tuple[list[dict], tuple[int, int]]:
    """Merged records and the frontier from files ordered newest first.

    Each year's records come from the first file that contains that year, since a retrieval
    replaces whole years; the frontier comes from the first file.
    """
    frontier, seen_years, records = None, set(), []
    for path in files:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if frontier is None:
            frontier = (raw["frontier"]["year"], raw["frontier"]["month"])
        take = {int(r["year"]) for r in raw["records"]} - seen_years
        records.extend(r for r in raw["records"] if int(r["year"]) in take)
        seen_years |= take
    return records, frontier


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """Build the Indonesia clean book, each year from its newest retrieval in ``raw_dir``.

    With ``vintage``, retrievals made after that date are ignored. ``fx_rates`` is unused:
    the publication reports US dollars.
    """
    files = raw_files(raw_dir, RAW_PATTERN, vintage)
    stamp = pulled_stamp(files[0])
    records, frontier = load_records(files)
    years = sorted({int(r["year"]) for r in records}) or [START_YEAR]
    df = process(records, frontier, iso_date(stamp))
    return [CleanBook(df, f"indonesia_T1-T2_MX_{min(years)}-{max(years)}_pulled{stamp}.csv")]
