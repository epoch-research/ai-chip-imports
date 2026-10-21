"""Mexico: INEGI monthly merchandise trade balance (BCMM), tariff-line detail (book ``mexico``).

Publication
    Balanza Comercial de Mercancías de México, monthly, published by INEGI (jointly compiled
    with SAT, the Secretaría de Economía and Banco de México), read from INEGI's monthly
    foreign-trade information cube (January 2023 onward). The book covers a corridor set of
    nine partners -- United States, Taiwan, China, South Korea, Japan, Netherlands, Germany,
    Singapore, Malaysia -- not every partner; all-partner HS-6 coverage of Mexico comes from
    the UN Comtrade companion book.

Codes
    Fracción arancelaria (8-digit TIGIE) plus NICO (2 digits): 10-digit national lines.
    ``code_native`` is the line with its dots removed, ``code_hs6`` its first six digits.
    ``code_desc`` is the cube's Spanish caption for the line (the TIGIE unit, Pza or Kg, is
    part of the cube's caption but not of the description kept here); lines missing from the
    file's caption catalogue get a blank description. One 8-digit line, 8517.62.00
    ("operaciones especiales reportadas en códigos a 6 dígitos"), is a special-operations
    stub, not a fracción: never sum it with the 10-digit lines.

Flows
    "Importaciones" -> M, "Exportaciones" -> X. Mexico publishes no re-export split, and
    imports are attributed to the country of origin rather than the country of consignment.

Partners
    The cube's Spanish captions are translated to the English names used across books
    (``PARTNER_MAP``); unmapped captions are kept verbatim.

Values
    The cube's measure is US dollars, plain (not thousands), so cells go straight to
    ``value_usd`` with no exchange rate (``fx_rates`` is unused). Both flows are valued FOB
    (imports too, unusually). A cell of "C" is a value INEGI masks because only one or two
    declarants are behind it: the row is kept with a blank ``value_usd`` (masked, not zero)
    so the corridor stays visible; blank cells mean no trade and produce no row.

Quantity and weight
    The cube carries no quantity or weight: ``qty``, ``qty_unit`` and ``weight_kg`` are blank.

Raw files
    ``mexico_bcmm_*_pulled<YYYYMMDD>.json``, one per retrieval, e.g.
    ``mexico_bcmm_T1-T2_MX_frac10_2023-2026_pulled20260914.json``: a JSON envelope around the
    cube's cells, with keys
      ``pulled_date``  retrieval date, ``YYYY-MM-DD``
      ``leaves``       the cube's caption catalogue for the basket lines,
                       ``{code_native: {"caption", "unit", "desc", "hs6", "is_stub"}}``
      ``records``      one entry per non-blank cell,
                       ``{"period": "YYYY-MM", "code_native", "partner_es", "flow": "M"|"X",
                       "raw": <cell text as displayed, e.g. "1,234,567" or "C">}``

Vintages
    Every retrieval re-reads the whole series, so ``build`` reads the newest file alone;
    ``vintage="YYYYMMDD"`` pins the file stamped that day.

Output
    ``mexico_T1-T2_MX_<first year>-<last year>_pulled<YYYYMMDD>.csv``, the years those of the
    months the archive spans (masked cells included). ``pulled_date`` is the archive's
    ``pulled_date`` field, falling back to the file's ``pulled<YYYYMMDD>`` stamp.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ._common import (
    SCHEMA_COLS,
    CleanBook,
    finalize,
    iso_date,
    pulled_stamp,
    select_vintage,
)

BOOK = "mexico"
REPORTER = "mexico"
SOURCE = "inegi_bcmm"
RAW_PATTERN = "mexico_bcmm_*_pulled*.json"

# INEGI's Spanish partner captions -> the English names used across books. Unmapped captions
# pass through verbatim so nothing is silently dropped.
PARTNER_MAP = {
    "Estados Unidos": "United States",
    "Taiwán (República de China)": "Taiwan",
    "China": "China",
    "Hong Kong (Región Administrativa Especial de la República de China)": "Hong Kong",
    "Corea del Sur": "South Korea",
    "Corea del Norte": "North Korea",
    "Japón": "Japan",
    "Alemania": "Germany",
    "Países Bajos": "Netherlands",
    "Reino Unido": "United Kingdom",
    "Singapur": "Singapore",
    "Malasia": "Malaysia",
    "Vietnam": "Vietnam",
    "Tailandia": "Thailand",
    "India": "India",
    "Canadá": "Canada",
    "Brasil": "Brazil",
    "Irlanda": "Ireland",
    "Israel": "Israel",
    "Suiza": "Switzerland",
    "Francia": "France",
    "Italia": "Italy",
    "España": "Spain",
    "Filipinas": "Philippines",
    "Indonesia": "Indonesia",
    "Costa Rica": "Costa Rica",
    "México": "Mexico",
}

def parse_value(raw) -> tuple[int | None, bool]:
    """A cube cell -> (value, is_confidential).

    ``""`` -> (None, False), no trade; ``"C"`` -> (None, True), masked;
    ``"1,234"`` -> (1234, False).
    """
    raw = (raw or "").strip()
    if raw == "":
        return None, False
    if raw.upper() == "C":
        return None, True
    digits = re.sub(r"[^\d]", "", raw)
    return (int(digits), False) if digits else (None, False)


def normalize_partner(caption: str) -> str:
    """INEGI partner caption -> the English name used across books."""
    return PARTNER_MAP.get(caption, caption)


def process(records: list[dict], leaves: dict, pulled_date: str) -> pd.DataFrame:
    """Clean-schema frame from the archived cube cells.

    Masked ("C") cells are kept with a blank ``value_usd``; cells with neither a value nor a
    mask are dropped.
    """
    leaves = leaves or {}
    rows = []
    for r in records:
        code_native = str(r["code_native"])
        value, masked = parse_value(r["raw"])
        if value is None and not masked:
            continue
        rows.append({
            "flow": r["flow"], "partner": normalize_partner(r["partner_es"]),
            "code_native": code_native, "code_hs6": code_native[:6],
            "code_desc": leaves.get(code_native, {}).get("desc", ""), "period": r["period"],
            "value_usd": "" if value is None else value,
        })
    if not rows:
        return pd.DataFrame(columns=SCHEMA_COLS)

    df = pd.DataFrame(rows)
    df["reporter"] = REPORTER
    df["source"] = SOURCE
    for col in ("value_local", "currency_local", "qty", "qty_unit", "weight_kg"):
        df[col] = ""
    df["pulled_date"] = pulled_date
    return finalize(df)


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """Build the Mexico book from the newest cube archive in ``raw_dir``, or the one stamped
    ``vintage``.

    ``fx_rates`` is unused: values are published in USD.
    """
    path = select_vintage(raw_dir, RAW_PATTERN, vintage)[-1]
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    stamp = pulled_stamp(path)
    records = raw["records"]
    if not records:
        raise ValueError(f"no cube cells in {path}")
    df = process(records, raw.get("leaves", {}), raw.get("pulled_date") or iso_date(stamp))

    # Coverage is named after the months the archive spans, masked cells included.
    periods = sorted({r["period"] for r in records})
    name = f"mexico_T1-T2_MX_{periods[0][:4]}-{periods[-1][:4]}_pulled{stamp}.csv"
    masked = int((df["value_usd"] == "").sum())
    print(f"  mexico: {path.name}: {len(df):,} rows, {masked:,} masked (C) cells")
    return [CleanBook(df, name)]
