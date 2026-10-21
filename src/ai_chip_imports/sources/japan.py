"""Japan: Ministry of Finance / Japan Customs, Trade Statistics of Japan (book ``japan``).

Publication
    The monthly "Commodity by Country" tables of the Trade Statistics of Japan, distributed as
    one CSV per flow x calendar year x HS section on e-Stat. Section XVI (chapters 84-85) holds
    the whole basket (8471, 8473, 8517, 8523, 8542).

Codes
    Native 9-digit statistical codes. Imports and exports use different schemes (the import
    Customs Tariff Schedule and the Export Statistical Schedule), so a 9-digit code must not
    be tracked across flows; HS-6 is the only level comparable between M and X. The CSVs carry
    no descriptions: import descriptions below are Japan Customs' English wording; the export
    schedule is published in Japanese only, so export descriptions are translations (using
    Japan Customs' English wherever the Japanese text is identical to an import line).

Flows
    Exports -> X, imports -> M (the file's ``Exp or Imp`` marker: 1 export, 2 import). Japan
    publishes no re-export split, so RX and DX never occur.

Partners
    Japan Customs' numeric country codes (not ISO, e.g. 103 Korea, 105 China, 106 Taiwan,
    304 USA), mapped to English names with Japan Customs' country-code list and normalized to
    the spellings the other books use.

Values
    Values are in thousands of yen -> x1000 -> ``value_local`` (JPY). ``value_usd`` divides it
    by the monthly period-average JPY-per-USD rate from the exchange-rate table; months
    without a rate leave it blank. Imports are valued CIF, exports FOB.

Quantity and weight
    Each row carries up to two (unit, quantity) pairs whose slot order is not fixed, so each
    pair is routed by its unit label: ``KG`` -> ``weight_kg``, any other unit (e.g. ``NO``,
    number of items) -> ``qty`` and ``qty_unit``.

Raw files
    ``japan_estat_T1-T2_MX_hs9_<YYYY>-<YYYY>_pulled<YYYYMMDD>.json``: a JSON envelope with

        pulled_date   retrieval date
        country_map   {numeric partner code: English name} from Japan Customs' code list
        files         one entry per downloaded Section XVI CSV: flow (M/X), year, revision,
                      frontier_month (latest month with any non-zero value anywhere in the
                      full chapter 84-85 file), the file's published title, update date and
                      e-Stat file id, and ``rows``: the CSV's rows for the basket's HS-6
                      subheadings, each as a {column header: cell text} object, cells verbatim

    The CSVs are wide: per row, an annual total triplet (``Quantity1-Year``,
    ``Quantity2-Year``, ``Value-Year``) and twelve monthly triplets (``Quantity1-<Mon>``,
    ``Quantity2-<Mon>``, ``Value-<Mon>``). Only the monthly columns are used; the annual total
    is not a thirteenth month. The current year's file shows zeros for months not yet
    published, so months after the file's ``frontier_month`` are dropped. Each file's revision
    state (Provisional, Detailed, Revised, Fixed) is carried per row in the extra column
    ``jp_revision``.

Vintages
    A refresh may re-retrieve only the trailing years, so ``build`` merges every retrieval in
    the directory. Files are ranked newest first (by ``pulled<YYYYMMDD>`` stamp; same-day files
    by name, descending, so a same-day trailing refresh outranks the full retrieval) and each
    flow x year CSV is taken whole from the highest-ranked file carrying it, since that one has
    the latest revision state: the merge key is (flow, year). Partner names come from the
    top-ranked file's country map. ``vintage="YYYYMMDD"`` leaves out the retrievals made after
    that date.

Output
    ``japan_T1-T2_MX_<first year>-<last year>_pulled<YYYYMMDD>.csv``, spanning every year any
    retrieval read covers, with the extra column ``jp_revision``. ``pulled_date`` on every row
    and the output stamp are the top-ranked retrieval's date.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ._common import (
    BASKET_DESC,
    CleanBook,
    finalize,
    iso_date,
    pulled_stamp,
    raw_files,
    to_usd,
)
from ._common import SCHEMA_COLS as _BASE_COLS

BOOK = "japan"
RAW_PATTERN = "japan_estat_*_pulled*.json"

REPORTER = "japan"
SOURCE = "estat_csv"
CURRENCY = "JPY"

# Standard clean schema plus the revision state of the file each row came from.
SCHEMA_COLS = _BASE_COLS + ["jp_revision"]

# The file's own ``Exp or Imp`` marker; rows without one take the flow of their file.
EXPIMP_FLOW = {"1": "X", "2": "M"}

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# 9-digit descriptions keyed by (flow, code), because the import and export schemes differ;
# codes not listed fall back to the basket label of their HS-6 code.
# Imports: Japan's Tariff Schedule (Statistical Code for Import), English, verbatim. Exports:
# the Export Statistical Schedule, translated from Japanese. Wording checked unchanged across
# the 2022-2026 vintages of both schedules.
DESC_9 = {
    ("M", "847149000"): "Other ADP machines, presented in the form of systems",
    ("M", "847150000"): "Processing units (excl. 8471.41/8471.49), w/n containing 1-2 of "
                        "storage/input/output units in the same housing",
    ("M", "847180000"): "Other units of automatic data processing machines",
    ("M", "847330011"): "DRAM modules — one or more DRAMs mounted on a circuit board with a "
                        "terminal to connect to ADP machines or units thereof",
    ("M", "847330019"): "Parts & accessories of ADP machines or units thereof — other",
    ("M", "847330090"): "Parts & accessories of the machines of heading 84.71 — other",
    ("M", "851762010"): "Switching and routing apparatus",
    ("M", "851762090"): "Reception/conversion/transmission or regeneration machines — other",
    ("M", "852351000"): "Solid-state non-volatile storage devices",
    ("M", "854231010"): "Processors & controllers — uncased (unpackaged)",
    ("M", "854231020"): "Processors & controllers, mounted — hybrid integrated circuits",
    ("M", "854231031"): "MPU (microprocessor unit)",
    ("M", "854231032"): "MCU (microcontroller unit)",
    ("M", "854231033"): "DSP (digital signal processor)",
    ("M", "854231039"): "Processors & controllers, mounted — other",
    ("M", "854232011"): "Memories, uncased — DRAM, MOS digital monolithic ICs",
    ("M", "854232019"): "Memories, uncased — other",
    ("M", "854232021"): "Memories, mounted, RAM — DRAM, MOS digital monolithic ICs",
    ("M", "854232029"): "Memories, mounted, RAM — other",
    ("M", "854232031"): "Memories, mounted, ROM — flash memory",
    ("M", "854232039"): "Memories, mounted, ROM — other",
    ("M", "854232090"): "Memories, mounted — other (neither RAM nor ROM)",
    ("M", "854239010"): "Other electronic ICs — uncased (unpackaged)",
    ("M", "854239091"): "Other electronic ICs, mounted — hybrid integrated circuits",
    ("M", "854239099"): "Other electronic ICs, mounted — other",
    ("X", "847149000"): "Other ADP machines, presented in the form of systems",
    ("X", "847150000"): "Processing units (excl. 8471.41/8471.49), w/n containing 1-2 of "
                        "storage/input/output units in the same housing",
    ("X", "847180000"): "Other units of automatic data processing machines",
    ("X", "847330000"): "Parts & accessories of the machines of heading 84.71 "
                        "(UNDIVIDED — no DRAM-module split on the export side)",
    ("X", "851762000"): "Reception/conversion/transmission or regeneration machines, incl. "
                        "switching and routing apparatus (UNDIVIDED on the export side)",
    ("X", "852351000"): "Solid-state non-volatile storage devices",
    ("X", "854231100"): "Processors & controllers — uncased (unpackaged)",
    ("X", "854231910"): "Processors & controllers, mounted — hybrid integrated circuits",
    ("X", "854231991"): "MPU (microprocessor unit)",
    ("X", "854231992"): "MCU (microcontroller unit)",
    ("X", "854231993"): "DSP (digital signal processor)",
    ("X", "854231999"): "Processors & controllers, mounted — other",
    ("X", "854232100"): "Memories — UNCASED (unpackaged: wafers / bare die, any memory type)",
    ("X", "854232911"): "Memories, mounted, RAM — DRAM (no monolithic qualifier on export)",
    ("X", "854232919"): "Memories, mounted, RAM — other",
    ("X", "854232921"): "Memories, mounted, ROM — flash memory (packaged NAND)",
    ("X", "854232929"): "Memories, mounted, ROM — other",
    ("X", "854232990"): "Memories, mounted — other (neither RAM nor ROM)",
    ("X", "854239100"): "Other electronic ICs — uncased (unpackaged)",
    ("X", "854239910"): "Other electronic ICs, mounted — hybrid integrated circuits",
    ("X", "854239990"): "Other electronic ICs, mounted — other",
}

# Japan Customs' English country names -> the names the other books use.
PARTNER_MAP = {
    "Republic of Korea": "South Korea",
    "Korea, Republic of": "South Korea",
    "North Korea": "North Korea",
    "People's Republic of China": "China",
    "Taiwan": "Taiwan",
    "Hong Kong": "Hong Kong",
    "Viet Nam": "Vietnam",
    "United States of America": "United States",
    "U.S.A.": "United States",
    "Russian Federation": "Russia",
    "Russia": "Russia",
    "Czech Republic": "Czech Republic",
    "Germany": "Germany",
    "United Kingdom": "United Kingdom",
    "Netherlands": "Netherlands",
    "United Arab Emirates": "United Arab Emirates",
    "Singapore": "Singapore",
    "Malaysia": "Malaysia",
    "Thailand": "Thailand",
    "Philippines": "Philippines",
    "Indonesia": "Indonesia",
    "India": "India",
    "Ireland": "Ireland",
    "Mexico": "Mexico",
}

_DEDUP_KEYS = ["flow", "partner", "code_native", "period"]


def _num(v) -> float:
    """CSV cell -> float; blank or malformed cells count as zero."""
    try:
        return float(str(v).strip() or 0)
    except (TypeError, ValueError):
        return 0.0


def _hs_code(row: dict) -> str:
    """The 9-digit code of a CSV row, without the apostrophes the CSV quotes it with
    (``'847150000'``) to keep its leading digits."""
    return (row.get("HS") or "").strip().strip("'").strip()


def melt(rows: list[dict], flow: str, year: int, revision: str,
         country_map: dict[str, str], frontier_month: int) -> list[dict]:
    """Wide CSV rows of one flow x year file -> long monthly records.

    Only the twelve monthly triplets are read (the annual total columns are ignored); months
    after ``frontier_month`` are dropped, as are months with no value, quantity or weight.
    """
    out = []
    for row in rows:
        hs = _hs_code(row)
        code_hs6 = hs[:6]
        partner_code = (row.get("Country") or "").strip()
        partner_name = country_map.get(partner_code, partner_code)
        partner = PARTNER_MAP.get(partner_name, partner_name)
        row_flow = EXPIMP_FLOW.get((row.get("Exp or Imp") or "").strip(), flow)
        units = [((row.get("Unit1") or "").strip(), "Quantity1"),
                 ((row.get("Unit2") or "").strip(), "Quantity2")]
        for i, mon in enumerate(MONTH_ABBR, 1):
            if frontier_month and i > frontier_month:
                continue
            value_k = _num(row.get(f"Value-{mon}"))
            qty = weight = None
            qty_unit = ""
            for unit, prefix in units:
                if not unit:
                    continue
                q = _num(row.get(f"{prefix}-{mon}"))
                if not q:
                    continue
                if unit.upper() == "KG":
                    weight = (weight or 0) + q
                else:
                    qty = (qty or 0) + q
                    qty_unit = unit
            if not value_k and qty is None and weight is None:
                continue
            out.append({
                "flow": row_flow, "partner": partner, "code_native": hs, "code_hs6": code_hs6,
                "code_desc": DESC_9.get((row_flow, hs)) or BASKET_DESC.get(code_hs6, ""),
                "period": f"{year:04d}-{i:02d}",
                "value_local": value_k * 1000.0,
                "qty": qty, "qty_unit": qty_unit, "weight_kg": weight,
                "jp_revision": revision,
            })
    return out


def standardize(records: list[dict], pulled_date: str,
                fx_rates: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Long records -> the clean frame.

    A (flow, partner, code, period) that occurs more than once keeps its last record, so
    overlapping inputs never double-count.
    """
    if not records:
        return pd.DataFrame(columns=SCHEMA_COLS)

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=_DEDUP_KEYS, keep="last")

    df["reporter"] = REPORTER
    df["source"] = SOURCE
    df["currency_local"] = CURRENCY
    df["value_local"] = df["value_local"].round().astype("int64")

    df["value_usd"] = to_usd(df["value_local"], df["period"], CURRENCY, fx_rates)
    df["pulled_date"] = pulled_date
    return finalize(df, SCHEMA_COLS)


def build(raw_dir: Path, fx_rates: dict[tuple[str, str], float],
          vintage: str | None = None) -> list[CleanBook]:
    """Clean Japan book from every raw retrieval in ``raw_dir``, newest file winning.

    Each flow x year CSV is taken from the highest-ranked retrieval that carries it, so the
    output spans every year any retrieval covers. With ``vintage``, retrievals made after
    that date are ignored.
    """
    files = raw_files(raw_dir, RAW_PATTERN, vintage)
    stamp = pulled_stamp(files[0])
    kept: dict[tuple[str, int], dict] = {}
    country_map: dict[str, str] = {}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        country_map = country_map or raw.get("country_map", {})
        for f in raw["files"]:
            kept.setdefault((f["flow"], int(f["year"])), f)

    records = []
    for (flow, year), f in sorted(kept.items()):
        records.extend(melt(f["rows"], flow, year, f.get("revision", ""),
                            country_map, f.get("frontier_month", 12)))
    df = standardize(records, iso_date(stamp), fx_rates)
    blank = int((df["value_usd"] == "").sum())
    if blank:
        print(f"  [japan] value_usd blank on {blank:,} rows (no exchange rate for the month)")
    years = sorted({y for _, y in kept})
    return [CleanBook(df, f"japan_T1-T2_MX_{years[0]}-{years[-1]}_pulled{stamp}.csv")]
