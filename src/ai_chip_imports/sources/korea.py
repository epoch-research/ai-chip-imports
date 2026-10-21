"""South Korea: Korea Customs Service (KCS) trade statistics, HSK 10-digit (book ``korea``).

Publication
    Korea Customs Service, Trade Statistics (수출입무역통계), the country-by-commodity report
    (국가별 품목별), monthly, confirmed (확정) basis. KCS also publishes a provisional (잠정)
    basis closer to real time; it is not used here.

Codes
    HSK 10-digit (Harmonised System of Korea), one schedule for both flows. Trade is reported
    on the 10-digit line Korea declares at, which is not always the finest line in the code
    list (8542.32.4000 carries the trade while its finer splits carry none). ``code_native`` is
    the reported code as a string, ``code_hs6`` its first six digits, and ``code_desc`` the
    official English label from the HSK code list (the Korean label when the list has no
    English one).

Flows
    Every report row carries both directions; it becomes up to two clean rows, X from the
    export fields and M from the import fields. Korea has no re-export concept, so DX and RX
    are never emitted.

Partners
    Partners are ISO 3166 alpha-2 codes. They are named from a fixed map for the partners that
    matter to the analysis (spellings shared with the other books), then from the English
    partner list stored in the raw file, then left as the alpha-2 code.

Values
    Published in thousands of US dollars: ``value_usd`` = value x 1,000, with no
    exchange-rate step (``fx_rates`` is unused), and ``value_local``/``currency_local`` stay
    blank. Exports are valued FOB, imports CIF. A direction with weight but zero value keeps a
    blank ``value_usd``; a direction with neither value nor weight is not emitted.

Quantity and weight
    Weight is published in metric tons: ``weight_kg`` = tons x 1,000, rounded to the kilogram.
    The report carries no statistical quantity, so ``qty``/``qty_unit`` stay blank.

Raw files
    ``korea_kcs_T1-T2_<hs6>_<YYYYMM>-<YYYYMM>_pulled<YYYYMMDD>.json``, one file per HS-6 basket
    parent per retrieval, covering the months in the name. Each is a JSON envelope with

        hs6            the HS-6 parent
        children       the HSK code list under it: ``{code: {"kor": ..., "eng": ...}}``
        partner_names  English partner names: ``{alpha-2: name}``
        responses      a list of report extracts, each with ``items``: the report rows, holding
                       cntyCd (alpha-2), cntyNm (Korean name), hsSgn (HSK code), korePrlstNm,
                       priodTitle (``YYYY.MM``), expUsdAmt, impUsdAmt (USD thousands) and
                       expTtwg, impTtwg (tons), as the publisher returns them (padded strings
                       with thousands separators)

Vintages
    Months are re-published as later confirmations arrive, so a retrieval may cover only the
    trailing months. ``build`` merges every retrieval in the directory and keeps, per (flow,
    code, partner, month) -- the merge key -- the row from the newest retrieval (same-day
    files: the later file name). ``vintage="YYYYMMDD"`` leaves out the retrievals made after
    that date.

Output
    ``korea_T1-T2_KR_<first year>-<last year>_pulled<YYYYMMDD>.csv``, stamped with the newest
    retrieval read. ``pulled_date`` on each row is the retrieval date of the file the row came
    from.
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
    keep_newest,
    pulled_stamp,
    raw_files,
    year_span,
)

BOOK = "korea"
REPORTER = "korea"
SOURCE = "kcs_cts"
RAW_PATTERN = "korea_kcs_T1-T2_*_pulled*.json"
START_YEAR = 2022            # names the output when no rows are produced

# Alpha-2 -> partner name, pinning the spellings the other books use (notably the China /
# Taiwan / Hong Kong / Korea labels). Takes precedence over the publisher's English names.
PARTNER_MAP = {
    "CN": "China", "TW": "Taiwan", "HK": "Hong Kong", "MO": "Macau",
    "KR": "South Korea", "KP": "North Korea", "US": "United States",
    "JP": "Japan", "SG": "Singapore", "MY": "Malaysia", "VN": "Vietnam",
    "RU": "Russia", "GB": "United Kingdom", "AE": "United Arab Emirates",
    "CH": "Switzerland", "IN": "India", "TH": "Thailand", "PH": "Philippines",
    "ID": "Indonesia", "DE": "Germany", "NL": "Netherlands", "FR": "France",
    "IT": "Italy", "IE": "Ireland", "IL": "Israel", "CA": "Canada",
    "MX": "Mexico", "BR": "Brazil", "TR": "Turkey", "ES": "Spain",
}

# A data row's period label, e.g. "2026.05"; the report also returns subtotal rows that lack one.
_PERIOD_RE = re.compile(r"^\d{4}\.\d{2}$")

# A clean row's identity, the key on which retrievals are merged.
_KEY = ["flow", "code_native", "partner", "period"]

# (flow, value field, weight field) for the two directions every report row carries.
_DIRECTIONS = (("X", "expUsdAmt", "expTtwg"), ("M", "impUsdAmt", "impTtwg"))


def _num(raw) -> int:
    """A padded, comma-grouped amount (``'  3,888,555 '``) as an int; blank or ``'-'`` is 0."""
    digits = re.sub(r"[^\d]", "", str(raw or ""))
    return int(digits) if digits else 0


def _weight_kg(raw):
    """A weight in tons (``' 17.7 '``) as whole kilograms, or ``''`` when zero or blank."""
    s = re.sub(r"[^\d.]", "", str(raw or ""))
    tons = float(s) if s not in ("", ".") else 0.0
    return int(round(tons * 1000)) if tons > 0 else ""


def _partner_name(cnty_cd: str, partner_names: dict) -> str:
    """Partner name for an alpha-2 code: the fixed map, then the publisher's English name,
    then the code itself."""
    return PARTNER_MAP.get(cnty_cd) or partner_names.get(cnty_cd) or cnty_cd


def _code_desc(code: str, children: dict) -> str:
    """Label of an HSK code from the code list: English where given, else Korean."""
    info = children.get(code) or {}
    return info.get("eng") or info.get("kor", "")


def _rows_of(path: Path) -> list[dict]:
    """Clean-schema rows (both directions) from one raw file."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    children = blob.get("children", {})
    partner_names = blob.get("partner_names", {})
    pulled_date = iso_date(pulled_stamp(path))
    rows = []
    for resp in blob.get("responses", []):
        for it in resp.get("items", []):
            code = str(it.get("hsSgn", "")).strip()
            cnty = str(it.get("cntyCd", "")).strip()
            title = str(it.get("priodTitle", ""))
            if not (code and cnty and _PERIOD_RE.match(title)):
                continue
            partner = _partner_name(cnty, partner_names)
            desc = _code_desc(code, children)
            for flow, value_key, weight_key in _DIRECTIONS:
                val = _num(it.get(value_key))
                wt = _weight_kg(it.get(weight_key))
                if val == 0 and wt == "":
                    continue                    # no trade in this direction
                rows.append({
                    "reporter": REPORTER, "flow": flow, "partner": partner,
                    "code_native": code, "code_hs6": code[:6], "code_desc": desc,
                    "period": title.replace(".", "-"),
                    "value_usd": val * 1000 if val > 0 else "",
                    "value_local": "", "currency_local": "", "qty": "", "qty_unit": "",
                    "weight_kg": wt, "source": SOURCE, "pulled_date": pulled_date,
                })
    return rows


def build_frame(files) -> pd.DataFrame:
    """The clean frame from a set of raw files, keeping the newest retrieval of every cell.

    Files are read in name order and a cell is keyed on (flow, code_native, partner, period),
    so the same cell re-published in a later retrieval replaces the earlier figure.
    """
    rows = []
    for path in sorted(files):
        rows.extend(_rows_of(path))
    if not rows:
        return pd.DataFrame(columns=SCHEMA_COLS)
    return finalize(keep_newest(pd.DataFrame(rows), _KEY))


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """Build the Korea clean book from every raw file in ``raw_dir``, newest retrieval winning.

    With ``vintage``, retrievals made after that date are ignored. ``fx_rates`` is unused:
    the publication reports US dollars.
    """
    files = raw_files(raw_dir, RAW_PATTERN, vintage)
    df = build_frame(files)
    coverage = year_span(df["period"]) if not df.empty else str(START_YEAR)
    return [CleanBook(df, f"korea_T1-T2_KR_{coverage}_pulled{pulled_stamp(files[0])}.csv")]
