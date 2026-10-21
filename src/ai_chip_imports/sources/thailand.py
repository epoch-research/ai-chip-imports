"""Thailand: Ministry of Commerce (MOC) trade statistics (book ``thailand``).

Publication
    Thailand's Trade Statistic Report System, published by the Ministry of Commerce from Thai
    Customs data: exports and imports by harmonized code and partner country, monthly, from
    January 2022.

Codes
    AHTN 2022 with Thailand's 3-digit statistical suffix, giving 11-digit lines (``000`` for
    the line itself; other suffixes for national splits such as ``800`` used goods, ``899``
    electronic waste and ``999`` CKD kits). Three levels are kept as separate rows: the HS-4
    parents 8471/8473/8542 (``code_hs6`` repeats the 4-digit code on those), the basket's nine
    HS-6 codes, and the 11-digit lines under those nine codes (``code_native`` is the 11-digit
    code; the 8-digit level is their sum and is not repeated). 11-digit rows carry the
    official English name from the publisher's AHTN 2022 nomenclature; HS-6 and HS-4 rows
    the short basket label.

Flows
    "Export" is total exports including re-exports -> X; "Import" is gross imports -> M. The
    publication has no domestic-export / re-export split by code and partner.

Partners
    ISO-2 partner codes, named from a fixed map for the spellings shared with the other books,
    else the publisher's English name, title-cased. World / total rows are dropped.

Values
    Each record carries the month's value in both US dollars and Thai baht, so ``value_usd``
    is the published dollar figure (no exchange-rate step; ``fx_rates`` is unused) and
    ``value_local`` the baht figure, both rounded to 2 decimals. Only the monthly fields are
    used, never the year-to-date ``acc_*`` twins. Records with no monthly value or quantity
    are dropped: the publication returns zero-filled records for months not yet published and
    for partners with year-to-date but no current-month trade. Imports are valued c.i.f.,
    exports f.o.b.

Quantity and weight
    ``quantity`` is only meaningful at the 11-digit level (it reads 0 wherever the lines
    under a code mix statistical units), so ``qty`` is filled on 11-digit rows only. The
    records carry no unit; ``qty_unit`` comes from ``UNIT8_MAP`` (``u`` pieces or ``kg``),
    the unit Thailand declares for each 8-digit line in its own tariff-line submissions to
    UN Comtrade (December 2024), inherited by the 11-digit lines beneath. Lines never seen
    with a unit stay blank. There is no weight field.

Raw files
    ``thailand_moc_T1-T2_<X|M>_<hs4|hs6|hs11>_<YYYYMM>-<YYYYMM>_pulled<YYYYMMDD>[_part<N>].json``:
    one file per flow, code level and retrieval, a long retrieval possibly split into parts.
    Each is a JSON envelope with ``flow``, ``pass`` (the code level), ``code_meta`` (code ->
    description) and ``queries``, a list of ``{"code", "year", "month", "rows"}`` whose rows
    are the publisher's records verbatim: ``country_code`` (ISO-2), ``country_name_en``,
    ``quantity``, ``value_usd``, ``value_baht`` and their ``acc_*`` twins, all as strings.

Vintages
    The publisher revises recent months, and refreshes are retrieved as short trailing
    windows on top of an earlier full history, so no single retrieval holds the whole book.
    ``build`` merges every retrieval in the directory and keeps, per (flow, code, partner,
    month) -- the merge key -- the record from the newest retrieval (same-day files: the later
    file name). ``vintage="YYYYMMDD"`` leaves out the retrievals made after that date.

Output
    ``thailand_T1-T2_MX_<first year>-<last year>_pulled<YYYYMMDD>.csv``, stamped with the
    newest retrieval read. ``pulled_date`` on each row is the retrieval date of the file the
    row came from.
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
    keep_newest,
    pulled_stamp,
    raw_files,
    year_span,
)

BOOK = "thailand"
REPORTER = "thailand"
SOURCE = "moc_api"
RAW_PATTERN = "thailand_moc_T1-T2_*_pulled*.json"

# ISO-2 partner code -> canonical English name used across books. Other partners take the
# publisher's English name, title-cased.
PARTNER_MAP = {
    "CN": "China", "TW": "Taiwan", "HK": "Hong Kong", "MO": "Macau",
    "KR": "South Korea", "KP": "North Korea", "US": "United States",
    "JP": "Japan", "SG": "Singapore", "MY": "Malaysia", "VN": "Vietnam",
    "RU": "Russia", "GB": "United Kingdom", "AE": "United Arab Emirates",
    "CH": "Switzerland", "IN": "India", "TH": "Thailand", "PH": "Philippines",
    "ID": "Indonesia", "DE": "Germany", "NL": "Netherlands", "FR": "France",
    "IT": "Italy", "IE": "Ireland", "IL": "Israel", "CA": "Canada",
    "MX": "Mexico", "BR": "Brazil", "TR": "Turkey", "ES": "Spain",
    "LA": "Laos", "MM": "Myanmar", "KH": "Cambodia", "BN": "Brunei",
    "CZ": "Czech Republic", "SK": "Slovakia", "MD": "Moldova", "TZ": "Tanzania",
    "CD": "Democratic Republic of the Congo",
}

# Statistical unit of each 8-digit line (``u`` number of items, ``kg`` kilograms), as declared
# in Thailand's tariff-line submissions to UN Comtrade for December 2024. 11-digit lines
# inherit their 8-digit line's unit; lines absent here never reported a quantity.
UNIT8_MAP = {
    "84714910": "kg", "84714990": "kg", "84715010": "kg", "84715090": "kg",
    "84718010": "kg", "84718070": "kg", "84718090": "kg",
    "84733010": "kg", "84733090": "kg",
    "85176210": "u", "85176230": "kg", "85176241": "kg", "85176242": "kg",
    "85176243": "kg", "85176249": "kg", "85176251": "u", "85176253": "u",
    "85176259": "kg", "85176269": "kg", "85176291": "u", "85176292": "u",
    "85176299": "u",
    "85235111": "u", "85235119": "u", "85235121": "u", "85235129": "u",
    "85235130": "u", "85235191": "u", "85235192": "u", "85235199": "u",
    "85423100": "u", "85423200": "u", "85423900": "u",
}

_AGGREGATE_PARTNERS = ("world", "total", "all countries")

# A clean row's identity, the key on which retrievals are merged.
_KEY = ["flow", "code_native", "partner", "period"]


def _num(raw) -> float:
    """A published numeric string (``".0000"``, ``"26071992.9276"``, ``""``) as a float."""
    s = str(raw or "").strip()
    if s in ("", ".", "-"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def has_trade(record) -> bool:
    """True if the record's monthly fields carry anything (the ``acc_*`` twins are ignored)."""
    return (_num(record.get("value_usd")) != 0 or _num(record.get("value_baht")) != 0
            or _num(record.get("quantity")) != 0)


def partner_name(record) -> str:
    """Canonical English partner name of a record."""
    cc = str(record.get("country_code", "")).strip().upper()
    if cc in PARTNER_MAP:
        return PARTNER_MAP[cc]
    name = str(record.get("country_name_en", "")).strip()
    return name.title() if name else cc


def qty_unit(code: str) -> str:
    """Statistical unit of a line, inherited from its 8-digit parent (see ``UNIT8_MAP``)."""
    return UNIT8_MAP.get(code[:8], "")


def read_raw(path: Path) -> list[dict]:
    """Clean rows of one raw file, before retrievals are merged."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    flow = blob["flow"]
    is_native = blob.get("pass", "") == "hs11"
    code_meta = blob.get("code_meta", {})
    pulled_date = iso_date(pulled_stamp(path))
    rows = []
    for q in blob.get("queries", []):
        code = str(q["code"])
        period = f"{int(q['year']):04d}-{int(q['month']):02d}"
        desc = code_meta.get(code) or BASKET_DESC.get(code, "")
        for rec in q.get("rows", []):
            if not has_trade(rec):
                continue
            partner = partner_name(rec)
            if partner.lower() in _AGGREGATE_PARTNERS:
                continue
            qty = _num(rec.get("quantity")) if is_native else 0.0
            rows.append({
                "reporter": REPORTER, "flow": flow, "partner": partner,
                "code_native": code, "code_hs6": code[:6], "code_desc": desc,
                "period": period,
                "value_usd": round(_num(rec.get("value_usd")), 2),
                "value_local": round(_num(rec.get("value_baht")), 2),
                "currency_local": "THB",
                "qty": qty if qty else "",
                "qty_unit": qty_unit(code) if qty else "",
                "weight_kg": "",
                "source": SOURCE, "pulled_date": pulled_date,
            })
    return rows


def standardize(files) -> pd.DataFrame:
    """All raw files -> the clean frame, keeping the newest retrieval of every
    flow x code x partner x month."""
    rows = [row for path in sorted(files) for row in read_raw(path)]
    if not rows:
        return pd.DataFrame(columns=SCHEMA_COLS)
    return finalize(keep_newest(pd.DataFrame(rows), _KEY))


def interior_gaps(df: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """Months missing inside each flow x HS-4/HS-6 series. Those codes trade every month, so a
    hole means a query came back empty; sparse 11-digit lines (used goods, waste) are not
    checked, as they legitimately skip months."""
    if df.empty:
        return {}
    all_periods = sorted(df["period"].unique())
    gaps = {}
    headline = df[df["code_native"].str.len() <= 6]
    for (flow, code), g in headline.groupby(["flow", "code_native"]):
        have = set(g["period"])
        missing = [p for p in all_periods if min(have) <= p <= max(have) and p not in have]
        if missing:
            gaps[(flow, code)] = missing
    return gaps


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """Build the Thailand clean book from every raw retrieval, newest retrieval winning.

    With ``vintage``, retrievals made after that date are ignored. ``fx_rates`` is unused:
    the publication carries US dollar values itself.
    """
    files = raw_files(raw_dir, RAW_PATTERN, vintage)
    df = standardize(files)
    for (flow, code), missing in interior_gaps(df).items():
        print(f"  warning: thailand {flow} {code} has no rows for {', '.join(missing)}")
    stamp = pulled_stamp(files[0])
    years = year_span(df["period"]) if not df.empty else stamp[:4]
    return [CleanBook(df, f"thailand_T1-T2_MX_{years}_pulled{stamp}.csv")]
