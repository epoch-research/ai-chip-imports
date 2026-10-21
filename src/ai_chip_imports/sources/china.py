"""China: General Administration of Customs (GACC) customs statistics (book ``china``).

Publication
    General Administration of Customs of the People's Republic of China, monthly customs
    statistics by commodity and trading partner, exported as CSV from the publisher's
    customs-statistics query portal (English interface), in US dollars with one row per
    month.

Codes
    HS-8, China's national tariff nomenclature (a single scheme for both flows).
    ``code_native`` is the 8-digit code; ``code_hs6`` its first six digits. The basket files
    list every HS-8 line under each HS-6 heading, so grouping by ``code_hs6`` gives exact
    HS-6 totals. ``code_desc`` is the publisher's (truncated) English description.

Flows
    Imports -> M, exports -> X. The files carry no flow column: the flow is read from the
    file name. There is no re-export split.

Partners
    English names as published, with the partners shared with other books canonicalized
    (``Republic of Korea`` -> South Korea, ``Viet Nam`` -> Vietnam) and the ", China" suffix
    the publisher appends to Taiwan, Hong Kong and Macao dropped. There is no world total.

Values
    ``US dollar`` is in plain dollars (not thousands), written with thousands separators,
    and goes straight to ``value_usd``; no exchange rate is involved, so ``fx_rates`` is
    unused. Imports are valued CIF, exports FOB.

Quantity and weight
    Each row carries two (quantity, unit) slots, ``Quantity``/``Unit`` and
    ``Supplementary Quantity``/``Supplementary Unit``; whichever is in kilograms becomes
    ``weight_kg`` and the other (usually "Number of item") becomes ``qty``/``qty_unit``.

Raw files
    ``china_gacc_T1-T2_<imports|exports>_hs8_<YYYYMM>-<YYYYMM>_pulled<YYYYMMDD>.csv``: the
    portal's CSV downloads saved as downloaded, GB18030-encoded (partner names contain a
    full-width comma inside quoted fields), columns addressed by header because their order
    varies between downloads. A book is assembled from many month-range files. Files whose
    name contains ``cumulative`` or ``ytd`` hold range aggregates rather than monthly rows and
    are skipped.

Vintages
    ``build`` merges every monthly file: where the same flow x code x partner x month (the
    merge key) appears in more than one file, the newest retrieval wins (same-day files: the
    later file name). ``vintage="YYYYMMDD"`` leaves out the files retrieved after that date.

Output
    ``china_T1-T2_CN_<first year>-<last year>_pulled<stamp>.csv``, ``<stamp>`` being the
    newest retrieval date in the book. ``pulled_date`` on each row is the retrieval stamp of
    the file the row came from.
"""
from __future__ import annotations

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
)

BOOK = "china"
REPORTER = "china"
SOURCE = "gacc"
RAW_ENCODING = "gb18030"
RAW_PATTERN = "china_gacc_T1-T2_*_hs8_*.csv"

# Only the partners that must line up with other books are renamed; other captions are
# kept as published.
PARTNER_MAP = {
    "Republic of Korea": "South Korea",
    "Viet Nam": "Vietnam",
    "Taiwan，China": "Taiwan",
    "Hong Kong，China": "Hong Kong",
    "Macao,China": "Macao",
    "Macao，China": "Macao",
    "Russian Federation": "Russia",
}

# Unit captions that mark the kilogram slot (English and Chinese interfaces).
KG_TOKENS = ("kilogram", "千克", "公斤")

# A clean row's identity, the key on which retrievals are merged.
_KEY = ["flow", "code_native", "partner", "period"]


def flow_from_name(path: Path) -> str:
    """M or X from the ``_imports_``/``_exports_`` token in a raw file name."""
    low = Path(path).name.lower()
    if "_imports_" in low or "_import_" in low:
        return "M"
    if "_exports_" in low or "_export_" in low:
        return "X"
    raise ValueError(f"cannot tell the flow (imports/exports) from the file name: {path}")


def _num(raw) -> int | None:
    """``"1,234"`` -> 1234; blank or ``"-"`` -> None. GACC figures are whole numbers."""
    digits = re.sub(r"[^\d]", "", str(raw or ""))
    return int(digits) if digits else None


def normalize_partner(caption) -> str:
    """Published partner caption -> the name used across books."""
    return PARTNER_MAP.get((caption or "").strip(), (caption or "").strip())


def _units(rec: dict) -> tuple[int | None, str | None, int | None]:
    """(qty, qty_unit, weight_kg) from the two unit slots, whichever slot holds kilograms."""
    qty = qty_unit = weight_kg = None
    slots = [(rec.get("Unit"), rec.get("Quantity")),
             (rec.get("Supplementary Unit"), rec.get("Supplementary Quantity"))]
    for unit, val in slots:
        u = (unit or "").strip()
        if any(t in u.lower() for t in KG_TOKENS):
            weight_kg = _num(val)
        elif u:
            v = _num(val)
            if v is not None and qty is None:
                qty, qty_unit = v, u
    return qty, qty_unit, weight_kg


def read_raw_file(path: Path) -> list[dict]:
    """Clean-schema rows of one GACC download."""
    flow = flow_from_name(path)
    pulled = iso_date(pulled_stamp(path))
    raw = pd.read_csv(path, encoding=RAW_ENCODING, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    rows = []
    for rec in raw.to_dict("records"):
        code = (rec.get("Commodity code") or "").strip()
        period_raw = (rec.get("Date of data") or "").strip()
        if not (code.isdigit() and len(period_raw) == 6 and period_raw.isdigit()):
            continue  # blank or malformed line
        qty, qty_unit, weight_kg = _units(rec)
        value = _num(rec.get("US dollar"))
        if value is None and qty is None and weight_kg is None:
            continue
        rows.append({
            "reporter": REPORTER,
            "flow": flow,
            "partner": normalize_partner(rec.get("Trading partner")),
            "code_native": code,
            "code_hs6": code[:6],
            "code_desc": (rec.get("Commodity") or "").strip(),
            "period": f"{period_raw[:4]}-{period_raw[4:6]}",
            "value_usd": "" if value is None else value,
            "value_local": "",
            "currency_local": "",
            "qty": "" if qty is None else qty,
            "qty_unit": qty_unit or "",
            "weight_kg": "" if weight_kg is None else weight_kg,
            "source": SOURCE,
            "pulled_date": pulled,
        })
    return rows


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """Build the China book from every monthly GACC download in ``raw_dir`` (all retrievals
    merged).

    With ``vintage``, files retrieved after that date are ignored. ``fx_rates`` is unused:
    values are published in USD.
    """
    # A download not broken down by month holds one aggregate for its whole range, labelled
    # with the end month; such files are marked in their name and must never be mixed with
    # monthly rows.
    files = sorted(f for f in raw_files(raw_dir, RAW_PATTERN, vintage)
                   if not re.search(r"cumulative|ytd", f.name, re.I))
    if not files:
        raise FileNotFoundError(f"no monthly raw files matching {RAW_PATTERN} in {raw_dir}")

    rows = []
    for f in files:
        rows.extend(read_raw_file(f))
    if not rows:
        raise ValueError(f"no usable rows in the GACC files under {raw_dir}")
    df = finalize(keep_newest(pd.DataFrame(rows, columns=SCHEMA_COLS), _KEY))

    periods = sorted(df["period"].unique())
    stamp = df["pulled_date"].max().replace("-", "")
    name = f"china_T1-T2_CN_{periods[0][:4]}-{periods[-1][:4]}_pulled{stamp}.csv"
    return [CleanBook(df, name)]
