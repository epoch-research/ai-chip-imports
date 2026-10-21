"""European Union: Eurostat COMEXT, "EU trade since 1988 by HS2-4-6 and CN8" (book ``eu``).

Publication
    Eurostat's COMEXT monthly trade-by-product dataset (DS-045409). One dataset carries every
    member state as a reporter plus the EU27 aggregate. All 27 member states are kept
    individually (Eurostat ISO-2 codes; Greece is ``GR`` in this dataset, not the usual
    Eurostat ``EL``) and the ``EU27_2020`` aggregate is kept as its own reporter, ``eu27``, so
    member rows and the aggregate are never summed together. Filter on ``reporter`` before
    aggregating.

Codes
    Three native levels arrive as separate rows: CN8 (Combined Nomenclature, the EU's 8-digit
    detail), the HS-6 subheadings of the basket, and the 4-digit parents 8471/8473/8542. Filter
    on ``code_native`` length before summing. Parent rows have a blank ``code_hs6``. CN8 rows
    are described by Eurostat's own product label (its current-validity segment); HS-6 and
    parent rows carry the short basket labels.

Flows
    COMEXT flow 1 (import) -> M, 2 (export) -> X. COMEXT has no re-export / domestic-export
    split, so RX and DX never occur. Intra-EU arrivals and dispatches are included, so a
    member's imports include goods re-dispatched from the EU port of entry (the quasi-transit
    effect).

Partners
    Country-level partners only. World, EU / euro-area blocs (intra/extra totals) and the
    "not specified" / "stores and provisions" special codes are dropped. Names are Eurostat's
    English labels without parenthetical qualifiers, with a few overrides for the spellings
    used across the other books (China, Taiwan, Hong Kong, South Korea, ...).

Values
    ``VALUE_IN_EUROS`` is in plain euros -> ``value_local`` (EUR). ``value_usd`` divides it by
    the monthly period-average EUR-per-USD rate from the exchange-rate table; months without a
    rate leave ``value_usd`` blank. Imports are valued CIF at the reporter's border, exports
    FOB (Eurostat's statistical value). Rows are kept only where the euro value is non-zero.

Quantity and weight
    ``QUANTITY_IN_100KG`` is in hundreds of kilograms and is multiplied by 100 into
    ``weight_kg``. ``SUPPLEMENTARY_QUANTITY`` is the CN supplementary unit, number of items
    (``p/st``) for every code in the basket -> ``qty`` / ``qty_unit``. Zero measures are left
    blank.

Raw files
    ``eu_comext_T1-T2_all_<YYYY>-<YYYY>_pulled<YYYYMMDD>.jsonl.gz``: gzip-compressed JSON Lines,
    one Eurostat JSON-stat 2.0 dataset response per line, stored exactly as returned (no
    envelope). Each response covers one reporter x flow over a block of months, all partners,
    a set of products and all three measures. Where two responses carry the same cell, the
    later line wins.

Vintages
    Every retrieval re-reads the whole series, so ``build`` reads the newest file alone;
    ``vintage="YYYYMMDD"`` pins the file stamped that day.

Output
    ``eu_T1-T2_MX_<first year>-<last year>_pulled<YYYYMMDD>.csv``. ``pulled_date`` is the raw
    file's ``pulled<YYYYMMDD>`` stamp.
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
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
    year_span,
)

BOOK = "eu"
RAW_PATTERN = "eu_comext_T1-T2_all_*_pulled*.jsonl.gz"

SOURCE = "eurostat_comext"
CURRENCY = "EUR"

FLOW_MAP = {"1": "M", "2": "X"}

# The supplementary unit of every basket code is p/st (number of items). The data response does
# not label it per code, so it is recorded explicitly wherever a quantity is present.
QTY_UNIT = "p/st"

REPORTERS = {
    "AT": "austria", "BE": "belgium", "BG": "bulgaria", "CY": "cyprus",
    "CZ": "czech republic", "DE": "germany", "DK": "denmark", "EE": "estonia",
    "ES": "spain", "FI": "finland", "FR": "france", "GR": "greece",
    "HR": "croatia", "HU": "hungary", "IE": "ireland", "IT": "italy",
    "LT": "lithuania", "LU": "luxembourg", "LV": "latvia", "MT": "malta",
    "NL": "netherlands", "PL": "poland", "PT": "portugal", "RO": "romania",
    "SE": "sweden", "SI": "slovenia", "SK": "slovakia",
    "EU27_2020": "eu27",
}

# Partner code -> the English name the other books use. Partners not listed keep Eurostat's own
# label with any parenthetical qualifier removed.
PARTNER_MAP = {
    "CN": "China", "TW": "Taiwan", "HK": "Hong Kong", "MO": "Macau",
    "KR": "South Korea", "KP": "North Korea", "US": "United States",
    "JP": "Japan", "SG": "Singapore", "MY": "Malaysia", "VN": "Vietnam",
    "RU": "Russia", "GB": "United Kingdom", "AE": "United Arab Emirates",
    "CH": "Switzerland", "IR": "Iran", "LA": "Laos", "TR": "Turkey",
    "MK": "North Macedonia", "SY": "Syria", "BN": "Brunei", "CZ": "Czech Republic",
    "MD": "Moldova", "BO": "Bolivia", "VE": "Venezuela", "TZ": "Tanzania",
}

# Aggregate and non-country partner codes: world, the EU / euro-area blocs and their intra/extra
# totals, and the "Q*" specials (not specified, stores and provisions). The label keywords catch
# any aggregate not listed. Qatar is ``QA``, a real country, and is kept.
PARTNER_DROP = {
    "WORLD", "EU", "EU27_2020", "EU28", "EA", "EA19", "EA20", "EA21",
    "EXT_EU", "EXT_EU27_2020", "EXT_EA", "EXT_EA21",
    "INT_EU", "INT_EU27_2020", "INT_EA", "INT_EA21",
    "QU", "QV", "QW", "QR", "QS", "QP", "QQ", "QX", "QY", "QZ", "QN", "QM",
}
_AGG_KEYWORDS = ("not specified", "stores and provisions", "all countries",
                 "not determined", "european union", "euro area")

_SORT_KEYS = ["reporter", "flow", "code_native", "period", "partner"]


# --------------------------------------------------------------------------------------------
# JSON-stat decoding
# --------------------------------------------------------------------------------------------
def _inverse_index(category: dict) -> dict[int, str]:
    """``category.index`` as ``{position: code}``; JSON-stat allows both a dict and a list."""
    idx = category["index"]
    if isinstance(idx, dict):
        return {pos: code for code, pos in idx.items()}
    return {i: code for i, code in enumerate(idx)}


def decode(resp: dict) -> Iterator[tuple]:
    """Observations of one JSON-stat response as
    ``(reporter, flow, partner, product, period, indicator, value)`` tuples.

    The ``value`` object is sparse and keyed by the flattened row-major position over ``size``
    (last dimension fastest). Null and zero values are skipped.
    """
    ids = resp["id"]
    size = resp["size"]
    dim = resp["dimension"]
    inv = {k: _inverse_index(dim[k]["category"]) for k in ids}
    strides = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        strides[i] = strides[i + 1] * size[i + 1]
    pos = {k: ids.index(k) for k in ids}
    for flat_s, val in resp["value"].items():
        if val in (None, 0):
            continue
        flat = int(flat_s)
        coord = {k: inv[k][(flat // strides[pos[k]]) % size[pos[k]]] for k in ids}
        yield (coord["reporter"], coord["flow"], coord["partner"], coord["product"],
               coord["time"], coord["indicators"], val)


def _labels(resp: dict, dim_id: str) -> dict[str, str]:
    """``{code: label}`` of one dimension of a JSON-stat response."""
    return dict(resp["dimension"][dim_id]["category"].get("label", {}))


def aggregate(responses: Iterable[dict]) -> tuple[dict, dict, dict]:
    """Fold responses into ``{(reporter, flow, partner, product, period): {indicator: value}}``.

    Also returns the partner and product labels seen across all responses (later responses
    win). Where two responses carry the same cell, the later one's value is kept.
    """
    cells: dict[tuple, dict] = {}
    partner_labels: dict[str, str] = {}
    product_labels: dict[str, str] = {}
    for resp in responses:
        partner_labels.update(_labels(resp, "partner"))
        product_labels.update(_labels(resp, "product"))
        for rep, flow, partner, product, period, indic, val in decode(resp):
            cells.setdefault((rep, flow, partner, product, period), {})[indic] = val
    return cells, partner_labels, product_labels


# --------------------------------------------------------------------------------------------
# Standardize
# --------------------------------------------------------------------------------------------
def _clean_label(label: str) -> str:
    """Drop a ``(...)`` qualifier and trailing punctuation from a Eurostat label."""
    s = str(label).split("(")[0].strip()
    return s.rstrip(",; ").strip()


def _keep_partner(code: str, label: str) -> bool:
    """False for world, bloc and special partner codes; True for countries."""
    if code in PARTNER_DROP:
        return False
    low = str(label).lower()
    return not any(k in low for k in _AGG_KEYWORDS)


def _partner_name(code: str, label: str) -> str:
    """The cross-book English name of a partner."""
    return PARTNER_MAP.get(code, _clean_label(label))


def _code_desc(product: str, product_labels: dict[str, str]) -> str:
    """Basket label for HS-6 / parent rows; for CN8, Eurostat's label.

    A CN8 label concatenates every historical wording with ``;``, the currently valid one
    first, so only the first segment is kept (capped at 80 characters).
    """
    if product in BASKET_DESC:
        return BASKET_DESC[product]
    seg = str(product_labels.get(product, "")).split(";")[0].strip()
    return seg[:80].rstrip(" ,;") if len(seg) > 80 else seg


def standardize(responses: Iterable[dict], pulled_date: str,
                fx_rates: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Build the clean frame from an iterable of JSON-stat responses."""
    cells, partner_labels, product_labels = aggregate(responses)
    cols: dict[str, list] = {c: [] for c in SCHEMA_COLS if c != "value_usd"}
    for (rep, flow, partner, product, period), m in cells.items():
        value = m.get("VALUE_IN_EUROS")
        if value in (None, 0):
            continue
        plabel = partner_labels.get(partner, partner)
        if not _keep_partner(partner, plabel):
            continue
        supp = m.get("SUPPLEMENTARY_QUANTITY")
        wt100 = m.get("QUANTITY_IN_100KG")
        has_qty = supp not in (None, 0)
        cols["reporter"].append(REPORTERS[rep])
        cols["flow"].append(FLOW_MAP[flow])
        cols["partner"].append(_partner_name(partner, plabel))
        cols["code_native"].append(product)
        cols["code_hs6"].append(product[:6] if len(product) >= 6 else "")
        cols["code_desc"].append(_code_desc(product, product_labels))
        cols["period"].append(period)
        cols["value_local"].append(int(round(value)))
        cols["currency_local"].append(CURRENCY)
        cols["qty"].append(int(round(supp)) if has_qty else "")
        cols["qty_unit"].append(QTY_UNIT if has_qty else "")
        cols["weight_kg"].append(int(round(wt100 * 100)) if wt100 not in (None, 0) else "")
        cols["source"].append(SOURCE)
        cols["pulled_date"].append(pulled_date)
    del cells
    df = pd.DataFrame(cols)
    df["value_usd"] = to_usd(df["value_local"], df["period"], CURRENCY, fx_rates)
    return finalize(df, order=_SORT_KEYS)


def iter_raw(path: Path) -> Iterator[dict]:
    """Each JSON-stat response stored in a gzip JSON Lines raw file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def build(raw_dir: Path, fx_rates: dict[tuple[str, str], float],
          vintage: str | None = None) -> list[CleanBook]:
    """Clean EU book from the newest raw file in ``raw_dir``, or the one stamped ``vintage``."""
    path = select_vintage(raw_dir, RAW_PATTERN, vintage)[-1]
    stamp = pulled_stamp(path)
    df = standardize(iter_raw(path), iso_date(stamp), fx_rates)
    if df.empty:
        raise ValueError(f"no rows with a euro value in {path}")
    blank = int((df["value_usd"] == "").sum())
    if blank:
        print(f"  [eu] value_usd blank on {blank:,} rows (no exchange rate for the month)")
    return [CleanBook(df, f"eu_T1-T2_MX_{year_span(df['period'])}_pulled{stamp}.csv")]
