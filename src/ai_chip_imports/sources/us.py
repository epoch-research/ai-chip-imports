"""United States: US Census Bureau merchandise trade by commodity and partner (book ``us``).

Publication
    US Census Bureau international trade statistics, the monthly series behind the FT-900
    report "U.S. International Trade in Goods and Services" and USA Trade Online: imports by
    HTSUS commodity and exports by Schedule B commodity, by partner country, monthly from
    2022-01, not seasonally adjusted.

Codes
    The nine basket HS-6 codes and the 4-digit parents 8471, 8473 and 8542, each at two
    granularities in the same file:

    * the HS-6 backbone: ``code_native`` is the 6- or 4-digit code, ``code_desc`` a short
      curated label. On the 4-digit parent rows ``code_hs6`` repeats the 4-digit code.
    * the native 10-digit lines: HTSUS for imports, Schedule B for exports. The two schedules
      subdivide the same HS-6 differently and both renumber their 10-digit lines (on 1 January,
      sometimes 1 July), so each row carries the official description published for that
      month.

    The granularities overlap (a 10-digit line is part of its HS-6 row, an HS-6 row part of
    its 4-digit parent): never add rows across granularities.

Flows
    Imports -> ``M``, at general-imports value (all goods entering the US, including into
    bonded warehouses and foreign-trade zones). Exports are published split by the Census
    domestic/foreign indicator: domestic exports (US-made goods) -> ``DX``, foreign exports
    (re-exports of imported goods in the condition they entered) -> ``RX``. ``X`` is built here
    as DX + RX per period x code x partner, so X == DX + RX exactly.

Partners
    Census country names as published, upper case (e.g. ``KOREA, SOUTH``). The publication
    lists aggregate rows (the world total, trade groupings such as OECD or APEC, continent
    rollups) next to the countries; they are dropped (see :func:`is_country`).

Values
    US dollars, plain units, as published; no exchange rate is applied (``fx_rates`` is unused)
    and ``value_local``/``currency_local`` stay blank. Imports are valued at customs value, the
    price paid excluding international freight, insurance and US duties (roughly FOB at the
    foreign port), unlike the CIF import values of most other books; exports at f.a.s. value
    (free alongside ship at the US port of export).

Quantity and weight
    ``qty``/``qty_unit`` are the first statistical quantity and its unit (``NO`` = number of
    units on the basket's lines). The publication carries quantity on the 10-digit lines only:
    the HS-6 aggregates report unit ``-`` and quantity 0, written blank here. ``weight_kg`` is
    air plus vessel shipping weight in kg: GROSS weight (goods plus packing) and only for
    shipments by air or sea, so it is not comparable with the net weights of other books.
    Zero quantities and weights are written blank.

Raw files
    ``us_T1-T2_MXRXDX_<first>-<last>_pulled<YYYYMMDD>.pkl``: a Python pickle of a list of dicts,
    one per published row, holding the publication's own fields as strings
    (``CTY_CODE``, ``CTY_NAME``, the month's value ``GEN_VAL_MO`` (imports) or ``ALL_VAL_MO``
    (exports), ``AIR_WGT_MO``, ``VES_WGT_MO``, the quantity fields ``GEN_QY1_MO``/``QTY_1_MO``
    and ``UNIT_QY1``, the domestic/foreign indicator ``DF`` on exports, the commodity
    ``I_COMMODITY``/``E_COMMODITY`` and, on 10-digit rows, its description ``..._LDESC``), plus
    five annotations made when the rows were saved: ``_flow`` (``M`` or ``X``), ``_level``
    (``hs6`` or ``hs10``), ``_code`` (the commodity the row was published under) and
    ``_year``/``_month`` (ints). The same list saved as JSON (``.pkl`` -> ``.json``) is read
    too; prefer it for files from elsewhere, since loading a pickle can execute code.
    Archives without the domestic/foreign split (``us_T1-T2_XM_*``) are not read.

Vintages
    Every retrieval re-reads the whole series, so ``build`` reads the newest archive alone;
    ``vintage="YYYYMMDD"`` pins the archive stamped that day.

Output
    ``us_T1-T2_MXRXDX_<first>-<last>_pulled<YYYYMMDD>.csv``, the raw archive's own name.
    ``pulled_date`` is the archive's ``pulled<YYYYMMDD>`` stamp.

Import-charges sidecar (``data/external/us_cif/``)
    The same publication also feeds the FOB model outside the clean book: customs value
    versus CIF value of the basket's imports. :func:`build_cif_charges` builds it; write the
    result into the same directory, e.g. ``write_clean(build_cif_charges(), CIF_DIR)``.

    Raw ``us_cif_charges_raw_pulled<YYYYMMDD>.json``: a JSON list of published rows for general
    imports at HS-6, monthly, by partner, for the nine basket codes plus 847141 (desktop
    computers, a low value-density anchor): ``CTY_CODE``, ``CTY_NAME``, ``time``
    (``YYYY-MM``), and ``GEN_VAL_MO`` customs value, ``GEN_CIF_MO`` CIF value, ``GEN_CHA_MO``
    import charges (freight, insurance and other costs of bringing the goods from the foreign
    port to the first US port, excluding duties), the same value/charges/shipping-weight
    triple for air (``AIR_*``), vessel (``VES_*``) and containerized vessel (``CNT_*``, a
    subset of vessel), plus the annotation ``_code`` (the HS-6 code). General minus air minus
    vessel is land and other modes. The newest file is read (or the one stamped ``vintage``).

    Clean ``us_cif_charges_M_<first>-<last>_pulled<YYYYMMDD>.csv``: ``partner, cty_code,
    code_hs6, period, gen_val, gen_cif, gen_cha, air_val, air_cha, air_kg, ves_val, ves_cha,
    ves_kg, cnt_val, cnt_cha, cnt_kg, pulled_date`` (USD and kg), country rows only.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from .. import paths
from ._common import (
    BASKET_DESC,
    SCHEMA_COLS,
    CleanBook,
    finalize,
    iso_date,
    pulled_stamp,
    select_vintage,
    year_span,
)

BOOK = "us"
SOURCE = "census_api"

RAW_PATTERNS = ("us_T1-T2_MXRXDX_*_pulled*.pkl", "us_T1-T2_MXRXDX_*_pulled*.json")

CIF_DIR = paths.EXTERNAL_DIR / "us_cif"
CIF_PATTERN = "us_cif_charges_raw_pulled*.json"

# Field names differ by flow: the commodity, value and first-quantity fields of the import
# (HTSUS) and export (Schedule B) tables. GEN_ marks general imports, an import-only concept.
FLOWS = {
    "X": {"val": "ALL_VAL_MO", "commodity": "E_COMMODITY", "qty": "QTY_1_MO"},
    "M": {"val": "GEN_VAL_MO", "commodity": "I_COMMODITY", "qty": "GEN_QY1_MO"},
}

# Census domestic/foreign indicator on exports -> schema flow.
DF_FLOW = {"1": "DX", "2": "RX"}

# Charges sidecar: published field -> column.
CIF_FIELDS = {
    "GEN_VAL_MO": "gen_val", "GEN_CIF_MO": "gen_cif", "GEN_CHA_MO": "gen_cha",
    "AIR_VAL_MO": "air_val", "AIR_CHA_MO": "air_cha", "AIR_WGT_MO": "air_kg",
    "VES_VAL_MO": "ves_val", "VES_CHA_MO": "ves_cha", "VES_WGT_MO": "ves_kg",
    "CNT_VAL_MO": "cnt_val", "CNT_CHA_MO": "cnt_cha", "CNT_WGT_MO": "cnt_kg",
}


def is_country(cty_code) -> bool:
    """True for a single partner country, False for an aggregate row.

    Census lists the world total (code ``-``), trade groupings (OECD ``0020``, APEC ``0026``,
    EU ``0003``, ...) and continent rollups (``1XXX`` North America, ``5XXX`` Asia, ...) in the
    same table as the countries; summing them all overstates trade about sixfold. Countries are
    exactly the 4-digit numeric codes 1000-7999.
    """
    return (isinstance(cty_code, str) and len(cty_code) == 4 and cty_code.isdigit()
            and "1000" <= cty_code <= "7999")


def _num(rec: dict, field: str) -> float:
    """A published figure as float; missing, blank or non-numeric reads as 0."""
    try:
        return float(rec.get(field, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def load_records(path: Path) -> list[dict]:
    """The archived rows of a raw file, from its pickle or JSON form."""
    path = Path(path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def standardize(records: list[dict], pulled_date: str) -> pd.DataFrame:
    """Map archived Census rows to the clean schema; see the module docstring for the rules."""
    rows = []
    for rec in records:
        flow = rec["_flow"]
        spec = FLOWS[flow]
        if not is_country(rec.get("CTY_CODE", "")):
            continue

        level = rec["_level"]
        code_native = rec.get(spec["commodity"]) if level == "hs10" else rec["_code"]
        if not code_native:
            continue
        code_hs6 = code_native[:6]
        if level == "hs10":
            code_desc = ((rec.get(f"{spec['commodity']}_LDESC") or "").strip()
                         or BASKET_DESC.get(code_hs6, ""))
        else:
            code_desc = BASKET_DESC.get(code_hs6, "")

        value = _num(rec, spec["val"])
        weight = _num(rec, "AIR_WGT_MO") + _num(rec, "VES_WGT_MO")
        if value == 0 and weight == 0:
            continue

        if flow == "M":
            out_flow = "M"
        else:
            # Exports are published as domestic (1), foreign (2) and their total (-). Only the
            # two components are kept; X is rebuilt from them below. Anything else is skipped
            # rather than mislabelled.
            out_flow = DF_FLOW.get(str(rec.get("DF", "")).strip())
            if out_flow is None:
                continue

        rows.append({
            "reporter": "us",
            "flow": out_flow,
            "partner": rec.get("CTY_NAME", ""),
            "code_native": code_native,
            "code_hs6": code_hs6,
            "code_desc": code_desc,
            "period": f'{rec["_year"]}-{rec["_month"]:02d}',
            "value_usd": value,
            "value_local": "",
            "currency_local": "",
            # Kept numeric until X is built so the DX + RX sum can add them; zeros are blanked
            # at the end. The HS-6 aggregates publish unit "-", which becomes "".
            "qty": _num(rec, spec["qty"]),
            "qty_unit": (rec.get("UNIT_QY1") or "").strip().strip("-"),
            "weight_kg": weight,
            "source": SOURCE,
            "pulled_date": pulled_date,
        })

    df = pd.DataFrame(rows, columns=SCHEMA_COLS)
    if df.empty:
        return df

    # X = DX + RX per period x code x partner. Value, weight and quantity are summed;
    # qty_unit stays a key so quantities in different units are never added together.
    exp = df[df["flow"].isin(["DX", "RX"])]
    if not exp.empty:
        keys = ["reporter", "partner", "code_native", "code_hs6", "code_desc", "period",
                "value_local", "currency_local", "qty_unit", "source", "pulled_date"]
        xagg = exp.groupby(keys, as_index=False).agg(value_usd=("value_usd", "sum"),
                                                     weight_kg=("weight_kg", "sum"),
                                                     qty=("qty", "sum"))
        xagg["flow"] = "X"
        df = pd.concat([df, xagg[SCHEMA_COLS]], ignore_index=True)

    df["weight_kg"] = df["weight_kg"].apply(lambda w: w if w else "")
    df["qty"] = df["qty"].apply(lambda q: q if q else "")
    return finalize(df)


def build(raw_dir: Path, fx_rates: dict, vintage: str | None = None) -> list[CleanBook]:
    """The US clean book from the newest raw archive, or the one stamped ``vintage``.

    ``fx_rates`` is unused: the publication is in US dollars.
    """
    raw = select_vintage(raw_dir, RAW_PATTERNS, vintage)[-1]
    frame = standardize(load_records(raw), iso_date(pulled_stamp(raw)))
    print(f"  us: {len(frame):,} rows from {raw.name}")
    return [CleanBook(frame, f"{raw.stem}.csv")]


def build_cif_charges(raw_dir: Path = CIF_DIR, vintage: str | None = None) -> CleanBook:
    """The import-charges sidecar from the newest raw file, or the one stamped ``vintage``.

    Keeps country rows only and checks the published identity CIF value = customs value +
    charges (to within the dollar rounding of the published figures).
    """
    raw = select_vintage(raw_dir, CIF_PATTERN, vintage)[-1]
    with open(raw, encoding="utf-8") as fh:
        df = pd.DataFrame(json.load(fh))
    for field in CIF_FIELDS:
        df[field] = pd.to_numeric(df[field], errors="coerce").fillna(0.0)
    df = df[df["CTY_CODE"].map(is_country)].copy()

    out = pd.DataFrame({"partner": df["CTY_NAME"].astype(str),
                        "cty_code": df["CTY_CODE"].astype(str),
                        "code_hs6": df["_code"],
                        "period": df["time"].astype(str)})
    for field, col in CIF_FIELDS.items():
        out[col] = df[field].values
    out["pulled_date"] = iso_date(pulled_stamp(raw))
    out = out.sort_values(["code_hs6", "period", "partner"]).reset_index(drop=True)

    gap = (out["gen_cif"] - out["gen_val"] - out["gen_cha"]).abs()
    if (gap > 2).any():
        raise ValueError(f"{raw.name}: CIF value != customs value + charges on "
                         f"{int((gap > 2).sum())} rows")
    return CleanBook(out, f"us_cif_charges_M_{year_span(out['period'])}"
                          f"_pulled{pulled_stamp(raw)}.csv")

