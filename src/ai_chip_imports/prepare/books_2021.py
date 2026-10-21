"""Build the three prepared 2021 books from raw customs files.

The 2021 non-AI reference reads three prepared books. Each is in the compact trade-frame
schema: one row per reporter x flow x partner x HS-6 code x month, values in US dollars on
the FOB basis, with canonical reporter and partner names (``rep_c``/``par_c``).

    comtrade_prepared.csv.gz    UN Comtrade monthly submissions: 15 national reporters
                                and the 27 EU member states
    eu_native_prepared.csv.gz   Eurostat COMEXT, the 27 member states
    uk_native_prepared.csv.gz   HMRC Overseas Trade in Goods Statistics

The measured history combines the Comtrade book's reporters other than the EU members and
the United Kingdom (``MEASURED_COMTRADE_REPORTERS``) with the two native books; the EU-member
and UK rows of the Comtrade book are not read downstream.

Comtrade book
    1. Keep each cell's total record (motCode 0, partner2Code 0, customsCode C00; Comtrade
       also publishes breakdowns of the same trade by transport mode, second partner and
       customs procedure), with a country partner (partnerCode != 0), for the four codes.
    2. Physical fields: ``weight_kg`` is netWgt unless flagged estimated; ``qty`` is the
       quantity when its unit is pieces (``u``) and it is not flagged estimated; otherwise 0.
       ``value_qtycov`` is the value of rows that carry a quantity.
    3. Austria filed no monthly 2021 records. Its annual records are spread evenly over the
       twelve months (value, quantity and weight each divided by 12): annual amounts are
       preserved, seasonality is not observed.
    4. Re-export split. Hong Kong, Singapore and Malaysia filed total exports only. Each
       total-export row is split into RX and DX with the code's 2022 re-export share
       RX / (RX + DX) from the national books (``RX_SHARE_2022``); the X rows are replaced.
       This is an assumption about 2021 composition, not an observation; RX + DX equals the
       filed total. The United States and Canada file RX and DX and are kept as filed. Other
       reporters keep every flow they filed.
    5. FOB conversion (below).

Native books
    1. Decode the publisher's records (JSON-stat for Eurostat; grouped OTS records for HMRC).
       The Eurostat responses cover the four HS-6 codes; HMRC reports CN8 lines, summed to
       HS-6. Values convert to US dollars at the 2021 monthly period-average rate
       (``value_local / units_per_usd``, unrounded).
    2. Sum to reporter x flow x partner x HS-6 x month.
    3. FOB conversion (below).

FOB conversion
    Import rows of CIF-reporting books (flow M, reporter not in ``FOB_EXEMPT``, value > 0) are
    divided by ``1 + tau``, the freight margin of ``fob.FobModel`` fitted on the US charges file
    ``FOB_SIDECAR``. 2021 precedes that file, so every 2021 month takes the level of its
    earliest complete quarter. A row's density is its own
    value per net kilogram where it carries weight; otherwise the code's pooled density
    ``POOLED_DENSITY`` stands in.

Fixed inputs
    ``RX_SHARE_2022`` and ``POOLED_DENSITY`` are statistics of the 2022+ national books as
    they stood when the reference was prepared. ``rx_shares_2022`` and ``pooled_densities``
    recompute them from a compact frame; the pinned values keep the 2021 books independent of
    later revisions to the national books. ``FOB_EXEMPT`` is the set of import books treated as
    FOB-type at preparation; it includes Australia, whose import rows the history loader
    deflates on read (``history.PACK_FOB_EXEMPT`` must equal this set).

Raw layout (``RAW_DIR``)
    comtrade/<reporter>_2021.json      Comtrade ``/data`` responses for the four codes, flows
                                       M, X, RX, DX: ``{"data": [record, ...], ...}``
                                       (reporter names with spaces use underscores)
    comtrade/austria_2021_annual.json  Austria's annual 2021 records, same envelope
    eurostat/<CC>_<flow>_2021.json     one JSON-stat response per member x flow (1 import,
                                       2 export), all partners, the four codes, 2021 months
    hmrc/UK_2021.json                  OTS records grouped by month x flow type x CN8 x
                                       partner: MonthId, FlowTypeId, Commodity.Cn8Code,
                                       Country.CountryName, V, NM, SU
    fx_2021.csv                        currency_local, period, units_per_usd (EUR and GBP)

Canonical names come from ``books.canon``.

    python -m ai_chip_imports.prepare.books_2021 [--raw-dir DIR] [--out-dir DIR] [--check DIR]
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .. import fob, paths
from ..sources.eu import decode as decode_jsonstat

CODES = ("847149", "847150", "847180", "847330")
BASELINE_DIR = paths.EXTERNAL_DIR / "baseline_2021"
RAW_DIR = BASELINE_DIR / "raw"
FX_2021_CSV = BASELINE_DIR / "fx_2021.csv"
FOB_SIDECAR = paths.EXTERNAL_DIR / "us_cif" / "us_cif_charges_M_2022-2026_pulled20260911.csv"

# Comtrade reporters, in the order their rows appear in the book.
COMTRADE_REPORTERS = (
    "mexico", "us", "china", "hong kong", "singapore", "malaysia", "korea", "uk", "japan",
    "indonesia", "india", "thailand", "australia", "canada", "saudi arabia",
    "austria", "belgium", "bulgaria", "croatia", "cyprus", "czech republic", "denmark",
    "estonia", "finland", "france", "germany", "greece", "hungary", "ireland", "italy",
    "latvia", "lithuania", "luxembourg", "malta", "netherlands", "poland", "portugal",
    "romania", "slovakia", "slovenia", "spain", "sweden",
)
# The Comtrade reporters the measured history reads (EU members and the UK come from native books).
MEASURED_COMTRADE_REPORTERS = tuple(r for r in COMTRADE_REPORTERS[:15] if r != "uk")

# Books filing total exports only, split with the 2022 code-level re-export share.
HUB_SPLIT = ("Hong Kong", "Singapore", "Malaysia")
# Books whose netting reads RX and DX (the hubs above plus the two that file the split).
SPLIT_BOOKS = HUB_SPLIT + ("United States", "Canada")

# RX / (RX + DX) by code, 2022, from the Hong Kong, Singapore and Malaysia national books.
RX_SHARE_2022 = {
    "Hong Kong": {"847149": 1.0, "847150": 0.9990521056361756,
                  "847180": 0.9999896048430407, "847330": 0.9999543183559852},
    "Singapore": {"847149": 0.8120519763035391, "847150": 0.23864156434488495,
                  "847180": 0.9576378264256531, "847330": 0.8667712328878263},
    "Malaysia": {"847149": 0.020094234245465723, "847150": 0.012172563206877159,
                 "847180": 0.3272746851087273, "847330": 0.09099907019088997},
}

# USD per kg of the weight-bearing CIF import rows of the national books by code: the density
# of 2021 import rows that carry no weight. Pooled over the compact frame as the estimator loads
# it (every reporter, the EU27 aggregate rows included; 2022-01 to 2026-06), before FOB conversion.
POOLED_DENSITY = {"847149": 181.08175161606016, "847150": 269.9917266186307,
                  "847180": 407.1420930290785, "847330": 182.62182573888458}

# Import books treated as FOB-type when the reference was prepared.
FOB_EXEMPT = frozenset({"us", "mexico", "canada", "australia"})

COMPACT_COLS = ["reporter", "flow", "partner", "code_hs6", "period", "value_usd",
                "weight_kg", "qty", "value_qtycov", "rep_c", "par_c"]
NATIVE_KEYS = ["reporter", "flow", "partner", "rep_c", "par_c", "code_hs6", "period"]
NATIVE_VALUES = ["value_usd", "qty", "weight_kg", "value_qtycov"]
SCALED = ["value_usd", "value_qtycov", "qty", "weight_kg"]

Canon = Callable[[str], "str | None"]


# --------------------------------------------------------------------------------------------
# FOB conversion
# --------------------------------------------------------------------------------------------
def freight_curve(sidecar: Path = FOB_SIDECAR) -> fob.FobModel:
    """The freight-margin model fitted on the pinned US charges file."""
    return fob.FobModel(path=str(sidecar), verbose=False)


def to_fob(frame: pd.DataFrame, curve: fob.FobModel, pooled: dict[str, float] = POOLED_DENSITY,
           fob_exempt: frozenset[str] = FOB_EXEMPT) -> pd.DataFrame:
    """Deflate the CIF import rows of a 2021 compact frame to FOB (value and value_qtycov)."""
    m = frame["flow"].eq("M") & ~frame["reporter"].isin(fob_exempt) & frame["value_usd"].gt(0)
    sub = frame.loc[m]
    missing = set(sub["code_hs6"]) - set(pooled)
    if missing:
        raise ValueError(f"no pooled density for {sorted(missing)}")
    value = sub["value_usd"].to_numpy(dtype=float)
    weight = sub["weight_kg"].to_numpy(dtype=float)
    own = np.divide(value, weight, out=np.full(len(value), np.nan), where=weight > 0)
    fallback = np.array([pooled[c] for c in sub["code_hs6"]], dtype=float)
    density = np.where(np.isfinite(own) & (weight > 0), own, fallback)
    f = 1.0 / (1.0 + curve.tau(sub["period"].map(fob.quarter).to_numpy(), density))
    out = frame.copy()
    out.loc[m, "value_usd"] = value * f
    out.loc[m, "value_qtycov"] = sub["value_qtycov"].to_numpy() * f
    return out


# --------------------------------------------------------------------------------------------
# Fixed inputs, recomputed
# --------------------------------------------------------------------------------------------
def rx_shares_2022(frame: pd.DataFrame, hubs: Iterable[str] = HUB_SPLIT) -> dict:
    """``RX_SHARE_2022`` from a compact frame of the national books (``rep_c`` attached)."""
    out = {}
    for leg in hubs:
        ref = frame[frame["rep_c"].eq(leg) & frame["period"].str.startswith("2022")]
        rx = ref[ref["flow"].eq("RX")].groupby("code_hs6")["value_usd"].sum()
        dx = ref[ref["flow"].eq("DX")].groupby("code_hs6")["value_usd"].sum()
        share = (rx / rx.add(dx, fill_value=0)).clip(0, 1)
        out[leg] = {c: float(share[c]) for c in CODES}
    return out


def pooled_densities(frame: pd.DataFrame, fob_exempt: frozenset[str] = FOB_EXEMPT) -> dict:
    """``POOLED_DENSITY`` from the compact frame of the national books, before FOB conversion."""
    m = (frame["flow"].eq("M") & ~frame["reporter"].isin(fob_exempt)
         & frame["weight_kg"].gt(0) & frame["value_usd"].gt(0))
    g = frame.loc[m].groupby("code_hs6").agg(v=("value_usd", "sum"), w=("weight_kg", "sum"))
    return {c: float(g.loc[c, "v"] / g.loc[c, "w"]) for c in CODES}


# --------------------------------------------------------------------------------------------
# Comtrade
# --------------------------------------------------------------------------------------------
def is_total_record(r: dict) -> bool:
    """The record carrying a cell's total, not one of its mode/partner2/procedure breakdowns."""
    return (r.get("motCode") in (0, "0") and r.get("partner2Code") in (0, "0")
            and str(r.get("customsCode")) == "C00")


def spread_annual(records: list[dict], year: int = 2021) -> list[dict]:
    """Annual records -> twelve monthly records, value, quantity and weight divided by 12."""
    out = []
    for r in records:
        for month in range(1, 13):
            z = r.copy()
            z["period"] = int(f"{year}{month:02}")
            for key in ("primaryValue", "qty", "netWgt"):
                if z.get(key) is not None:
                    z[key] /= 12
            out.append(z)
    return out


def comtrade_records(reporter: str, raw_dir: Path = RAW_DIR) -> list[dict]:
    """One reporter's 2021 records; Austria's annual records when it has no monthly ones."""
    d = Path(raw_dir) / "comtrade"
    records = json.loads((d / f"{reporter.replace(' ', '_')}_2021.json").read_text())["data"]
    if not records and reporter == "austria":
        annual = json.loads((d / "austria_2021_annual.json").read_text())["data"]
        records = spread_annual(annual)
    return records


def comtrade_frame(records: list[dict], reporter: str, canon: Canon) -> pd.DataFrame:
    """Comtrade records -> the compact frame, before the export split and FOB conversion."""
    rows = [r for r in records if is_total_record(r) and r["partnerCode"] != 0]
    d = pd.DataFrame(rows)
    d = d[d.cmdCode.astype(str).isin(CODES)].copy()
    period = d.period.astype(str)
    out = pd.DataFrame({
        "reporter": reporter, "flow": d.flowCode, "partner": d.partnerDesc,
        "code_hs6": d.cmdCode.astype(str), "period": period.str[:4] + "-" + period.str[4:6],
        "value_usd": d.primaryValue.astype(float),
        "weight_kg": d.netWgt.where(~d.isNetWgtEstimated.fillna(True)).fillna(0),
        "qty": d.qty.where(~d.isQtyEstimated.fillna(True) & d.qtyUnitAbbr.eq("u")).fillna(0),
    })
    out["value_qtycov"] = out.value_usd.where(out.qty.gt(0), 0)
    out["rep_c"] = out.reporter.map(canon)
    if reporter == "thailand":            # not in the canonical table; its leg takes this name
        out["rep_c"] = "Thailand"
    out["par_c"] = out.partner.map(canon)
    return out


def split_hub_exports(h: pd.DataFrame, share: dict[str, float]) -> pd.DataFrame:
    """Replace total-export rows with RX (share) and DX (1 - share) rows, by code."""
    x = h[h.flow.eq("X")]
    if not set(x.code_hs6) <= set(share):
        raise ValueError(f"missing re-export shares for {sorted(set(x.code_hs6) - set(share))}")
    fac = x.code_hs6.map(share)
    rx = x.copy()
    rx[SCALED] = rx[SCALED].mul(fac, axis=0)
    rx["flow"] = "RX"
    dx = x.copy()
    dx[SCALED] = dx[SCALED].mul(1 - fac, axis=0)
    dx["flow"] = "DX"
    return pd.concat([h[h.flow.eq("M")], rx, dx], ignore_index=True)


def build_comtrade(canon: Canon, raw_dir: Path = RAW_DIR, reporters=COMTRADE_REPORTERS,
                   curve: fob.FobModel | None = None,
                   rx_share: dict = RX_SHARE_2022,
                   pooled: dict[str, float] = POOLED_DENSITY) -> pd.DataFrame:
    """The prepared Comtrade book for ``reporters`` (in that order)."""
    curve = curve or freight_curve()
    frames = []
    for reporter in reporters:
        h = comtrade_frame(comtrade_records(reporter, raw_dir), reporter, canon)
        for flow in ("M", "X"):
            if h.loc[h.flow.eq(flow), "period"].nunique() != 12:
                raise ValueError(f"incomplete 2021 Comtrade book: {reporter} {flow}")
        leg = h.rep_c.iloc[0]
        if leg in SPLIT_BOOKS:
            if h.flow.eq("RX").any():
                if not h.flow.eq("DX").any():
                    raise ValueError(f"{reporter} files RX without DX")
            else:
                h = split_hub_exports(h, rx_share[leg])
        frames.append(to_fob(h, curve, pooled))
    return pd.concat(frames, ignore_index=True)[COMPACT_COLS]


# --------------------------------------------------------------------------------------------
# Eurostat COMEXT
# --------------------------------------------------------------------------------------------
EU_REPORTERS = {
    "AT": "austria", "BE": "belgium", "BG": "bulgaria", "CY": "cyprus",
    "CZ": "czech republic", "DE": "germany", "DK": "denmark", "EE": "estonia",
    "ES": "spain", "FI": "finland", "FR": "france", "GR": "greece",
    "HR": "croatia", "HU": "hungary", "IE": "ireland", "IT": "italy",
    "LT": "lithuania", "LU": "luxembourg", "LV": "latvia", "MT": "malta",
    "NL": "netherlands", "PL": "poland", "PT": "portugal", "RO": "romania",
    "SE": "sweden", "SI": "slovenia", "SK": "slovakia",
}
EU_FLOWS = {"1": "M", "2": "X"}
EU_PARTNER_NAMES = {
    "CN": "China", "TW": "Taiwan", "HK": "Hong Kong", "MO": "Macau",
    "KR": "South Korea", "KP": "North Korea", "US": "United States",
    "JP": "Japan", "SG": "Singapore", "MY": "Malaysia", "VN": "Vietnam",
    "RU": "Russia", "GB": "United Kingdom", "AE": "United Arab Emirates",
    "CH": "Switzerland", "IR": "Iran", "LA": "Laos", "TR": "Turkey",
    "MK": "North Macedonia", "SY": "Syria", "BN": "Brunei", "CZ": "Czech Republic",
    "MD": "Moldova", "BO": "Bolivia", "VE": "Venezuela", "TZ": "Tanzania",
}
# World, EU / euro-area blocs and their intra/extra totals, and the "Q*" special codes.
EU_PARTNER_DROP = {
    "WORLD", "EU", "EU27_2020", "EU28", "EA", "EA19", "EA20", "EA21",
    "EXT_EU", "EXT_EU27_2020", "EXT_EA", "EXT_EA21",
    "INT_EU", "INT_EU27_2020", "INT_EA", "INT_EA21",
    "QU", "QV", "QW", "QR", "QS", "QP", "QQ", "QX", "QY", "QZ", "QN", "QM",
}
_AGG_KEYWORDS = ("not specified", "stores and provisions", "all countries",
                 "not determined", "european union", "euro area")


def eurostat_clean(responses: Iterable[dict]) -> pd.DataFrame:
    """JSON-stat responses -> rows with a euro value and a country partner.

    Quantity (supplementary units) and weight (100 kg -> kg) are rounded to integers and left
    missing where not reported, as in the clean EU book.
    """
    cells, labels = {}, {}
    for resp in responses:
        labels.update(resp["dimension"]["partner"]["category"].get("label", {}))
        for rep, flow, partner, product, period, indic, val in decode_jsonstat(resp):
            cells.setdefault((rep, flow, partner, product, period), {})[indic] = val
    rows = []
    for (rep, flow, partner, product, period), m in cells.items():
        value = m.get("VALUE_IN_EUROS")
        if value in (None, 0):
            continue
        label = labels.get(partner, partner)
        if partner in EU_PARTNER_DROP or any(k in str(label).lower() for k in _AGG_KEYWORDS):
            continue
        supp, wt100 = m.get("SUPPLEMENTARY_QUANTITY"), m.get("QUANTITY_IN_100KG")
        rows.append({
            "reporter": EU_REPORTERS[rep], "flow": EU_FLOWS[flow],
            "partner": EU_PARTNER_NAMES.get(partner, str(label).split("(")[0].strip().rstrip(",; ").strip()),
            "code_native": product, "code_hs6": product[:6] if len(product) >= 6 else "",
            "period": period, "value_local": int(round(value)), "currency_local": "EUR",
            "qty": int(round(supp)) if supp not in (None, 0) else np.nan,
            "weight_kg": int(round(wt100 * 100)) if wt100 not in (None, 0) else np.nan,
        })
    df = pd.DataFrame(rows)
    df["qty"] = df["qty"].astype(float)
    df["weight_kg"] = df["weight_kg"].astype(float)
    return df.sort_values(["reporter", "flow", "code_native", "period", "partner"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# HMRC OTS
# --------------------------------------------------------------------------------------------
HMRC_FLOWS = {1: ("M", "EU"), 2: ("X", "EU"), 3: ("M", "NON_EU"), 4: ("X", "NON_EU")}
HMRC_PARTNER_NAMES = {
    "USA": "United States", "United States of America": "United States",
    "Korea, South": "South Korea", "Korea (South)": "South Korea",
    "Korea, North": "North Korea", "Viet Nam": "Vietnam",
    "Russian Federation": "Russia", "Czechia": "Czech Republic",
    "Slovak Republic": "Slovakia", "Irish Republic": "Ireland",
}


def hmrc_clean(records: Iterable[dict]) -> pd.DataFrame:
    """OTS records -> CN8 rows summed over partner-name variants.

    A (month, flow type, CN8, partner) record seen twice is read once. Records with no value,
    mass or units are skipped. The HS-6 code is the CN8 code's first six digits.
    """
    rows, seen = [], set()
    for r in records:
        try:
            month_id = int(r["MonthId"])
            flow, eu_flag = HMRC_FLOWS[int(r["FlowTypeId"])]
        except (KeyError, TypeError, ValueError):
            continue
        cn8 = str((r.get("Commodity") or {}).get("Cn8Code") or "").strip()
        partner = (r.get("Country") or {}).get("CountryName")
        if not cn8 or not partner:
            continue
        y, m = month_id // 100, month_id % 100
        if not 1 <= m <= 12:
            continue
        ident = (month_id, int(r["FlowTypeId"]), cn8, str(partner))
        if ident in seen:
            continue
        seen.add(ident)
        value, mass, units = float(r.get("V") or 0), r.get("NM"), r.get("SU")
        if value == 0 and not mass and not units:
            continue
        name = str(partner or "").strip()
        rows.append({
            "flow": flow, "uk_eu_flag": eu_flag, "partner": HMRC_PARTNER_NAMES.get(name, name),
            "code_native": cn8, "code_hs6": cn8[:6], "period": f"{y:04d}-{m:02d}",
            "value_local": value,
            "weight_kg": float(mass) if mass not in (None, "") else None,
            "qty": float(units) if units not in (None, "") else None,
        })
    df = (pd.DataFrame(rows)
          .groupby(["flow", "uk_eu_flag", "partner", "code_native", "code_hs6", "period"],
                   as_index=False)
          .agg(value_local=("value_local", "sum"), weight_kg=("weight_kg", "sum"),
               qty=("qty", "sum")))
    df["value_local"] = df["value_local"].round().astype("int64")
    df["reporter"] = "uk"
    df["currency_local"] = "GBP"
    return df.sort_values(["flow", "code_native", "period", "partner"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Native books
# --------------------------------------------------------------------------------------------
def load_fx_2021(path: Path = FX_2021_CSV) -> dict[tuple[str, str], float]:
    """``{(currency, period): units per USD}`` for the 2021 months."""
    fx = pd.read_csv(path, dtype={"period": str})
    return {(r.currency_local, r.period): float(r.units_per_usd) for r in fx.itertuples()}


def native_frame(clean: pd.DataFrame, fx: dict, canon: Canon) -> pd.DataFrame:
    """A native clean frame -> the compact frame, before FOB conversion."""
    d = clean.copy()
    rates = np.array([fx[(c, p)] for c, p in zip(d["currency_local"], d["period"])])
    d["value_usd"] = d["value_local"] / rates
    d["rep_c"] = d["reporter"].map(canon)
    d["par_c"] = d["partner"].map(canon)
    d["value_qtycov"] = d["value_usd"].where(d["qty"].fillna(0).gt(0), 0)
    d["qty"] = d["qty"].fillna(0)
    d["weight_kg"] = d["weight_kg"].fillna(0)
    d = d.groupby(NATIVE_KEYS, dropna=False, as_index=False)[NATIVE_VALUES].sum()
    d[NATIVE_VALUES] = d[NATIVE_VALUES].astype(float)
    return d


def build_eu(canon: Canon, raw_dir: Path = RAW_DIR, fx: dict | None = None,
             curve: fob.FobModel | None = None,
             pooled: dict[str, float] = POOLED_DENSITY) -> pd.DataFrame:
    """The prepared EU book: 27 member books from the Eurostat responses."""
    files = sorted((Path(raw_dir) / "eurostat").glob("*_2021.json"))
    if len(files) != 2 * len(EU_REPORTERS):
        raise ValueError(f"expected {2 * len(EU_REPORTERS)} Eurostat responses, found {len(files)}")
    clean = eurostat_clean(json.loads(p.read_text()) for p in files)
    frame = native_frame(clean, fx or load_fx_2021(), canon)
    return to_fob(frame, curve or freight_curve(), pooled)


def build_uk(canon: Canon, raw_dir: Path = RAW_DIR, fx: dict | None = None,
             curve: fob.FobModel | None = None,
             pooled: dict[str, float] = POOLED_DENSITY) -> pd.DataFrame:
    """The prepared UK book from the HMRC records."""
    records = json.loads((Path(raw_dir) / "hmrc" / "UK_2021.json").read_text())
    clean = hmrc_clean(records)
    frame = native_frame(clean, fx or load_fx_2021(), canon)
    return to_fob(frame, curve or freight_curve(), pooled)


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------
def csv_bytes(frame: pd.DataFrame) -> bytes:
    """The book as UTF-8 CSV bytes, the form the prepared files hold (gzip-compressed)."""
    return frame.to_csv(index=False).encode("utf-8")


def write_csv_gz(frame: pd.DataFrame, path: Path, member: str | None = None,
                 mtime: int = 0) -> Path:
    """Write a gzip CSV whose header names ``member`` and carries ``mtime``.

    pandas' own gzip writer is used: the compressed stream depends on how the CSV is fed to
    the compressor, so only the same writer reproduces a pandas-written archive. Passing a
    reference file's header member name and timestamp (``gzip_header``) then reproduces that
    file byte for byte when the CSV content is identical.
    """
    path = Path(path)
    member = member if member is not None else path.name.removesuffix(".gz")
    with open(path, "wb") as fh:
        frame.to_csv(fh, index=False,
                     compression={"method": "gzip", "mtime": mtime, "filename": member})
    return path


def gzip_header(path: Path) -> tuple[str, int]:
    """(member name, mtime) from a gzip file's header."""
    with open(path, "rb") as fh:
        head = fh.read(1024)
    if head[:2] != b"\x1f\x8b" or not head[3] & 0x08:
        raise ValueError(f"{path}: gzip header without a member name")
    mtime = int.from_bytes(head[4:8], "little")
    end = head.index(b"\x00", 10)
    return head[10:end].decode("latin-1"), mtime


def build_all(canon: Canon, raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """All three prepared books, keyed by file name."""
    curve = freight_curve()
    fx = load_fx_2021()
    return {"comtrade_prepared.csv.gz": build_comtrade(canon, raw_dir, curve=curve),
            "eu_native_prepared.csv.gz": build_eu(canon, raw_dir, fx, curve),
            "uk_native_prepared.csv.gz": build_uk(canon, raw_dir, fx, curve)}


def check(prepared: dict[str, pd.DataFrame], reference_dir: Path = BASELINE_DIR) -> bool:
    """Compare rebuilt books with the prepared files in ``reference_dir`` (CSV content)."""
    ok = True
    for name, frame in prepared.items():
        with gzip.open(Path(reference_dir) / name, "rb") as fh:
            ref = fh.read()
        new = csv_bytes(frame)
        same = new == ref
        ok &= same
        detail = "" if same else _first_difference(new, ref)
        print(f"  {name}: {'identical' if same else 'DIFFERS'} ({len(frame):,} rows){detail}")
    return ok


def _first_difference(a: bytes, b: bytes) -> str:
    la, lb = a.splitlines(), b.splitlines()
    if len(la) != len(lb):
        return f"; {len(la) - 1:,} vs {len(lb) - 1:,} rows"
    i = next(i for i, (x, y) in enumerate(zip(la, lb)) if x != y)
    n = sum(x != y for x, y in zip(la, lb))
    return f"; {n:,} lines differ, first at line {i + 1}:\n    {la[i]!r}\n    {lb[i]!r}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ai_chip_imports.prepare.books_2021",
                                 description="Build the prepared 2021 reference books.")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=BASELINE_DIR,
                    help="where to write the books (default: data/external/baseline_2021)")
    ap.add_argument("--check", type=Path, metavar="DIR",
                    help="only compare with the prepared books in DIR; write nothing")
    args = ap.parse_args(argv)
    from ..books import canon
    prepared = build_all(canon, args.raw_dir)
    if args.check:
        if not check(prepared, args.check):
            raise SystemExit(1)
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in prepared.items():
        write_csv_gz(frame, args.out_dir / name)


if __name__ == "__main__":
    main()
