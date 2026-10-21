"""Loading the clean customs books into one HS-6 frame, and country-name canonicalization.

``load_all`` reads the newest clean CSV of every book (``data/customs/<book>/clean/``), keeps
the four flows the pipeline uses (M, X, DX, RX), and collapses each book to HS-6:

* **One code granularity at a time.** Books carry their explicit HS-6 line and its native
  sub-lines side by side; summing both would double the value. Where a reporter has the explicit
  HS-6 line for a code, only that line is used; otherwise the native detail is summed. A book
  whose HS-6 line carries no quantity takes its quantity from the detail lines.
* **Quantities in pieces.** Reported quantity units are converted to pieces where they count
  pieces (``UNIT_TO_PIECES``); a kilogram-denominated quantity is used as the row's weight.
  ``value_qtycov`` is the value carried by rows that report pieces, so unit values are only
  computed over the value they describe.
* **Companion files.** Mexico's national cube covers a set of partner corridors from 2023;
  its UN Comtrade submission fills the remaining partner x flow x month cells. Malaysia's
  national file carries no physical fields; they are taken from its own Comtrade submission.

Reporter and partner names are canonicalized for the countries that have a book (``CANON``).
Canonicalization doubles as a selector: a destination without a book keeps ``par_c`` empty,
which is how its re-added flows are found. Only add a country to ``CANON`` together with its
own book, or its re-added flows silently disappear.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from . import codes, paths

CUSTOMS_DIR = str(paths.CUSTOMS_DIR)

# The latest months of every book are provisional (late filings arrive for two to three
# months); the last N months present in the pooled frame are dropped.
DROP_TRAILING_MONTHS = 2

# Books read by load_all (directory name == reporter id in the clean CSV). The EU file holds
# all 27 members as reporters. Order matters only for reproducibility of float sums.
LEGS = ["us", "taiwan", "china", "korea", "malaysia", "singapore", "mexico", "eu", "hk",
        "uk", "japan", "indonesia", "india"]
# Saudi Arabia, Australia and Canada are read from their UN Comtrade submissions.
COMTRADE_LEGS = ["saudi_arabia", "australia", "canada"]
COMTRADE_LEG_NAMES = {"saudi_arabia": "Saudi Arabia", "australia": "Australia", "canada": "Canada"}
LEGS += COMTRADE_LEGS


def NORM(s):
    """Lower-cased, stripped, whitespace-collapsed name ('' for non-strings)."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


# Normalized spelling -> canonical name, for the countries with a book. North Korea is
# deliberately absent so that no spelling of it can map to South Korea.
CANON = {
    # United States
    "united states": "United States", "usa": "United States", "u.s.": "United States",
    "u.s": "United States", "us": "United States", "america": "United States",
    # South Korea
    "south korea": "South Korea", "korea, south": "South Korea", "korea, rep.": "South Korea",
    "republic of korea": "South Korea", "rep. of korea": "South Korea", "korea": "South Korea",
    # China (mainland only; Hong Kong and Taiwan are separate books)
    "china": "China", "people's republic of china": "China", "mainland china": "China",
    "chinese mainland": "China",
    # Hong Kong
    "hk": "Hong Kong", "hong kong": "Hong Kong", "china, hong kong sar": "Hong Kong",
    "hong kong, china": "Hong Kong",
    # Taiwan ("Other Asia, nes" is UN Comtrade's label for Taiwan)
    "taiwan": "Taiwan", "chinese taipei": "Taiwan", "taiwan, province of china": "Taiwan",
    "other asia, nes": "Taiwan",
    "mexico": "Mexico", "malaysia": "Malaysia", "singapore": "Singapore",
    # EU-27 members (partner spellings and lower-case reporter ids normalize alike)
    "austria": "Austria", "belgium": "Belgium", "bulgaria": "Bulgaria", "croatia": "Croatia",
    "cyprus": "Cyprus", "czech republic": "Czech Republic", "czechia": "Czech Republic",
    "czechia（czech republic）": "Czech Republic", "denmark": "Denmark", "estonia": "Estonia",
    "finland": "Finland", "france": "France", "germany": "Germany", "greece": "Greece",
    "hungary": "Hungary", "ireland": "Ireland", "italy": "Italy", "latvia": "Latvia",
    "lithuania": "Lithuania", "luxembourg": "Luxembourg", "malta": "Malta",
    "netherlands": "Netherlands", "poland": "Poland", "portugal": "Portugal",
    "romania": "Romania", "slovakia": "Slovakia", "slovenia": "Slovenia", "spain": "Spain",
    "sweden": "Sweden",
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "great britain": "United Kingdom",
    "japan": "Japan", "indonesia": "Indonesia", "india": "India",
    "saudi arabia": "Saudi Arabia", "kingdom of saudi arabia": "Saudi Arabia",
    "australia": "Australia", "canada": "Canada",
}

EU_MEMBERS = {
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark",
    "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy",
    "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal",
    "Romania", "Slovakia", "Slovenia", "Spain", "Sweden",
}

# Comtrade partner spellings that differ from the national books' (join key for the Malaysia
# field fill; reporter countries canonicalize through CANON first).
COMTRADE_PARTNER_ALIASES = {
    "viet nam": "vietnam", "türkiye": "turkey", "turkiye": "turkey", "russian federation": "russia",
    "china, macao sar": "macau", "macao": "macau", "lao people's dem. rep.": "laos",
    "brunei darussalam": "brunei", "iran, islamic republic of": "iran", "iran (islamic republic of)": "iran",
    "united rep. of tanzania": "tanzania", "bolivia (plurinational state of)": "bolivia",
    "venezuela (bolivarian republic of)": "venezuela", "syrian arab republic": "syria",
    "republic of moldova": "moldova", "united states of america": "united states",
    "cabo verde": "cape verde", "côte d'ivoire": "cote d'ivoire", "state of palestine": "palestine",
    "dem. people's rep. of korea": "north korea", "korea, dem. people's rep. of": "north korea",
    "burma": "myanmar", "east timor": "timor-leste",
}
KG_UNITS = {"KGM", "KILOGRAM", "KG", "KGS", "kg"}


def canon(name):
    """Canonical name of a country with a book, else None."""
    return CANON.get(NORM(name))


USECOLS = ["reporter", "flow", "partner", "code_native", "code_hs6", "period",
           "value_usd", "weight_kg", "qty", "qty_unit"]

# Reported quantity unit -> pieces multiplier. MIL (Singapore) is the UN/ECE "mille", a
# thousand units. Weight-denominated units are not pieces and are handled as weight.
UNIT_TO_PIECES = {"PCE": 1.0, "SET": 1.0, "THOUSAND": 1e3, "MIL": 1e3,
                  "Number of item": 1.0, "p/st": 1.0, "UNIT": 1.0, "NMB": 1.0,
                  "NO": 1.0, "NOS": 1.0, "u": 1.0}
_UNMAPPED_QTY_UNITS = set()   # collected during load, printed once


def find_national_csv(leg):
    """Newest clean CSV of a book (its national file; for the Comtrade books, the Comtrade file)."""
    pat = os.path.join(CUSTOMS_DIR, leg, "clean", f"{leg}_T1-T2_*_pulled*.csv")
    # Mexico's partial checkpoint files (*_grpN.csv) are not books.
    hits = sorted(f for f in glob.glob(pat) if "_comtrade_" not in os.path.basename(f)
                  and not re.search(r"_grp\d+\.csv$", os.path.basename(f)))
    if hits:
        return hits[-1]
    if leg in COMTRADE_LEGS:
        ct = sorted(glob.glob(os.path.join(CUSTOMS_DIR, leg, "clean", f"{leg}_comtrade_T1-T2_*_pulled*.csv")))
        return ct[-1] if ct else None
    return None


def find_mexico_comtrade():
    """Newest clean CSV of Mexico's UN Comtrade submission."""
    hits = sorted(glob.glob(os.path.join(CUSTOMS_DIR, "mexico", "clean",
                                         "mexico_comtrade_T1-T2_*_pulled*.csv")))
    return hits[-1] if hits else None


def _compact_chunk(df):
    """One chunk of a clean CSV: M/X/DX/RX rows with a 6-digit code, quantities in pieces,
    pre-aggregated and tagged with whether each row is the explicit HS-6 line. Granularity is
    resolved after all chunks are read, so a chunk boundary cannot split a decision."""
    df = df[df["flow"].isin(["M", "X", "DX", "RX"])]
    code6 = df["code_hs6"].fillna("")
    df = df[code6.str.len() == 6]
    if df.empty:
        return None
    is_hs6line = (df["code_native"].fillna("") == df["code_hs6"])
    mult = df["qty_unit"].map(UNIT_TO_PIECES)
    _UNMAPPED_QTY_UNITS.update(df.loc[mult.isna() & df["qty_unit"].notna()
                                      & pd.to_numeric(df["qty"], errors="coerce").gt(0),
                                      "qty_unit"].unique())
    value = pd.to_numeric(df["value_usd"], errors="coerce").fillna(0.0)
    qty = (pd.to_numeric(df["qty"], errors="coerce") * mult).fillna(0.0)
    weight = pd.to_numeric(df["weight_kg"], errors="coerce").fillna(0.0)
    kgq = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    weight = weight.where(weight > 0, kgq.where(df["qty_unit"].isin(KG_UNITS), 0.0))
    out = pd.DataFrame({
        "reporter": df["reporter"], "flow": df["flow"], "partner": df["partner"],
        "code_hs6": df["code_hs6"], "period": df["period"], "is_hs6line": is_hs6line.values,
        "value_usd": value,
        "weight_kg": weight,
        "qty": qty,
        "value_qtycov": value.where(qty > 0, 0.0),
    })
    keys = ["reporter", "flow", "partner", "code_hs6", "period", "is_hs6line"]
    return out.groupby(keys, as_index=False).agg(value_usd=("value_usd", "sum"),
                                                 weight_kg=("weight_kg", "sum"),
                                                 qty=("qty", "sum"),
                                                 value_qtycov=("value_qtycov", "sum"))


def _read_compact(path):
    """Read one clean CSV in chunks into the pre-aggregated compact frame."""
    parts = []
    reader = pd.read_csv(path, usecols=USECOLS,
                         dtype={"code_native": str, "code_hs6": str, "reporter": str,
                                "partner": str, "period": str,
                                # quantity columns are mixed-type in some books
                                "qty": str, "qty_unit": str},
                         chunksize=500_000)
    for chunk in reader:
        c = _compact_chunk(chunk)
        if c is not None:
            parts.append(c)
    if not parts:
        return None
    keys = ["reporter", "flow", "partner", "code_hs6", "period", "is_hs6line"]
    return pd.concat(parts, ignore_index=True).groupby(keys, as_index=False).agg(
        value_usd=("value_usd", "sum"), weight_kg=("weight_kg", "sum"), qty=("qty", "sum"),
        value_qtycov=("value_qtycov", "sum"))


RESOLVE_KEYS = ["reporter", "flow", "partner", "code_hs6", "period"]


def _resolve(compact):
    """Apply the one-granularity rule per reporter x code and drop ``is_hs6line``.

    The explicit HS-6 line is used where it exists, else the native detail is summed. Where the
    HS-6 line carries no quantity for a reporter x code but the detail does, quantity (and the
    value it covers) comes from the detail; value and weight are untouched.
    """
    has6 = compact.groupby(["reporter", "code_hs6"])["is_hs6line"].transform("any")
    resolved = compact[(has6 & compact["is_hs6line"]) | (~has6)]
    out = resolved.groupby(RESOLVE_KEYS, as_index=False).agg(
        value_usd=("value_usd", "sum"), weight_kg=("weight_kg", "sum"), qty=("qty", "sum"),
        value_qtycov=("value_qtycov", "sum"))
    lvl = compact.groupby(["reporter", "code_hs6", "is_hs6line"])["qty"].sum().unstack(fill_value=0.0)
    if True in lvl.columns and False in lvl.columns:
        need = set(lvl[(lvl[True] <= 0) & (lvl[False] > 0)].index)
        if need:
            dq = (compact[~compact["is_hs6line"]]
                  .groupby(RESOLVE_KEYS, as_index=False)
                  .agg(_qty_detail=("qty", "sum"), _cov_detail=("value_qtycov", "sum")))
            out = out.merge(dq, on=RESOLVE_KEYS, how="left")
            mask = pd.MultiIndex.from_frame(out[["reporter", "code_hs6"]]).isin(need)
            out.loc[mask, "qty"] = out.loc[mask, "_qty_detail"].fillna(0.0)
            out.loc[mask, "value_qtycov"] = out.loc[mask, "_cov_detail"].fillna(0.0)
            out = out.drop(columns=["_qty_detail", "_cov_detail"])
    return out


def _dedup_key(name):
    """Partner identity for de-duplication: canonical name where one exists (so two spellings of
    a reporter country collapse), else the normalized name (distinct places stay distinct)."""
    return canon(name) or NORM(name)


def load_all():
    """Every book at HS-6, one row per reporter x flow x partner x code x month."""
    parts = []
    for leg in LEGS:
        path = find_national_csv(leg)
        if not path:
            print(f"  [skip] {leg}: no clean CSV found")
            continue
        print(f"  [load] {leg}: {os.path.basename(path)} ({os.path.getsize(path)/1e6:.0f} MB)")
        c = _read_compact(path)
        if c is not None:
            parts.append(c)
    compact = pd.concat(parts, ignore_index=True)
    compact = compact.groupby(["reporter", "flow", "partner", "code_hs6", "period", "is_hs6line"],
                              as_index=False).agg(value_usd=("value_usd", "sum"),
                                                  weight_kg=("weight_kg", "sum"),
                                                  qty=("qty", "sum"),
                                                  value_qtycov=("value_qtycov", "sum"))
    resolved = _resolve(compact)
    if _UNMAPPED_QTY_UNITS:
        print(f"  [qty] unmapped quantity units (contribute no pieces): {sorted(_UNMAPPED_QTY_UNITS)}")

    # Mexico: the national cube wins wherever it has a partner x flow x month cell; Comtrade
    # fills the rest (other partners, and 2022, before the cube starts).
    mx_ct = find_mexico_comtrade()
    if mx_ct is not None:
        print(f"  [load] mexico (comtrade fill): {os.path.basename(mx_ct)} "
              f"({os.path.getsize(mx_ct)/1e6:.0f} MB)")
        ct = _resolve(_read_compact(mx_ct))
        nat = resolved[resolved["reporter"] == "mexico"]
        nat_keys = set(zip(nat["partner"].map(_dedup_key), nat["flow"], nat["period"]))
        keep = [k not in nat_keys
                for k in zip(ct["partner"].map(_dedup_key), ct["flow"], ct["period"])]
        added = ct[keep]
        n_part = added["partner"].map(_dedup_key).nunique()
        print(f"  [mexico] +{len(added):,} Comtrade rows filled ({n_part} partners)")
        resolved = pd.concat([resolved, added], ignore_index=True)
    return _fill_fields_from_comtrade(resolved, "malaysia")


def _comtrade_partner_key(name):
    c = canon(name)
    n = NORM(c) if c else NORM(name)
    return COMTRADE_PARTNER_ALIASES.get(n, n)


def _fill_fields_from_comtrade(resolved, leg):
    """Give a book without physical fields the fields of its own Comtrade submission.

    The same customs records are filed to Comtrade with pieces on the machine codes and chips
    and kilograms on 847330. Rows are joined by flow x partner x code x month: M and X rows take
    Comtrade's quantities directly; DX and RX rows (which Comtrade lacks) inherit the unit value
    and density of the same total-X cell. Values stay the national book's, and a field the
    national book does carry is never overwritten.
    """
    hits = sorted(glob.glob(os.path.join(CUSTOMS_DIR, leg, "clean", f"{leg}_comtrade_T1-T2_*_pulled*.csv")))
    if not hits:
        print(f"  [own-fields] {leg}: no paired Comtrade file -- fields not filled")
        return resolved
    ct = pd.read_csv(hits[-1], dtype={"code_hs6": str, "qty_unit": str}, low_memory=False,
                     usecols=["flow", "partner", "code_hs6", "period", "value_usd", "qty", "qty_unit"])
    ct = ct[ct["flow"].isin(["M", "X"])].copy()
    ct["key"] = ct["partner"].map(_comtrade_partner_key)
    ct["q"] = pd.to_numeric(ct["qty"], errors="coerce").fillna(0.0)
    ct["v"] = pd.to_numeric(ct["value_usd"], errors="coerce").fillna(0.0)
    unit = ct["qty_unit"].astype(str).str.strip()
    keys = ["flow", "key", "code_hs6", "period"]
    u = ct[unit.eq("u")].groupby(keys).agg(v=("v", "sum"), n=("q", "sum"))
    u = u[u["n"] > 0]
    u["uv"] = u["v"] / u["n"]
    kg = ct[unit.isin(KG_UNITS)].groupby(keys).agg(v=("v", "sum"), kg=("q", "sum"))
    kg = kg[kg["kg"] > 0]
    kg["dens"] = kg["v"] / kg["kg"]
    m = (resolved["reporter"] == leg).to_numpy()
    sub = resolved.loc[m, ["flow", "partner", "code_hs6", "period", "value_usd", "qty", "weight_kg"]].copy()
    sub["key"] = sub["partner"].map(_comtrade_partner_key)
    sub["flow_src"] = sub["flow"].replace({"DX": "X", "RX": "X"})
    j = sub.merge(u[["n", "uv"]].reset_index().rename(columns={"flow": "flow_src"}), how="left",
                  on=["flow_src", "key", "code_hs6", "period"])
    j = j.merge(kg[["kg", "dens"]].reset_index().rename(columns={"flow": "flow_src"}), how="left",
                on=["flow_src", "key", "code_hs6", "period"])
    direct = j["flow"].isin(["M", "X"]).to_numpy()
    val = j["value_usd"].to_numpy(dtype=float)
    uv = j["uv"].to_numpy(dtype=float)
    dens = j["dens"].to_numpy(dtype=float)
    qty = np.where(direct, j["n"].fillna(0.0).to_numpy(),
                   np.where(np.isfinite(uv) & (uv > 0), val / np.where(uv > 0, uv, 1.0), 0.0))
    wkg = np.where(direct, j["kg"].fillna(0.0).to_numpy(),
                   np.where(np.isfinite(dens) & (dens > 0), val / np.where(dens > 0, dens, 1.0), 0.0))
    own_q = sub["qty"].to_numpy(dtype=float)
    own_w = sub["weight_kg"].to_numpy(dtype=float)
    qty = np.where(own_q > 0, own_q, qty)
    wkg = np.where(own_w > 0, own_w, wkg)
    resolved.loc[m, "qty"] = qty
    resolved.loc[m, "weight_kg"] = wkg
    resolved.loc[m, "value_qtycov"] = np.where(qty > 0, val, 0.0)
    basket = sub["code_hs6"].isin(["847149", "847150", "847180", "847330"]).to_numpy()
    ai = (sub["period"] >= "2023-07").to_numpy() & basket
    v = val[ai].sum()
    with_pieces = np.where(qty[ai] > 0, val[ai], 0).sum() / v
    with_weight = np.where(wkg[ai] > 0, val[ai], 0).sum() / v
    print(f"  [own-fields] {leg}: fields from {os.path.basename(hits[-1])} -- AI-window basket value "
          f"with pieces {with_pieces:.2f}, with weight {with_weight:.2f}")
    return resolved


def load_taiwan_exports():
    """Taiwan's basket exports to the world, per code x month (its book, trailing months dropped)."""
    path = find_national_csv("taiwan")
    print(f"[load] taiwan: {os.path.basename(path)}")
    res = drop_provisional(_resolve(_read_compact(path)), DROP_TRAILING_MONTHS)
    x = res[(res["flow"] == "X") & res["code_hs6"].isin(codes.BASKET)]
    return x.groupby(["code_hs6", "period"], as_index=False)["value_usd"].sum()


def drop_provisional(df, n):
    """Drop the last ``n`` calendar months present anywhere in ``df``."""
    if n <= 0:
        return df
    periods = sorted(df["period"].dropna().unique())
    if len(periods) <= n:
        return df
    dropped = periods[-n:]
    print(f"  [provisional] dropping trailing months: {', '.join(dropped)}")
    return df[df["period"].isin(set(periods[:-n]))]


def book_frontiers(res):
    """{canonical reporter: last month present}. The pooled frame's frontier is set by the
    fastest book; a reporter whose last month is earlier stopped early, and its later months are
    missing, not zero."""
    d = res[res["rep_c"].notna()]
    return d.groupby("rep_c")["period"].max().to_dict()


# Re-exports are booked by destination, not by the origin of the goods; the origin mix of a
# hub's re-exports is estimated from its imports of the code over the previous N months.
ATTR_WINDOW_MONTHS = 3


def split_legs(df):
    """Reporters that publish a domestic-export / re-export split (have DX rows)."""
    return set(df.loc[df["flow"] == "DX", "rep_c"].dropna().unique())


def _origin_weights(imp_hc, periods, window):
    """Origin shares of one hub x code's imports per period, over the trailing ``window`` months
    (goods dwell in an entrepot before re-export), falling back to the pooled mix when the window
    is empty. Shares are over all origins. Returns {period: Series(origin -> share)}."""
    pooled = imp_hc.groupby("origin")["value_usd"].sum()
    if pooled.sum() <= 0:
        return {}
    pooled_w = pooled / pooled.sum()
    if window is None:
        return {p: pooled_w for p in periods}
    mat = imp_hc.pivot_table(index="period", columns="origin", values="value_usd",
                             aggfunc="sum").sort_index()
    trail = mat.shift(1).rolling(window, min_periods=1).sum()
    out = {}
    for p in periods:
        row = trail.loc[p].dropna() if p in trail.index else None
        out[p] = (row / row.sum()) if (row is not None and row.sum() > 0) else pooled_w
    return out
