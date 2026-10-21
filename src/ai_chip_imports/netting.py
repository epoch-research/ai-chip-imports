"""The single-counted destination universe: retained imports per destination x code x month.

Every destination is counted once, at its own final-market book, under one netting rule
(``uniform_universe``). Sixteen destinations are measured from their own import books: the US,
Hong Kong, Singapore, Malaysia, China, South Korea, Japan, the UK, Indonesia, India, the EU27
(as one aggregate of its members' extra-EU trade), Mexico, Thailand, Saudi Arabia, Australia and
Canada. Every other country (the thin tier) is reached through other books' exports to it.

**The netting rule**, per book x code x month, with a negative carry (a month whose deductions
exceed its imports carries the shortfall into the next month, ``_carry_net``):

    retained = gross imports, excluding the reporter's own-origin re-imports
             - re-exports at full value, where the book separates them (US, HK, SG, MY, Canada).
               EU members and the UK publish no split; their re-export share of exports is set
               from their own production statistics (``ReexportShares``)
             - own-made exports of the two server codes (847149, 847150) at k, the share of an
               exported machine's value that is imported counted content (``_un_k_rows``)
             - for Malaysia, half of the positive quarterly gap between China's server imports
               from Malaysia and Malaysia's declared server exports to China
               (``malaysia_server_mirror``)

k is read from the exporting row group's value density through three anchors -- $250/kg ->
0.10 (small devices: the value is chips and a circuit board), $600/kg -> 0.40 (CPU and storage
servers: imported boards and memory), $2,000/kg -> 0.90 (GPU systems: the board is most of the
machine) -- log-linear between them and flat outside; from unit value where a row has no weight
($1k / $8k / $100k per unit); from the book's value-weighted k where it carries a field on at
least half its server exports; and from its own import content where it carries none. The
deduction is charged first to the same month's imports of the same code and the remainder is
spread over the book's other counted lines. Own-made exports of boards and other units are not
deducted: their inputs are bare chips and memory, which the basket never counts.

**Thin-tier re-adds.** Flows netted out of a hub but landing in no measured book are re-added
as destination rows at the exporter's declared value: US re-exports and own-made exports, the
EU members' extra-EU dispatches, and the declared exports of Malaysia, Singapore, Hong Kong, the
UK, Indonesia and India -- finished codes only (boards are assembly inputs). Taiwan's exports
to destinations without a book form the residual bucket, lifted by one pooled ratio of
importers' declared values to Taiwan's (``residual_value``). Assembly nodes (Vietnam, Thailand
as a destination of re-exports) are excluded: their output returns inside other books.
Exports of China, Korea, Japan, Thailand, Mexico, Saudi Arabia, Australia and Canada to
non-reporters are not re-added.

Bare chips (854231) are carried through and dropped by the estimator.
"""
import glob
import os

import numpy as np
import pandas as pd

from . import books, codes, fob, history, ledger, paths

AI_LO = codes.AI_LO
# Books that flag re-exports, and gross books (no re-export flag, no production statistic).
NETIMP = ["United States", "Hong Kong", "Singapore", "Malaysia"]
CONS_M = ["China", "South Korea", "Japan"]
# Books that net their exports and re-add their finished-code dispatches to non-reporters.
NEWLEG_NETTED = ("United Kingdom", "Indonesia", "India")
# Destinations that assemble and re-ship: re-exports to them are not re-added, and Taiwan's
# exports to them leave the residual bucket (their output returns inside other books).
ASSEMBLY_NODES = {"vietnam", "viet nam", "thailand"}
FINISHED = ("847149", "847150", "847180")      # finished hardware; boards are assembly inputs
TH_FINISHED = EU_FINISHED = SGHK_FINISHED = MY_FINISHED = NEWLEG_FINISHED = US_DX_FINISHED = FINISHED
SGHK_HUBS = ("Singapore", "Hong Kong")
# China's imports of China-origin goods are bonded-zone round trips, not foreign hardware. The
# own-origin exclusion applies on every book but only bites in China. Vietnam-origin flows are
# kept: they are a plausible route for diverted accelerators.
CN_EXCLUDED_ORIGINS = {"china"}

# --- k: imported counted content per dollar of own-made server exports -----------------------
K_SERVER_CODES = ("847149", "847150")
K_DENS_ANCH = ((250.0, 0.10), (600.0, 0.40), (2000.0, 0.90))      # ($/kg, k)
K_UV_ANCH = ((1000.0, 0.10), (8000.0, 0.40), (100000.0, 0.90))    # ($/unit, k)
K_FALLBACK = 0.40                                                 # no reading at all
# Band corners: less and more deduction.
K_BRACKET = {"klo": (((250.0, 0.05), (600.0, 0.30), (2000.0, 0.80)),
                     ((1000.0, 0.05), (8000.0, 0.30), (100000.0, 0.80))),
             "khi": (((250.0, 0.15), (600.0, 0.50), (2000.0, 0.95)),
                     ((1000.0, 0.15), (8000.0, 0.50), (100000.0, 0.95)))}
K_WINDOW0 = "2024-01"            # window for book-average k and the re-export shares

# --- Malaysia's undeclared server exports ------------------------------------------------------
# The share of the China-import / Malaysia-export gap deducted from Malaysia. A structural
# judgement about undeclared onward shipments, not an estimated smuggling rate; 0 and 1 are the
# band endpoints.
MY_SERVER_MIRROR_SHARE = float(os.environ.get("AI_CHIP_IMPORTS_MALAYSIA_MIRROR_SHARE", "0.5"))
if not np.isfinite(MY_SERVER_MIRROR_SHARE) or not 0 <= MY_SERVER_MIRROR_SHARE <= 1:
    raise ValueError("AI_CHIP_IMPORTS_MALAYSIA_MIRROR_SHARE must be between 0 and 1")

# --- re-export shares from production statistics -----------------------------------------------
# Eurostat PRODCOM sold production of 26.20.14/15 (computers and their units), EUR.
K_PRODCOM = str(paths.EXTERNAL_DIR / "eurostat_prodcom_2620_sold_production.csv")
K_EURUSD = 1.08
# Processing-fee producers: their sold production understates output, so only the import cap
# is used.
K_CONTRACT_MFG = ("Hungary", "Czech Republic", "Poland")
K_GEO = {"AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia", "CY": "Cyprus",
         "CZ": "Czech Republic", "DK": "Denmark", "EE": "Estonia", "FI": "Finland", "FR": "France",
         "DE": "Germany", "GR": "Greece", "EL": "Greece", "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
         "LV": "Latvia", "LT": "Lithuania", "LU": "Luxembourg", "MT": "Malta", "NL": "Netherlands",
         "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SK": "Slovakia", "SI": "Slovenia",
         "ES": "Spain", "SE": "Sweden"}
# Non-EU books with their own production statistic: ONS ProdCom for the UK (GBP).
K_GBPUSD = 1.27
K_PRODCOM_OWN = {"United Kingdom": (str(paths.EXTERNAL_DIR / "ons_prodcom_2620_sold_production.csv"),
                                    "prodval_gbp", K_GBPUSD)}   # book -> (file, value column, FX)
# Books with a declared split: (re-export flow deducted at full value, own-made flow at k).
UN_SPLIT = {"United States": ("RX", "DX"), "Singapore": ("RX", "DX"), "Malaysia": ("RX", "DX"),
            "Hong Kong": ("RX", "DX"), "Canada": ("RX", "DX")}
UN_LEDGER = {}     # per universe tag: the per-book AI-window netting ledger
UN_LEG_RX = {}     # the non-EU books' re-export share table (headline)


def _grp(df):
    return df.groupby(["code_hs6", "period"], as_index=False)["value_usd"].sum()


def _basket_M(res, reporter, partner=None):
    d = res[(res["rep_c"] == reporter) & (res["flow"] == "M") & res["code_hs6"].isin(codes.BASKET)]
    if partner is not None:
        d = d[d["par_c"] == partner]
    return d


def _k_from(x, anch):
    """k at value ``x`` from anchors ((x, k), ...): piecewise-linear in log x, flat outside."""
    if x is None or not np.isfinite(x) or x <= 0:
        return np.nan
    return float(np.interp(np.log10(x), [np.log10(a) for a, _ in anch], [k for _, k in anch]))


def _pkey(name):
    c = books.canon(name)
    return books.NORM(c) if c else books.NORM(name)


class ReexportShares:
    """Re-export shares of server exports for books that publish no re-export split.

    For each EU member, and for the UK: sold production of computers over the window (the latest
    two years' mean) caps own-made exports from above, so ``1 - production / exports`` is a floor
    on the re-export share; imports cap it from above (a book cannot re-export more than it
    imported). The share is the midpoint of floor and cap, or the cap where the floor exceeds
    it. Contract manufacturers use the cap only. Members without data use the EU's
    extra-EU-export-weighted share.
    """

    def __init__(self, res):
        self.eu_rx, self.eu_rx_table, self.eu_rx_wtd = self._eu_rx_shares(res)
        self.leg_rx, self.leg_rx_table = self._leg_rx_shares(res)

    def _eu_rx_shares(self, res):
        prod = {}
        if os.path.exists(K_PRODCOM):
            pc = pd.read_csv(K_PRODCOM, dtype={"geo": str, "prccode": str})
            pc = pc[pc["geo"].isin(K_GEO) & pc["prccode"].isin(["26201400", "26201500"])]
            pc["member"] = pc["geo"].map(K_GEO)
            yr = pc.groupby(["member", "year"])["prodval_eur"].sum()
            for m in yr.index.get_level_values(0).unique():
                ys = yr.xs(m)
                latest = ys.index.max()
                prev = latest - 1
                prod[m] = float(ys.reindex([prev, latest]).dropna().mean())   # annual EUR
        W = res[res["rep_c"].isin(books.EU_MEMBERS) & res["code_hs6"].isin(K_SERVER_CODES)
                & (res["period"] >= K_WINDOW0)]
        months = W["period"].nunique()
        X = W[W["flow"] == "X"].groupby("rep_c")["value_usd"].sum()
        Xx = W[(W["flow"] == "X") & ~W["par_c"].isin(books.EU_MEMBERS)].groupby("rep_c")["value_usd"].sum()
        M = W[W["flow"] == "M"].groupby("rep_c")["value_usd"].sum()
        rows = []
        for m in sorted(books.EU_MEMBERS):
            x = float(X.get(m, 0.0))
            if x <= 0:
                continue
            mm = float(M.get(m, 0.0))
            cap = min(x, mm) / x
            if m in K_CONTRACT_MFG or m not in prod:
                lo, rx = 0.0, cap
            else:
                own_max = prod[m] * (months / 12.0) * K_EURUSD
                lo = max(0.0, 1.0 - own_max / x)
                rx = 0.5 * (lo + cap) if cap > lo else cap
            rows.append({"member": m, "X_B": x / 1e9, "X_extraEU_B": float(Xx.get(m, 0.0)) / 1e9,
                         "M_B": mm / 1e9, "prod_annual_MEUR": prod.get(m, np.nan) / 1e6 if m in prod else np.nan,
                         "rx_lo": lo, "rx_cap": cap, "rx": rx})
        t = pd.DataFrame(rows).set_index("member") if rows else pd.DataFrame()
        if len(t) and t["X_extraEU_B"].sum() > 0:
            wtd = float((t["rx"] * t["X_extraEU_B"]).sum() / t["X_extraEU_B"].sum())
        else:
            wtd = 1.0
        return (t["rx"].to_dict() if len(t) else {}), t, wtd

    def _leg_rx_shares(self, res):
        rows = {}
        for leg, (path, col, fx) in K_PRODCOM_OWN.items():
            if not os.path.exists(path):
                continue
            pc = pd.read_csv(path, dtype={"prccode": str})
            pc = pc[pc["reporter"].eq(leg) & pc["prccode"].isin(["26201400", "26201500"])]
            ys = pc.groupby("year")[col].sum()
            if ys.empty:
                continue
            latest = int(ys.index.max())
            prod = float(ys.reindex([latest - 1, latest]).dropna().mean())   # annual, local currency
            W = res[(res["rep_c"] == leg) & res["code_hs6"].isin(K_SERVER_CODES)
                    & (res["period"] >= K_WINDOW0)]
            months = int(W["period"].nunique())
            x = float(W.loc[W["flow"] == "X", "value_usd"].sum())
            if x <= 0 or months == 0:
                continue
            mm = float(W.loc[W["flow"] == "M", "value_usd"].sum())
            cap = min(x, mm) / x
            own_max = prod * fx * (months / 12.0)
            lo = max(0.0, 1.0 - own_max / x)
            rx = 0.5 * (lo + cap) if cap > lo else cap
            rows[leg] = {"X_B": x / 1e9, "M_B": mm / 1e9, "prod_annual_local_M": prod / 1e6, "fx": fx,
                         "months": months, "rx_lo": lo, "rx_cap": cap, "rx": rx}
        t = pd.DataFrame.from_dict(rows, orient="index") if rows else pd.DataFrame()
        return {leg: float(r["rx"]) for leg, r in rows.items()}, t


_RX_CACHE = {}


def reexport_shares(res):
    """``ReexportShares`` for the frame ``res``, cached by its identity."""
    key = id(res)
    if key not in _RX_CACHE:
        _RX_CACHE.clear()
        _RX_CACHE[key] = ReexportShares(res)
    return _RX_CACHE[key]


def _k_anchors(k_anch):
    """(density anchors, unit-value anchors): the headline (None) or a K_BRACKET corner."""
    if k_anch is None:
        return K_DENS_ANCH, K_UV_ANCH
    return k_anch


def _carry_net(M, OUT):
    """``M - OUT`` per code x period with a negative carry: d_t = flow_t + carry_{t-1};
    net_t = max(d_t, 0); carry_t = min(d_t, 0). The carry runs over the union of months, but
    only (code, period) cells present in ``M`` are returned."""
    m = M.merge(OUT, on=["code_hs6", "period"], how="outer")
    m[["value_usd", "out"]] = m[["value_usd", "out"]].fillna(0.0)
    m["d"] = m["value_usd"] - m["out"]
    m = m.sort_values(["code_hs6", "period"]).reset_index(drop=True)
    net = pd.Series(0.0, index=m.index)
    for _, g in m.groupby("code_hs6", sort=False):
        c = 0.0
        for i, v in zip(g.index, g["d"].to_numpy()):
            x = v + c
            net.loc[i] = max(x, 0.0)
            c = min(x, 0.0)
    m["value_usd"] = net
    keep = set(zip(M["code_hs6"], M["period"]))
    m = m[[k in keep for k in zip(m["code_hs6"], m["period"])]]
    return m[["code_hs6", "period", "value_usd"]]


def eu_extra_flows(res, flow):
    """The EU members' extra-EU rows of ``flow`` (the EU27 aggregate reporter is never used)."""
    return res[res["rep_c"].isin(books.EU_MEMBERS) & (res["flow"] == flow)
               & res["code_hs6"].isin(codes.BASKET) & ~res["par_c"].isin(books.EU_MEMBERS)]


def _dense(x, periods):
    """Exporter rows -> (dest, code_hs6, period, value_usd), every destination x code present
    over every period in ``periods`` (zeros included), so a corridor's 2021 and 2022 anchors
    form even where it traded only some months."""
    p = (x.pivot_table(index=["dest", "code_hs6"], columns="period", values="value_usd",
                       aggfunc="sum").reindex(columns=periods).fillna(0.0))
    return p.stack().rename("value_usd").reset_index()[["dest", "code_hs6", "period", "value_usd"]]


def eu_x_readd_rows(res):
    """EU members' extra-EU dispatches of finished codes to destinations without a book."""
    x = eu_extra_flows(res, "X")
    x = x[x["code_hs6"].isin(EU_FINISHED)].copy()
    norm = x["partner"].map(books.NORM)
    x = x[x["par_c"].isna() & ~norm.isin({"world", "total"} | ASSEMBLY_NODES)]
    x["dest"] = x["partner"].map(books.NORM).str.title()
    periods = sorted(eu_extra_flows(res, "X")["period"].unique())
    return _dense(x, periods)


def _tag(df, dest_name):
    df = df.copy()
    df["dest"] = dest_name
    return df[["dest", "code_hs6", "period", "value_usd"]]


def rx_readd_rows(res, hubs=("United States",)):
    """Hubs' re-exports to destinations without a book, over each hub's own reporting span
    (a union of spans would read zeros past a hub's frontier as a dip)."""
    rx = res[(res["flow"] == "RX") & res["code_hs6"].isin(codes.BASKET)
             & res["rep_c"].isin(hubs)].copy()
    norm = rx["partner"].map(books.NORM)
    rx["dest"] = rx["partner"].map(books.NORM).str.title()
    rx["keep"] = rx["par_c"].isna() & ~norm.isin({"world", "total"} | ASSEMBLY_NODES)
    parts = []
    for _, g in rx.groupby("rep_c"):
        gk = g[g["keep"]]
        if gk.empty:
            continue
        periods = sorted(g["period"].unique())
        p = (gk.pivot_table(index=["dest", "code_hs6"], columns="period", values="value_usd",
                            aggfunc="sum").reindex(columns=periods).fillna(0.0))
        parts.append(p.stack().rename("value_usd").reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out.groupby(["dest", "code_hs6", "period"], as_index=False)["value_usd"].sum()


def hub_x_readd_rows(res, reporter, finished, flow="X"):
    """A book's declared exports (``flow``: total X, or the US's DX) of finished codes to
    destinations without a book, over the book's own reporting span."""
    x = res[(res["rep_c"] == reporter) & (res["flow"] == flow)
            & res["code_hs6"].isin(finished)].copy()
    norm = x["partner"].map(books.NORM)
    x = x[x["par_c"].isna() & ~norm.isin({"world", "total"} | ASSEMBLY_NODES)]
    periods = sorted(res[(res["rep_c"] == reporter) & (res["flow"] == flow)]["period"].unique())
    if x.empty or not periods:
        print(f"  [x-readd] {reporter}: no finished-code exports to non-reporters -- nothing re-added")
        return pd.DataFrame(columns=["dest", "code_hs6", "period", "value_usd"])
    x["dest"] = x["partner"].map(books.NORM).str.title()
    return _dense(x, periods)


def thailand_books(basket_only=True):
    """Thailand's newest clean book (HS-6 rows with its 11-digit lines' physical fields), on the
    FOB basis, or None if absent. Read outside the loader because only the 11-digit lines carry
    quantities."""
    hits = sorted(f for f in glob.glob(os.path.join(books.CUSTOMS_DIR, "thailand", "clean",
                                                    "thailand_T1-T2_*_pulled*.csv"))
                  if "_comtrade_" not in os.path.basename(f))
    if not hits:
        print("[thailand] no thailand/clean CSV found -- book skipped")
        return None
    th = pd.read_csv(hits[-1], dtype={"code_native": str, "code_hs6": str}, low_memory=False)
    th6 = th[th["code_native"] == th["code_hs6"]]          # HS-6 rows (4-digit parents out)
    th = _thai_own_fields(th, th6)
    if basket_only:
        th = th[th["code_hs6"].isin(codes.BASKET)]
    return fob.fobize_book(th)


def _thai_own_fields(th, th6):
    """Attach the 11-digit lines' quantities to their HS-6 rows where the lines reconcile to the
    HS-6 value within 5%: pieces when the 'u' lines carry at least 90% of the 11-digit value,
    weight when the 'kg' lines do (a mixed cell gets neither)."""
    t11 = th[th["code_native"].astype(str).str.len() == 11].copy()
    if t11.empty:
        return th6
    t11["q"] = pd.to_numeric(t11["qty"], errors="coerce").fillna(0.0)
    t11["v"] = pd.to_numeric(t11["value_usd"], errors="coerce").fillna(0.0)
    unit = t11["qty_unit"].astype(str).str.strip().str.lower()
    t11["n_u"] = np.where(unit.eq("u"), t11["q"], 0.0)
    t11["kg"] = np.where(unit.eq("kg"), t11["q"], 0.0)
    t11["v_u"] = np.where(unit.eq("u"), t11["v"], 0.0)
    t11["v_kg"] = np.where(unit.eq("kg"), t11["v"], 0.0)
    keys = ["flow", "partner", "code_hs6", "period"]
    agg = t11.groupby(keys, as_index=False).agg(v11=("v", "sum"), n_u=("n_u", "sum"), kg=("kg", "sum"),
                                                v_u=("v_u", "sum"), v_kg=("v_kg", "sum"))
    base = th6.merge(agg, on=keys, how="left")
    val = pd.to_numeric(base["value_usd"], errors="coerce").fillna(0.0)
    ok = base["v11"].notna() & ((base["v11"] - val).abs() <= 0.05 * val.clip(lower=1.0))
    piece_ok = ok & (base["v_u"] >= 0.9 * base["v11"])
    kg_ok = ok & (base["v_kg"] >= 0.9 * base["v11"])
    base["qty"] = np.where(piece_ok, base["n_u"], np.nan)
    base["weight_kg"] = np.where(kg_ok, base["kg"], np.nan)
    base = base.drop(columns=["v11", "n_u", "kg", "v_u", "v_kg"])
    ai = (base["period"] >= "2023-07") & base["code_hs6"].isin(["847149", "847150", "847180", "847330"])
    for fl in ("M", "X"):
        sel = ai & (base["flow"] == fl)
        v = val[sel].sum()
        if v > 0:
            print(f"  [own-fields] thailand {fl}: 11-digit lines give pieces on "
                  f"{val[sel & base['qty'].fillna(0).gt(0)].sum() / v:.2f} and weight on "
                  f"{val[sel & base['weight_kg'].fillna(0).gt(0)].sum() / v:.2f} of the AI-window basket value")
    return base


def _un_legs():
    """The measured books, in netting order."""
    legs = list(NETIMP) + list(CONS_M) + list(NEWLEG_NETTED)
    legs += ["Mexico", "European Union", "Thailand"]
    legs += [books.COMTRADE_LEG_NAMES[k] for k in books.COMTRADE_LEGS]
    return legs


def _un_thailand(res):
    """Thailand's book as a reporter frame, cut at the pooled frontier."""
    th = thailand_books()
    if th is None:
        return None
    th = th[th["period"] <= res["period"].max()].copy()
    th["rep_c"] = "Thailand"
    th["par_c"] = th["partner"].map(books.canon)
    for c in ("qty", "weight_kg"):
        th[c] = pd.to_numeric(th[c], errors="coerce") if c in th.columns else np.nan
    return th


def _un_own_excluded(d, leg):
    norm = d["partner"].map(books.NORM)
    return d[~(d["par_c"] == leg) & ~(norm == books.NORM(leg))]


def _un_import_content(cm):
    """(book, code, quarter) -> the content the book's own imports of the code carry in the
    content model (cell, line prior, eligible mix, flat constant): k for export books that
    carry no physical field."""
    from . import content

    def f(leg, code, q):
        if cm is None:
            return K_FALLBACK
        c = cm.cells.get((leg, code, q))
        if c is None:
            c = cm.prior.get((leg, code))
        if c is None and code in content.MEASURED_CODES:
            c = cm.mix(cm._side(leg)).get(q, np.nan)
            if not np.isfinite(c):
                c = None
        if c is None:
            c = content.SALE_FLAT.get(code, codes.CHIP_CONTENT.get(code, K_FALLBACK))
        return float(c)
    return f


def _un_k_rows(rows, leg, danch, uanch, import_c, import_leg=None, k_shift=None):
    """k per own-made export row (the caller zeroes non-server codes).

    The row group's (partner x code x quarter) value density, else its unit value; rows with
    neither take the book's value-weighted k where it carries a field on at least half of its
    server exports in the window, and its own import content where it does not.
    ``k_shift`` (book -> additive shift, for uncertainty draws) moves the server-code k read
    from the book's own fields, clipped to [0, 1]; k taken from import content is left to the
    content model.
    """
    from . import content
    if rows.empty:
        return pd.Series(dtype="float")
    vq = rows["value_qtycov"] if "value_qtycov" in rows.columns else rows["value_usd"]
    vq = pd.Series(np.where(vq.fillna(0) > 0, vq, rows["value_usd"]), index=rows.index)
    pk = rows["partner"].map(_pkey)
    q = rows["period"].map(codes.quarter)
    g = (rows.assign(_pk=pk, _q=q, _vq=vq).groupby(["_pk", "code_hs6", "_q"])
         .agg(v=("value_usd", "sum"), vq=("_vq", "sum"), kg=("weight_kg", "sum"), n=("qty", "sum")))
    dens = (g["v"] / g["kg"]).where(g["kg"] > 0)
    uv = (g["vq"] / g["n"]).where(g["n"] > 0)
    kd = dens.map(lambda x: _k_from(x, danch))
    ku = uv.map(lambda x: _k_from(x, uanch))
    kk = kd.where(kd.notna(), ku)
    idx = pd.MultiIndex.from_arrays([pk, rows["code_hs6"], q])
    k = pd.Series(kk.reindex(idx).to_numpy(dtype=float), index=rows.index, dtype="float")
    win = rows["code_hs6"].isin(K_SERVER_CODES) & (rows["period"] >= K_WINDOW0)
    known = win & k.notna()
    v_win = float(rows.loc[win, "value_usd"].sum())
    v_known = float(rows.loc[known, "value_usd"].sum())
    il = import_leg or leg
    if v_win > 0 and v_known / v_win >= content.QTYCOV_MIN:
        kbar = float((rows.loc[known, "value_usd"] * k[known]).sum() / v_known)
        k = k.fillna(kbar)
        observed = k.notna()
    else:
        observed = k.notna()
        fill = pd.Series([import_c(il, c, qq) for c, qq in zip(rows["code_hs6"], q)],
                         index=rows.index, dtype="float")
        k = k.where(observed, fill)
    if k_shift is not None:
        shifted = observed & rows["code_hs6"].isin(K_SERVER_CODES)
        k.loc[shifted] = (k.loc[shifted] + k_shift(il)).clip(0, 1)
    return k


def _un_spread(deductions, Mn):
    """Place a deduction frame: each code's deduction first against the month's imports of the
    same code; any remainder over the month's other counted lines in proportion to their size
    (kept on the code itself if there are none, where the carry holds it)."""
    if deductions.empty:
        return deductions
    M = Mn.groupby(["code_hs6", "period"])["value_usd"].sum()
    Oc = deductions.groupby(["code_hs6", "period"])["out"].sum()
    cap = M.reindex(Oc.index).fillna(0.0).to_numpy()
    same = np.minimum(Oc.to_numpy(), cap)
    parts = [pd.DataFrame({"code_hs6": Oc.index.get_level_values(0), "period": Oc.index.get_level_values(1),
                           "out": same})]
    spill = pd.Series(Oc.to_numpy() - same, index=Oc.index)
    spill = spill[spill > 0]
    for (code, period), amt in spill.items():
        others = Mn[(Mn["period"] == period) & (Mn["code_hs6"] != code) & (Mn["value_usd"] > 0)]
        if others.empty:
            parts.append(pd.DataFrame({"code_hs6": [code], "period": [period], "out": [amt]}))
            continue
        sh = others["value_usd"] / others["value_usd"].sum()
        parts.append(pd.DataFrame({"code_hs6": others["code_hs6"].to_numpy(), "period": period,
                                   "out": (sh * amt).to_numpy()}))
    out = pd.concat(parts, ignore_index=True).groupby(["code_hs6", "period"], as_index=False)["out"].sum()
    return out[out["out"] != 0.0]


def malaysia_server_mirror(res, share=None):
    """Malaysia's deduction for undeclared server exports to China.

    Per server code and quarter: ``share`` x max(China's imports from Malaysia - Malaysia's
    declared exports to China, 0), both on resolved FOB books, spread over China's recorded
    arrival months by value. Applies to every year including the reference year; China gets no
    extra credit. A gap need not be an undeclared shipment: timing, indirect routing and
    origin/destination asymmetry also create gaps. Malaysia's declared exports are its total X;
    months without one (its 2021 history has only the DX/RX split) read DX + RX.
    """
    share = MY_SERVER_MIRROR_SHARE if share is None else share
    if not np.isfinite(share) or not 0 <= share <= 1:
        raise ValueError("Malaysia server mirror share must be between 0 and 1")
    cols = ["code_hs6", "period", "out"]
    if share == 0:
        return pd.DataFrame(columns=cols)
    servers = res[res.code_hs6.isin(K_SERVER_CODES)]
    cn = servers[servers.rep_c.eq("China") & servers.flow.eq("M")
                 & servers.partner.map(books.NORM).eq("malaysia")].copy()
    my = servers[servers.rep_c.eq("Malaysia") & servers.partner.map(books.NORM).eq("china")]
    x_months = set(servers.loc[servers.rep_c.eq("Malaysia") & servers.flow.eq("X"), "period"])
    my = my[my.flow.eq("X") | (my.flow.isin(["DX", "RX"]) & ~my.period.isin(x_months))].copy()
    for frame in (cn, my):
        frame["q"] = frame.period.map(codes.quarter)
    keys = ["code_hs6", "q"]
    gap = cn.groupby(keys).value_usd.sum().subtract(
        my.groupby(keys).value_usd.sum(), fill_value=0).clip(lower=0).rename("gap")
    months = cn.groupby(keys + ["period"], as_index=False).value_usd.sum().merge(
        gap.reset_index(), on=keys)
    total = months.groupby(keys).value_usd.transform("sum")
    months["out"] = (share * months.gap * months.value_usd / total.where(total > 0)).fillna(0)
    return months[cols]


def _un_leg_rows(res, th, ck, leg, danch, uanch, import_c, book_ledger, mirror_share=None,
                 k_shift=None):
    """One book under the netting rule -> (dest, code_hs6, period, value_usd), plus a ledger row."""
    empty = res.iloc[0:0]
    if leg == "European Union":
        Mrows = eu_extra_flows(res, "M")
    elif leg == "Thailand":
        Mrows = th[th["flow"] == "M"] if th is not None else empty
    else:
        Mrows = _basket_M(res, leg)
    Mrows = _un_own_excluded(Mrows, leg)
    M = _grp(Mrows)
    Mn = M[~M["code_hs6"].isin(codes.CHIPS)]
    if leg == "European Union":
        x = eu_extra_flows(res, "X")
        x = x[~x["code_hs6"].isin(codes.CHIPS)].copy()
        rx = x["rep_c"].map(lambda m: ck.eu_rx.get(m, ck.eu_rx_wtd)).astype(float)
        full = x.assign(value_usd=x["value_usd"] * rx)
        own = x.assign(value_usd=x["value_usd"] * (1.0 - rx))
        # k is read on the members' unscaled rows (value / kg)
        k = (pd.concat([_un_k_rows(x[x["rep_c"] == m], m, danch, uanch, import_c, leg, k_shift)
                        for m in x["rep_c"].unique()]).reindex(own.index)
             if not own.empty else pd.Series(dtype="float"))
    elif leg in ck.leg_rx:
        x = res[(res["rep_c"] == leg) & (res["flow"] == "X")]
        x = x[x["code_hs6"].isin(codes.BASKET) & ~x["code_hs6"].isin(codes.CHIPS)].copy()
        rx = float(ck.leg_rx[leg])
        full = x.assign(value_usd=x["value_usd"] * rx)
        own = x.assign(value_usd=x["value_usd"] * (1.0 - rx))
        k = _un_k_rows(x, leg, danch, uanch, import_c, k_shift=k_shift)
    else:
        if leg == "Thailand":
            full, own = empty, (th[th["flow"] == "X"] if th is not None else empty)
        else:
            fr, fo = UN_SPLIT.get(leg, (None, "X"))
            full = res[(res["rep_c"] == leg) & (res["flow"] == fr)] if fr else empty
            own = res[(res["rep_c"] == leg) & (res["flow"] == fo)]
        full = full[full["code_hs6"].isin(codes.BASKET) & ~full["code_hs6"].isin(codes.CHIPS)]
        own = own[own["code_hs6"].isin(codes.BASKET) & ~own["code_hs6"].isin(codes.CHIPS)].copy()
        k = _un_k_rows(own, leg, danch, uanch, import_c, k_shift=k_shift)
    srv = own["code_hs6"].isin(K_SERVER_CODES)
    w = k.copy()
    w[~srv] = 0.0
    OUT_full = (_grp(full).rename(columns={"value_usd": "out"}) if not full.empty
                else pd.DataFrame(columns=["code_hs6", "period", "out"]))
    OUT_own = (_grp(own.assign(value_usd=own["value_usd"] * w.to_numpy())).rename(columns={"value_usd": "out"})
               if not own.empty else pd.DataFrame(columns=["code_hs6", "period", "out"]))
    OUT = (pd.concat([OUT_full, _un_spread(OUT_own, Mn)], ignore_index=True)
           .groupby(["code_hs6", "period"], as_index=False)["out"].sum())
    mirror = pd.DataFrame(columns=["code_hs6", "period", "out"])
    if leg == "Malaysia":
        mirror = malaysia_server_mirror(res, mirror_share)
        if len(mirror):
            OUT = pd.concat([OUT, mirror], ignore_index=True).groupby(
                ["code_hs6", "period"], as_index=False)["out"].sum()
    netted = _carry_net(M, OUT)
    ai = lambda f, col: float(f[f["period"].map(codes.quarter) >= AI_LO][col].sum()) / 1e9   # noqa: E731
    own_srv = own[srv.to_numpy()] if len(own) else own
    ws = own_srv[own_srv["period"] >= K_WINDOW0] if len(own_srv) else own_srv
    wv = float(ws["value_usd"].sum()) if len(ws) else 0.0
    cov_n = float(ws.loc[ws["qty"].fillna(0) > 0, "value_usd"].sum()) / wv if wv > 0 else 0.0
    cov_kg = float(ws.loc[ws["weight_kg"].fillna(0) > 0, "value_usd"].sum()) / wv if wv > 0 else 0.0
    from . import content
    fields = "+".join([f for f, c in (("units", cov_n), ("kg", cov_kg)) if c >= content.QTYCOV_MIN]) or "none"
    book_ledger.append({"leg": leg, "gross_M": ai(Mn, "value_usd"),
                   "reexports_at_1": ai(OUT_full, "out") if len(OUT_full) else 0.0,
                   "own_x_gross": ai(own[["period", "value_usd"]], "value_usd") if len(own) else 0.0,
                   "own_servers_gross": ai(own_srv[["period", "value_usd"]], "value_usd") if len(own_srv) else 0.0,
                   "own_servers_at_k": ai(OUT_own, "out") if len(OUT_own) else 0.0,
                   "k_servers": _k_ai(own_srv, k[srv]),
                   "export_fields": fields,
                   "netted_level": ai(netted[~netted["code_hs6"].isin(codes.CHIPS)], "value_usd"),
                   "server_mirror_deduction": ai(mirror, "out") if len(mirror) else 0.0})
    return _tag(netted, leg)


def _k_ai(own_srv, k):
    """Value-weighted k on own-made server exports over the AI window."""
    if not len(own_srv):
        return np.nan
    ai = (own_srv["period"].map(codes.quarter) >= AI_LO).to_numpy()
    v = own_srv["value_usd"].to_numpy()[ai]
    kk = np.asarray(k, dtype=float)[ai]
    return float((v * kk).sum() / v.sum()) if v.sum() > 0 else np.nan


def uniform_universe(res, k_anch=None, content_model=None, tag="headline", mirror_share=None,
                     k_shift=None):
    """Every measured book under the netting rule, plus the thin-tier re-adds, with the 2021
    history attached and padded.

    ``k_anch`` selects a K_BRACKET corner (band members); ``content_model`` supplies the import
    content for export books without physical fields; ``mirror_share`` overrides the Malaysia
    deduction share; ``k_shift`` shifts each book's measured server k (see ``_un_k_rows``).
    The per-book AI-window ledger lands in ``UN_LEDGER[tag]``.
    """
    ck = reexport_shares(res)
    danch, uanch = _k_anchors(k_anch)
    th = _un_thailand(res)
    import_c = _un_import_content(content_model)
    book_ledger = []
    book_res, book_th = res, th
    if history.ENABLED:
        hist = history.history()
        book_res = history.warm_res(res)
        if th is not None:
            book_th = pd.concat([th, hist[hist.rep_c.eq("Thailand")]], ignore_index=True)
    parts = [_un_leg_rows(book_res, book_th, ck, leg, danch, uanch, import_c, book_ledger, mirror_share,
                          k_shift)
             for leg in _un_legs()]
    export_res = history.thin_res(res, books.EU_MEMBERS) if history.ENABLED else res
    cols = ["dest", "code_hs6", "period", "value_usd"]
    parts.append(rx_readd_rows(export_res, hubs=["United States"])[cols])
    parts.append(hub_x_readd_rows(export_res, "United States", US_DX_FINISHED, flow="DX")[cols])
    parts.append(eu_x_readd_rows(export_res)[cols])
    parts.append(hub_x_readd_rows(export_res, "Malaysia", MY_FINISHED)[cols])
    for r in SGHK_HUBS:
        parts.append(hub_x_readd_rows(export_res, r, SGHK_FINISHED)[cols])
    for r in NEWLEG_NETTED:
        parts.append(hub_x_readd_rows(export_res, r, NEWLEG_FINISHED)[cols])
    fa = pd.concat(parts, ignore_index=True)
    fa["dest"] = fa["dest"].map(ledger.clean_thin)
    fa = fa.groupby(["dest", "code_hs6", "period"], as_index=False)["value_usd"].sum()
    if history.ENABLED:
        fa = history.pad_history(fa)
    L = pd.DataFrame(book_ledger)
    UN_LEDGER[tag] = L
    if tag == "headline":
        print(f"[netting] {len(L)} books under one rule; k anchors $/kg {danch}, $/unit {uanch}")
        print("  AI-window ledger ($B, counted codes): gross M | re-exports at 1 | own-made X | "
              "of which servers | servers at k | k | export fields | netted level")
        show = L.set_index("leg")[["gross_M", "reexports_at_1", "own_x_gross", "own_servers_gross",
                                    "own_servers_at_k", "k_servers", "export_fields", "netted_level"]]
        print("    " + show.round(2).to_string().replace("\n", "\n    "))
        t = ck.eu_rx_table
        if len(t):
            big = t[t["X_extraEU_B"] > 0.15].sort_values("X_extraEU_B", ascending=False)
            print("  EU member re-export shares of server exports (production floor, import cap):")
            print("    " + big[["X_B", "X_extraEU_B", "M_B", "prod_annual_MEUR", "rx_lo", "rx_cap", "rx"]]
                  .round(2).to_string().replace("\n", "\n    "))
        if len(ck.leg_rx_table):
            print("  Re-export shares of exports on non-EU books with a production statistic:")
            cols = ["X_B", "M_B", "prod_annual_local_M", "fx", "months", "rx_lo", "rx_cap", "rx"]
            print("    " + ck.leg_rx_table[cols].round(2).to_string().replace("\n", "\n    "))
        UN_LEG_RX.clear()
        if len(ck.leg_rx_table):
            UN_LEG_RX.update({leg: {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
                                    for k, v in r.items()}
                              for leg, r in ck.leg_rx_table.to_dict("index").items()})
    return fa


def taiwan_export_by_dest(res):
    """Taiwan's basket exports per destination x code x month, 2021 history included on the
    2021 reference (destinations without a canonical name keep their raw label in parentheses)."""
    x = res[(res["rep_c"] == "Taiwan") & (res["flow"] == "X") & res["code_hs6"].isin(codes.BASKET)].copy()
    x["dest"] = x["par_c"].where(x["par_c"].notna(), "(" + x["partner"].astype(str) + ")")
    cols = ["dest", "code_hs6", "period", "value_usd"]
    if history.ENABLED:
        h = history.taiwan_history().copy()
        h["dest"] = h.partner.map(lambda s: books.CANON.get(books.NORM(s), "(" + s + ")"))
        x = pd.concat([x[cols], h[cols]], ignore_index=True)
    return x.groupby(["dest", "code_hs6", "period"], as_index=False)["value_usd"].sum()


def residual_value(res, r_code, exclude=()):
    """Taiwan's exports to destinations without a book, as one residual bucket, each code lifted
    by ``r_code`` (importers' declared value per dollar of Taiwan's declared exports).
    ``exclude``: normalized destination names to leave out (the assembly nodes)."""
    reporters = set(books.CANON.values())
    r = taiwan_export_by_dest(res)
    r = r[~r["dest"].isin(reporters)]
    if exclude:
        norm = r["dest"].map(lambda s: books.NORM(str(s).strip("()")))
        r = r[~norm.isin(set(exclude))]
    r = _grp(r)
    r["value_usd"] = r.apply(lambda row: row["value_usd"] * r_code.get(row["code_hs6"], 1.0), axis=1)
    r = _tag(r, "(residual: no importer leg)")
    return history.pad_history(r) if history.ENABLED else r
