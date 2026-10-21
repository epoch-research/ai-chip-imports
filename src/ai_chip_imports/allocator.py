"""The allocator: Epoch's per-model chip sales distributed across destinations.

Epoch's Chip Sales data give units by model and quarter but no geography; the estimator gives
chip dollars by destination but no model identity. Per arrival quarter, the allocator combines
them in pools:

* ``cn_domestic`` -- China-fabricated accelerators (Huawei, Cambricon) never appear in the
  Taiwan-centred trade the basket observes; they go to China.
* ``cn_spec`` -- models Epoch flags as sold primarily for the Chinese market; they split
  between China and Hong Kong by their chip dollars, which those destinations then spend.
* ``row`` -- every other model, split across all destinations by their remaining chip dollars.
  Within the pool every eligible destination receives the same model mix, with three rules:
  captive accelerators (TPU, Trainium, Inferentia) go only where their operators run them
  (``assumptions/captive_geography.csv``); AMD and captive accelerators are withheld from
  China, Hong Kong and Macau (the observed diversion channel is Nvidia-shaped); and those
  destinations' budget is deflated by a dated diversion premium before it buys units at Epoch's
  prices (``PREMIUM_SERIES``). Each destination still spends its whole budget.

Destinations allocated less than one unit in a quarter drop out of that quarter. Allocations
conserve Epoch's units per model and quarter. Stocks are cumulative arrivals since 2022, with no
retirement.

Bands: every share set from the estimator (baseline x content combinations, k corners, Malaysia
mirror endpoints) is allocated, as are the allocation variants ``uniformmix`` (no eligibility
rules), ``captiveus`` (captives US-only) and ``nopremium``; each is combined with Epoch's own
5th/95th percentiles (perfectly correlated across models). The bands are assumption envelopes,
not probability intervals. A three-quarter-smoothed share set is reported as a timing
sensitivity outside the envelope.

Destinations with their own book are tier "measured"; the rest, reached through other books'
exports and the split of Taiwan's residual bucket, are tier "thin". An annex splits the EU total
among members by their gross extra-EU arrivals (indicative).
"""
import os
import re
import unicodedata

import numpy as np
import pandas as pd

from . import books, codes, epoch, exporter_content, history, ledger, netting, paths, timing

AI_LO = codes.AI_LO
RESIDUAL_DEST = "(residual: no importer leg)"
HEADLINE_DESTS = {"United States", "China", "European Union", "South Korea",
                  "Malaysia", "Singapore", "Hong Kong", "Thailand",
                  "United Kingdom", "Japan", "Indonesia", "India", "Mexico"}
HEADLINE_DESTS |= {books.COMTRADE_LEG_NAMES[k] for k in books.COMTRADE_LEGS}
CN_DESTS = ("China", "Hong Kong")            # the cn_spec pool's destinations

# Dated diversion premium on China, Hong Kong and Macau's row-pool budget, indexed by arrival
# quarter: customs dollars of diverted hardware include an intermediary margin over Epoch's
# sell-in price. Calibrated on dated price observations in China against Epoch's own model
# prices; no premium before the 2022-23 export controls took effect, and none on cn_spec models
# (licensed sales).
PREMIUM_SERIES = (
    # (from arrival quarter, factor, basis)
    ("2022Q1", 1.00, "before the export controls: standard A100/H100 sales to China were lawful"),
    ("2023Q4", 1.29, "8-GPU Hopper servers in 2024: +23% to +35% over four observations"),
    ("2025Q1", 1.13, "2025: Hopper servers +6%, B200 rack +20%; midpoint"),
    ("2026Q1", 1.20, "8x H200 server, January 2026: +19.7%"),
    ("2026Q2", 2.35, "enforcement crackdown: B300 +116%, DGX B300 +147%; midpoint of 2.2-2.5x"),
)
PREMIUM_DESTS = ("China", "Hong Kong", "Macau")

DEST_QBAR = 1.0               # units: a destination-quarter below this drops out of the quarter
# Evidence columns: noise floors of $2M a quarter and 10% of the destination's own baseline,
# so an empty or flat history never reads as precise.
EVIDENCE_ABS_FLOOR = 2e6
EVIDENCE_CV_FLOOR = 0.10

CAPTIVE_DESIGNERS = {"Google", "Amazon"}      # never sold on the merchant market
CONTROLLED_DESIGNERS = CAPTIVE_DESIGNERS | {"AMD"}
CONTROLLED_DESTS = {"China", "Hong Kong", "Macau"}
CAPTIVE_GEO_CSV = str(paths.ASSUMPTIONS_DIR / "captive_geography.csv")
US_DEST = "United States"
SMOOTH_Q = 3                                 # the smoothed-shares timing sensitivity
EU_SMOOTH_Q = 4                              # EU member annex: trailing mean, quarters
HEADLINE_NAME = "indexed|corridor"
NON_CI_VARIANTS = {"headline|smoothshares"}

THIN_EXPORTERS = ("United States", "Malaysia", "Singapore", "Hong Kong", "United Kingdom",
                  "Indonesia", "India", "Taiwan")        # + the EU members (their dispatches)
THIN_MARK = "(thin tier)"
ORIGIN_VN = {"vietnam", "viet nam"}
GRP_BUCKET = {"taiwan": "taiwan", "vietnam": "vietnam_assembly",
              "mexico": "mexico_assembly", "thailand": "thailand_assembly",
              "united states": "united_states", "china": "china_origin"}
_GEO_CACHE = {}


def premium_factor(q):
    """The diversion premium for arrival quarter ``q``: the latest PREMIUM_SERIES step at or
    before ``q``."""
    f = 1.0
    for q0, fac, _ in PREMIUM_SERIES:
        if q >= q0:
            f = fac
    return f


def _warn_label_collisions(labels):
    """Warn when two thin-tier labels differ only in accents or punctuation (e.g. Curacao and
    Curaçao): an exporter spelling ``ledger.THIN_ALIASES`` does not yet unify. Renames such as
    Turkey/Türkiye cannot be detected this way and need an alias entry."""

    def norm(s):
        a = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z]", "", a.lower()).replace("saint", "st")
    seen = {}
    for lab in labels:
        seen.setdefault(norm(lab), set()).add(lab)
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    if dups:
        print("  ** WARNING: thin-tier label near-duplicates -- extend ledger.THIN_ALIASES: "
              + "; ".join(" / ".join(sorted(v)) for v in dups.values()))


def residual_country_split(res, r_code, bindex, cmodel):
    """Per-country shares (quarter x country) that split Taiwan's residual bucket.

    Taiwan's own exports to each destination without a book (assembly nodes and bare chips
    excluded), lifted by ``r_code``, run through the same baseline and ledger as the estimator,
    weighted by the content of Taiwan's own export records. Quarters in which the bucket has no
    positive excess use the pooled shares of the whole window. Returns
    ``(shares, fallback quarters, excess frame)``.
    """
    reporters = set(books.CANON.values())
    r = netting.taiwan_export_by_dest(res)
    r = r[~r["dest"].isin(reporters)].copy()
    r = r[~r["code_hs6"].isin(codes.CHIPS)]
    norm = r["dest"].map(lambda s: books.NORM(str(s).strip("()")))
    r = r[~norm.isin(set(netting.ASSEMBLY_NODES) | {"world", "total"})]
    r["dest"] = norm.str.title().map(ledger.clean_thin)
    r = r.groupby(["dest", "code_hs6", "period"], as_index=False)["value_usd"].sum()
    r["value_usd"] = r["value_usd"] * r["code_hs6"].map(lambda c: r_code.get(c, 1.0))
    if history.ENABLED:
        r = history.pad_history(r)
    co = ledger.corridor_chiponly(r, "dest", baseline_index=bindex, historical=history.ENABLED)
    w = exporter_content.residual_weights(co, cmodel, res)
    co["chiponly"] = co.excess.to_numpy() * w
    m = co.groupby(["dest", "q"])["chiponly"].sum().unstack("dest").fillna(0.0)
    m = m.clip(lower=0.0)   # split keys must be nonnegative
    tot = m.sum(axis=1)
    pooled = m.sum(axis=0) / m.sum().sum()
    sh = m.div(tot.where(tot > 0), axis=0)
    fell_back = sh.index[tot <= 0].tolist()
    for q in fell_back:
        sh.loc[q] = pooled
    return sh, fell_back, co


def clean_D(D, resid_shares, carry=None):
    """One share set (chip dollars per destination x quarter) -> quarter x destination matrix:
    thin-tier labels harmonized, the residual bucket split by ``resid_shares``, negatives
    clipped. ``carry``: destination-quarters whose books stop before the quarter ends, which take
    the share they held in their last complete quarter (``carry_forward_shares``)."""
    assert list(D.index.names) == ["dest", "q"], D.index.names
    d = D.reset_index()
    d.columns = ["dest", "q", "usd"]
    thin = ~d["dest"].isin(HEADLINE_DESTS | {RESIDUAL_DEST})
    d.loc[thin, "dest"] = d.loc[thin, "dest"].map(ledger.clean_thin)
    m = d.groupby(["q", "dest"])["usd"].sum().unstack("dest").fillna(0.0)
    if RESIDUAL_DEST in m.columns:
        resid = m.pop(RESIDUAL_DEST)
        split = resid_shares.reindex(m.index).fillna(0.0).mul(resid, axis=0)
        m = m.add(split, fill_value=0.0)
    m = m.clip(lower=0.0)
    if carry:
        m = carry_forward_shares(m, carry)
    return m


def allocation_quarters(epoch_m, head_D, res, verbose=True):
    """The allocated arrival quarters: up to the last complete customs quarter, minus trailing
    quarters whose interpolated sales draw on an incomplete Epoch sales quarter. Leading
    incomplete quarters are the start of the sales history and are kept."""
    last_dq = head_D.index.get_level_values("q").max()
    frontier = res["period"].max()                          # e.g. "2026-04"
    fy, fm = int(frontier[:4]), int(frontier[5:7])
    qn = (fm - 1) // 3                                      # 0-based quarter of the frontier month
    last_cq = f"{fy}Q{fm // 3}" if fm % 3 == 0 else (f"{fy}Q{qn}" if qn else f"{fy - 1}Q4")
    all_q = sorted(epoch_m["q"].unique())
    quarters = [q for q in all_q if q <= min(last_dq, last_cq)]
    if "partial_sales_coverage" in epoch_m.columns:
        partial = epoch_m.groupby("q")["partial_sales_coverage"].any()
        while len(quarters) > 1 and bool(partial.get(quarters[-1], False)):
            quarters = quarters[:-1]
    dropped = [q for q in all_q if q not in set(quarters)]
    if dropped and verbose:
        print(f"[alloc] quarters dropped as incomplete: "
              f"{dropped} (trade frontier {frontier} -> last complete quarter {last_cq})")
    return quarters


def incomplete_dest_quarters(U, quarters):
    """{(dest, q)} for allocated quarters that end after a contributing book's last month: a
    measured destination's own book (for the EU, any member's) and, for the thin tier, any of
    its exporters (marked (THIN_MARK, q)). Those months are missing, not zero, so the quarter's
    dollars would understate the destination."""
    res = U["res"]
    last = books.book_frontiers(res)
    cm = U.get("cmodel")
    th = getattr(cm, "_extra_books", {}).get("Thailand") if cm is not None else None
    if th is not None and len(th):
        last["Thailand"] = min(th["period"].max(), res["period"].max())
    eu = [last[m] for m in books.EU_MEMBERS if m in last]
    thin = [last[r] for r in THIN_EXPORTERS if r in last] + eu
    out = set()
    for q in quarters:
        end = str(pd.Period(q, freq="Q").asfreq("M", "end"))
        for dname in HEADLINE_DESTS:
            b = min(eu) if (dname == "European Union" and eu) else last.get(dname)
            if b is not None and b < end:
                out.add((dname, q))
        if thin and min(thin) < end:
            out.add((THIN_MARK, q))
    if out:
        print("[alloc] books stop before the quarter ends -> shares carried forward: "
              + ", ".join(f"{d} {q} (book to {min(eu) if d == 'European Union' else last.get(d, '?')})"
                          for d, q in sorted(out)))
    return out


def carry_forward_shares(m, carry):
    """Give each incomplete destination-quarter the share it held in the latest quarter its books
    covered: with S the carried shares and T the complete destinations' dollars,
    x_d = s_d T / (1 - S). The complete destinations' dollars are untouched."""
    m = m.copy()
    thin_cols = [c for c in m.columns if c not in HEADLINE_DESTS]
    def incomplete(dname, q):
        return (dname, q) in carry or ((THIN_MARK, q) in carry and dname in thin_cols)
    done = []
    for q in sorted({q for _, q in carry if q in m.index}):
        inc = [c for c in m.columns if incomplete(c, q)]
        if not inc:
            continue
        shares = {}
        for c in inc:
            prev = [p for p in m.index if p < q and not incomplete(c, p) and m.loc[p].sum() > 0]
            shares[c] = float(m.loc[prev[-1], c] / m.loc[prev[-1]].sum()) if prev else 0.0
        comp = [c for c in m.columns if c not in inc]
        total_s, rest = sum(shares.values()), float(m.loc[q, comp].sum())
        if rest <= 0 or total_s >= 1.0:
            continue
        for c in inc:
            m.loc[q, c] = shares[c] * rest / (1.0 - total_s)
            done.append((c, q))
    m.attrs["carried_forward"] = done
    return m


def origin_mix(res, bindex=None, cmodel=None, cells=None):
    """Origin mix of each measured destination's chip dollars, over the AI window (a diagnostic).

    Buckets: Taiwan (fab-direct; entrepot transit does not change declared origin), Mexico,
    Vietnam and Thailand assembly, the US, China origin (non-China destinations only), and other.
    Lines the cell classifier measures are routed by their cells (each cell's excess over its
    own 2022 pace x its own content reading); the machine lines with origin slices by the
    slices; any remaining line by raw excess over a per-origin baseline at the line's content.
    With ``cells`` (the headline chip dollars per destination x code x quarter), every line is
    rescaled to the headline's own dollars, so only the within-line split comes from origins.
    EU weights use gross extra-EU member arrivals although part of them is netted as
    re-exports. Returns one row per destination x bucket: excess, chiponly, share.
    """
    mix_codes = [c for c in codes.BASKET if c not in codes.CHIPS]
    frames = []
    for rep in sorted(HEADLINE_DESTS - {"European Union", "Thailand"}):
        d = res[(res["rep_c"] == rep) & (res["flow"] == "M")
                & res["code_hs6"].isin(mix_codes)].copy()
        if rep == "China":
            d = d[~d["partner"].map(books.NORM).isin(netting.CN_EXCLUDED_ORIGINS)]
        d["dest"] = rep
        frames.append(d)
    if "Thailand" in HEADLINE_DESTS:
        # Thailand's book is read outside the loader; only its finished codes are retained.
        th = netting.thailand_books()
        if th is not None:
            d = th[(th["flow"] == "M") & th["code_hs6"].isin(netting.TH_FINISHED)].copy()
            d["par_c"] = d["partner"].map(books.canon)
            d["dest"] = "Thailand"
            frames.append(d)
    eu = res[res["rep_c"].isin(books.EU_MEMBERS) & (res["flow"] == "M")
             & res["code_hs6"].isin(mix_codes) & ~res["par_c"].isin(books.EU_MEMBERS)].copy()
    eu["dest"] = "European Union"
    frames.append(eu)
    m = pd.concat(frames, ignore_index=True)
    norm = m["partner"].map(books.NORM)
    m = m[~norm.isin({"world", "total"})]        # match the content model's evidence universe
    norm = m["partner"].map(books.NORM)
    m["bucket"] = np.select(
        [m["par_c"].eq("Taiwan"), norm.eq("mexico"), norm.isin(ORIGIN_VN),
         norm.eq("thailand"), m["par_c"].eq("United States"), m["par_c"].eq("China")],
        ["taiwan", "mexico_assembly", "vietnam_assembly", "thailand_assembly",
         "united_states", "china_origin"], default="other")
    parts = []
    sliced = set()
    if cmodel is not None and getattr(cmodel, "_cellcls_frame", None) is not None:
        # the classifier's cells; a cell with no reading takes the line's content
        cf = cmodel._cellcls_frame
        cf = cf[(cf["q"] >= AI_LO) & cf["dest"].isin(set(m["dest"]))].copy()
        # Japan's memory/remainder cells carry a product label after the origin
        origin = cf["cell"].astype(str).str.replace(r"\|(memory|remainder)$", "", regex=True).str.split("|").str[-1]
        onorm, ocan = origin.map(books.NORM), origin.map(books.canon)
        cf["bucket"] = np.select(
            [ocan.eq("Taiwan"), onorm.eq("mexico"), onorm.isin(ORIGIN_VN), onorm.eq("thailand"),
             ocan.eq("United States"), ocan.eq("China")],
            ["taiwan", "mexico_assembly", "vietnam_assembly", "thailand_assembly",
             "united_states", "china_origin"], default="other")
        cell_c = cmodel.lookup(cf["dest"], cf["code_hs6"], cf["q"],
                               cf["code_hs6"].map(codes.CHIP_CONTENT))
        cf["excess"] = cf["excess"].clip(lower=0.0)
        cf["chiponly"] = cf["excess"] * cf["c"].where(cf["c"].notna(), cell_c)
        sliced |= set(zip(cf["dest"], cf["code_hs6"]))
        parts.append(cf[["dest", "code_hs6", "bucket", "q", "excess", "chiponly"]])
    if cmodel is not None:
        # the origin-group slices, each at its own content (the line's where it has none)
        sl = cmodel.evidence
        sl = sl[(sl["q"] >= AI_LO) & sl["dest"].isin(set(m["dest"]))].copy()
        if sliced:                                   # cells take precedence
            sl = sl[~pd.MultiIndex.from_arrays([sl["dest"], sl["code_hs6"]]).isin(sliced)]
        sl["bucket"] = sl["grp"].map(GRP_BUCKET).fillna("other")
        cell_c = cmodel.lookup(sl["dest"], sl["code_hs6"], sl["q"],
                               sl["code_hs6"].map(codes.CHIP_CONTENT))
        sl["chiponly"] = sl["excess"] * sl["c_grp"].where(sl["c_grp"].notna(), cell_c)
        sliced |= set(zip(sl["dest"], sl["code_hs6"]))
        parts.append(sl[["dest", "code_hs6", "bucket", "q", "excess", "chiponly"]])
    keep = ~pd.MultiIndex.from_arrays([m["dest"], m["code_hs6"]]).isin(sliced)
    g = (m[keep].groupby(["dest", "code_hs6", "bucket", "period"], as_index=False)
         ["value_usd"].sum())
    b = (g[g["period"].str[:4] == "2022"].groupby(["dest", "code_hs6", "bucket"])["value_usd"]
         .mean().rename("mean2022").reset_index())
    g = g.merge(b, on=["dest", "code_hs6", "bucket"], how="left")
    g["mean2022"] = g["mean2022"].fillna(0.0)
    gf = (1.0 + codes.NONAI_GROWTH["systems"]) ** g["period"].map(codes.years_since_mid2022)
    if bindex is not None:
        g = g.reset_index(drop=True)
        gf = bindex.lookup(g["dest"], g["code_hs6"], g["period"], gf)
    g["excess"] = (g["value_usd"] - g["mean2022"] * gf).clip(lower=0.0)
    g["q"] = g["period"].map(codes.quarter)
    g = g[g["q"] >= AI_LO]
    fallback = g["code_hs6"].map(codes.CHIP_CONTENT)
    c = cmodel.lookup(g["dest"], g["code_hs6"], g["q"], fallback) if cmodel is not None \
        else fallback
    g["chiponly"] = g["excess"] * c
    parts.append(g.groupby(["dest", "code_hs6", "bucket", "q"], as_index=False)
                 [["excess", "chiponly"]].sum())
    allp = pd.concat(parts, ignore_index=True)
    if cells is not None:
        # rescale every line-quarter to the headline's chip dollars; buckets keep their
        # within-line weights (raw-excess weights where the line has no measured weight)
        key = ["dest", "code_hs6", "q"]
        allp["cell_usd"] = pd.MultiIndex.from_frame(allp[key]).map(cells)
        allp = allp[allp["cell_usd"].notna()]
        wtot = allp.groupby(key)["chiponly"].transform("sum")
        etot = allp.groupby(key)["excess"].transform("sum")
        wsh = (allp["chiponly"] / wtot.where(wtot > 0)).fillna(
            allp["excess"] / etot.where(etot > 0))
        allp["chiponly"] = allp["cell_usd"] * wsh.fillna(0.0)
        routed = allp["chiponly"].sum()
        in_mix = (cells.index.get_level_values("dest").isin(set(m["dest"]))
                  & (cells.index.get_level_values("q") >= AI_LO))
        total = float(cells[in_mix].sum())
        if total > 0 and abs(routed - total) / total > 0.02:
            print(f"  [origin-mix] NB: {1 - routed / total:.1%} of mix-dest headline $ had "
                  f"no origin evidence and is unrouted")
    out = allp.groupby(["dest", "bucket"], as_index=False)[["excess", "chiponly"]].sum()
    out["share"] = out["chiponly"] / out.groupby("dest")["chiponly"].transform("sum")
    return out


def eu_member_shares(res):
    """Indicative split of the EU total among members (quarter x member).

    Members' gross extra-EU imports through the parametric baseline and ledger, at flat content,
    as a trailing four-quarter mean. Gross on purpose: netting does not work at member level
    (assembly members' extra-EU exports exceed their imports), so the annex shows where extra-EU
    arrivals land first. The Netherlands, a distribution hub, is an upper bound.
    """
    counted = [c for c in codes.BASKET if c not in codes.CHIPS]
    d = res[res["rep_c"].isin(books.EU_MEMBERS) & (res["flow"] == "M")
            & res["code_hs6"].isin(counted) & ~res["par_c"].isin(books.EU_MEMBERS)]
    g = d.groupby(["rep_c", "code_hs6", "period"], as_index=False)["value_usd"].sum()
    co = ledger.corridor_chiponly(g, "rep_c")
    m = co.groupby(["dest", "q"])["chiponly"].sum().unstack("dest").fillna(0.0)
    m = m.clip(lower=0.0)
    m = m.rolling(EU_SMOOTH_Q, min_periods=1).mean()
    return m.div(m.sum(axis=1).where(m.sum(axis=1) > 0), axis=0).fillna(0.0)


def destination_evidence(coC, resid_co, pre_hi="2023Q2"):
    """How far each destination's excess sits above its own noise (reported, never enforced).

    Per destination (the headline frame plus the residual split, so countries reached only
    through Taiwan's exports are covered): sigma is the standard deviation of the quarterly gaps
    up to ``pre_hi``, floored at EVIDENCE_ABS_FLOOR and at EVIDENCE_CV_FLOOR x the mean quarterly
    baseline. ``excess_z`` (flows) = the quarter's excess / sigma; ``excess_z_cum`` (stocks) =
    cumulative excess / (sigma x sqrt(quarters so far)), using data up to the quarter only;
    ``evidence`` labels it weak (<1), moderate (1-3) or strong (>=3).
    """
    frames = []
    for co in (coC, resid_co):
        if co is None or len(co) == 0:
            continue
        c = co[["dest", "q", "value_usd", "baseline", "excess"]].copy()
        c["dest"] = c["dest"].map(ledger.clean_thin)
        frames.append(c[c["dest"] != RESIDUAL_DEST])
    allc = pd.concat(frames, ignore_index=True)
    g = (allc.assign(gap=allc["value_usd"] - allc["baseline"])
         .groupby(["dest", "q"])[["gap", "excess", "baseline"]].sum())
    gap = g["gap"].unstack("q").fillna(0.0).sort_index(axis=1)
    exc = g["excess"].unstack("q").fillna(0.0).sort_index(axis=1)
    base = g["baseline"].unstack("q").fillna(0.0).sort_index(axis=1)
    pre = [q for q in gap.columns if q <= pre_hi]
    sig = gap[pre].std(axis=1) if len(pre) >= 3 else pd.Series(np.nan, index=gap.index)
    sigma = pd.concat([sig, EVIDENCE_CV_FLOOR * base.mean(axis=1),
                       pd.Series(EVIDENCE_ABS_FLOOR, index=gap.index)], axis=1).max(axis=1)
    zq = exc.div(sigma, axis=0)
    zc = exc.cumsum(axis=1).div(sigma, axis=0) / np.sqrt(np.arange(1, exc.shape[1] + 1))
    per_q = zq.stack().rename("excess_z").reset_index()
    cum = zc.stack().rename("excess_z_cum").reset_index()
    cum["evidence"] = pd.cut(cum["excess_z_cum"], [-np.inf, 1.0, 3.0, np.inf],
                             labels=["weak", "moderate", "strong"]).astype(str)
    last = exc.columns[-1]
    cnt = cum[cum["q"] == last]["evidence"].value_counts()
    print(f"  [evidence] noise-scaled excess columns for {len(sigma)} destinations (sigma floors "
          f"${EVIDENCE_ABS_FLOOR/1e6:.0f}M/quarter, {EVIDENCE_CV_FLOOR:.0%} of own baseline); at {last}: "
          + ", ".join(f"{k} {int(v)}" for k, v in cnt.items()))
    return per_q, cum


def load_captive_geography(path=CAPTIVE_GEO_CSV):
    """{model: {destination: eligible-from quarter or None}} from the captive geography table
    (rows with eligible == 1; rows with 0 document deliberate exclusions)."""
    if path in _GEO_CACHE:
        return _GEO_CACHE[path]
    if not os.path.exists(path):
        raise SystemExit(f"[captive-geo] missing {path}")
    t = pd.read_csv(path, dtype=str).fillna("")
    geo = {}
    for r in t.itertuples():
        if str(r.eligible).strip() != "1":
            continue
        geo.setdefault(r.model.strip(), {})[r.destination.strip()] = (r.eligible_from.strip() or None)
    _GEO_CACHE[path] = geo
    return geo


def _ipf_shares(cost, mask, target, iters=5000, tol=1e-12):
    """Per-model destination shares under a 0/1 eligibility mask, by iterative proportional
    fitting: rows (each model's cost) exact, columns (each destination's budget) fitted. Returns
    (shares, max relative column error)."""
    Y = cost[:, None] * mask * target[None, :]
    for _ in range(iters):
        rs = Y.sum(1)
        Y = Y * np.where(rs > 0, cost / np.where(rs > 0, rs, 1.0), 0.0)[:, None]
        cs = Y.sum(0)
        Y = Y * np.where(cs > 0, target / np.where(cs > 0, cs, 1.0), 0.0)[None, :]
        rs = Y.sum(1)
        if float(np.max(np.abs(rs - cost) / np.where(cost > 0, cost, 1.0))) < tol:
            break
    rs = Y.sum(1)
    Y = Y * np.where(rs > 0, cost / np.where(rs > 0, rs, 1.0), 0.0)[:, None]
    colerr = float(np.max(np.abs(Y.sum(0) - target) / np.where(target > 0, target, 1.0)))
    return Y / np.where(cost > 0, cost, 1.0)[:, None], colerr


def _geo_eligible(designer, model, dst, q, geo, captive_mode, restr_designers, restr_dests, missing):
    """1.0 if ``model`` may be allocated to ``dst`` in quarter ``q``, else 0.0."""
    if designer in CAPTIVE_DESIGNERS:
        if captive_mode == "us_only":
            return 1.0 if dst == US_DEST else 0.0
        elig = geo.get(model)
        if elig is None:                      # a captive model the table does not list: US only
            missing.add(model)
            return 1.0 if dst == US_DEST else 0.0
        if dst not in elig:
            return 0.0
        since = elig[dst]
        return 1.0 if (since is None or q >= since) else 0.0
    if designer in restr_designers and dst in restr_dests:
        return 0.0
    return 1.0


def allocate(epoch_m, Dm, quarters, eligibility=None, premium=None, captive=None):
    """Allocate Epoch's per-model sales to destinations for one share set.

    ``epoch_m``: per-model sales by arrival quarter (``epoch.models_by_arrival_quarter``).
    ``Dm``: quarter x destination chip dollars (``clean_D``). Per quarter:

    * ``cn_domestic`` models (China-fabricated) go to China;
    * ``cn_spec`` models (sold primarily for the Chinese market) split between China and Hong
      Kong by their dollars, and the dollars they absorb are netted out of those destinations;
    * ``row`` models split across all destinations by their remaining dollars. China, Hong Kong
      and Macau's budget is first deflated by the diversion premium (``premium``). Restricted
      designers (``eligibility``, default AMD and the captive designers) are withheld from the
      controlled destinations, and captive models (``captive``: "table" = operator geography,
      "us_only" = the strict corner) go only where their operators run them; the per-model
      eligibility masks are fitted jointly so every destination still spends its whole budget
      (a quarter where the fit is infeasible falls back to the closed form for the
      eligibility rule alone).

    A destination allocated fewer than DEST_QBAR units in a quarter is removed from that
    quarter and its dollars renormalize onto the rest. Allocations conserve Epoch's units per
    model and quarter (asserted). Returns one row per destination x quarter x model.
    """
    em = epoch_m[epoch_m["q"].isin(set(quarters))]
    if captive is None:
        captive = "table"
    geo = load_captive_geography() if captive == "table" else {}
    geo_missing, geo_fallback = set(), {}
    if eligibility is None:
        eligibility = (CONTROLLED_DESIGNERS, CONTROLLED_DESTS)
    restr_designers, restr_dests = eligibility
    if premium is None:
        premium = True
    restricted_hits, prem_removed = {}, {}
    last_dq = Dm.index.max()
    rows = []
    screened = {}

    def _one_q(q, eq, D, provisional, hits, prem):
        qrows = []
        # CN pool: split across CN_DESTS by their D shares; net the absorbed $ out of D'
        cn = eq[eq["pool"] == "cn_spec"]
        cn_tot = float(sum(D.get(c, 0.0) for c in CN_DESTS))
        cn_sh = {c: (D.get(c, 0.0) / cn_tot if cn_tot > 0 else (1.0 if c == "China" else 0.0))
                 for c in CN_DESTS}
        cn_usd = float((cn["units"] * cn["asp"]).fillna(0.0).sum())
        Dp = D.copy()
        for c in CN_DESTS:
            if c in Dp.index:
                Dp[c] = max(0.0, Dp[c] - cn_usd * cn_sh[c])
        # the diverted budget is in customs dollars: deflate it to sell-in dollars before it
        # buys units at Epoch's prices (cn_spec above is untouched)
        pf = premium_factor(q) if premium else 1.0
        if pf != 1.0:
            for c in PREMIUM_DESTS:
                if c in Dp.index and Dp[c] > 0:
                    prem[q] = prem.get(q, 0.0) + Dp[c] * (1.0 - 1.0 / pf)
                    Dp[c] = Dp[c] / pf
        row_sh = Dp / Dp.sum() if Dp.sum() > 0 else Dp * 0.0

        for r in cn.itertuples():
            for c in CN_DESTS:
                if cn_sh[c] > 0:
                    qrows.append((c, q, r.model, r.designer, "cn_spec", cn_sh[c],
                                  r.units * cn_sh[c], r.h100e * cn_sh[c],
                                  r.units * cn_sh[c] * r.asp, provisional))
        # row pool. Restricted designers skip the restricted destinations; the restricted dests'
        # claim on the ELIGIBLE models is scaled up so that every destination still deploys its
        # WHOLE dollar budget s_d x C (C = total row cost). Closed form, both margins exact:
        #   restricted models (cost C_R): y_d = s_d/(1-S_K) for d outside K, 0 inside
        #   eligible models  (cost C_N): x_d = s_d x C/C_N            for d in K
        #                                x_d = s_d x (C - C_R/(1-S_K))/C_N  otherwise
        # (sum x_d = (C-C_R)/C_N = 1; K = restricted dests, S_K = their dollar share.) The
        # feasibility condition is C_R/C <= 1-S_K -- the models the controlled destinations
        # cannot hold must fit inside the other destinations' budgets; violations fall back to
        # the unrestricted mix for that quarter, loudly.
        row_models = eq[eq["pool"] == "row"]
        x_sh = y_sh = None
        geo_shares = None
        if captive and restr_designers and len(row_models):
            # per-model eligibility fitted jointly so every destination keeps its whole budget
            dests = list(row_sh.index)
            s_arr = row_sh.to_numpy(dtype=float)
            cost = (row_models["units"] * row_models["asp"]).fillna(0.0).to_numpy(dtype=float)
            mask = np.array([[_geo_eligible(des, mod, d_, q, geo, captive, restr_designers,
                                            restr_dests, geo_missing) for d_ in dests]
                             for des, mod in zip(row_models["designer"], row_models["model"])],
                            dtype=float)
            SH, colerr = _ipf_shares(cost, mask, s_arr * cost.sum())
            if colerr < 1e-6 or cost.sum() == 0:
                geo_shares = []
                for i in range(len(row_models)):
                    if cost[i] > 0:
                        geo_shares.append(pd.Series(SH[i], index=dests))
                    else:
                        m_ = mask[i] * s_arr
                        geo_shares.append(pd.Series(m_ / m_.sum() if m_.sum() > 0 else m_, index=dests))
            else:
                geo_fallback[q] = colerr
        if geo_shares is None and restr_designers and restr_dests and len(row_models):
            isr = row_models["designer"].isin(restr_designers).to_numpy()
            cost = (row_models["units"] * row_models["asp"]).fillna(0.0).to_numpy()
            C_R, C_N = float(cost[isr].sum()), float(cost[~isr].sum())
            C = C_R + C_N
            K = [c for c in row_sh.index if c in restr_dests]
            S_K = float(row_sh.reindex(K).fillna(0.0).sum())
            if C_N > 0 and (1.0 - S_K) > 1e-9 and C_R / C <= (1.0 - S_K):
                y = row_sh.copy()
                y[K] = 0.0
                y_sh = y / y.sum() if y.sum() > 0 else y
                x_sh = row_sh * ((C - C_R / (1.0 - S_K)) / C_N)
                x_sh[K] = row_sh.reindex(K).fillna(0.0) * (C / C_N)
                x_sh = x_sh.clip(lower=0.0)
                x_sh = x_sh / x_sh.sum() if x_sh.sum() > 0 else x_sh
            else:
                hits.setdefault("(infeasible -> unrestricted)", set()).add(q)
        for i, r in enumerate(row_models.itertuples()):
            if geo_shares is not None:
                sh = geo_shares[i]
                if r.designer in restr_designers:
                    hits.setdefault(r.designer, set()).add(q)
            else:
                sh = row_sh
                if x_sh is not None:
                    sh = y_sh if r.designer in restr_designers else x_sh
                    if r.designer in restr_designers:
                        hits.setdefault(r.designer, set()).add(q)
            for c, s in sh[sh > 0].items():
                qrows.append((c, q, r.model, r.designer, "row", s,
                              r.units * s, r.h100e * s, r.units * s * r.asp, provisional))
        for r in eq[eq["pool"] == "cn_domestic"].itertuples():
            qrows.append(("China", q, r.model, r.designer, "cn_domestic", 1.0,
                          r.units, r.h100e, r.units * r.asp, provisional))
        return qrows

    for q in quarters:
        eq = em[em["q"] == q]
        if eq.empty:
            continue
        provisional = q >= last_dq or bool(eq.incomplete.any())
        D = Dm.loc[min(q, last_dq)]
        # a trial pass measures each destination's unit claim; destinations below the bar are
        # zeroed and the quarter re-allocated (survivors only scale up, so one pass suffices)
        tot = {}
        for r_ in _one_q(q, eq, D, provisional, {}, {}):
            tot[r_[0]] = tot.get(r_[0], 0.0) + r_[6]
        drop = {d for d, u in tot.items() if u < DEST_QBAR}
        if drop:
            D = D.copy()
            D[D.index.isin(drop)] = 0.0
            screened[q] = (len(drop), sum(tot[d] for d in drop))
        rows.extend(_one_q(q, eq, D, provisional, restricted_hits, prem_removed))
    out = pd.DataFrame(rows, columns=["dest", "q", "model", "designer", "pool", "dest_share",
                                      "units", "h100e", "usd", "provisional"])
    # conservation: the allocation must reproduce Epoch's global numbers exactly
    ref = em.set_index(["model", "q"])["units"]
    chk = out.groupby(["model", "q"])["units"].sum().reindex(ref.index).fillna(0.0)
    rel = float(((chk - ref).abs() / ref.where(ref > 0, 1.0)).max())
    assert rel < 1e-6, f"conservation broken: max rel err {rel:.2e}"
    for des, qs_ in sorted(restricted_hits.items()):
        if captive and des in CAPTIVE_DESIGNERS:
            print(f"  [captive-geo] {des}: allocated by operator geography "
                  f"({'strict US-only' if captive == 'us_only' else os.path.basename(CAPTIVE_GEO_CSV)}) "
                  f"in {len(qs_)} quarters")
        else:
            print(f"  [eligibility] {des}: withheld from {sorted(restr_dests)} in "
                  f"{len(qs_)} quarters (their shares renormalized onto eligible destinations)")
    if geo_missing:
        print(f"  [captive-geo] WARNING: captive models absent from the table, allocated US-only: "
              f"{sorted(geo_missing)}")
    if geo_fallback:
        print(f"  [captive-geo] WARNING: infeasible joint fit in {sorted(geo_fallback)} -> "
              f"closed-form eligibility rule for those quarters")
    if prem_removed:
        qs_ = sorted(prem_removed)
        ai_rm = sum(v for k, v in prem_removed.items() if k >= AI_LO)
        print(f"  [premium] diversion premium applied to {sorted(PREMIUM_DESTS)} over "
              f"{qs_[0]}..{qs_[-1]} (dated series "
              + ", ".join(f"{q0}+ {fac:.2f}x" for q0, fac, _ in PREMIUM_SERIES if fac != 1.0)
              + f"): ${ai_rm/1e9:.1f}B of AI-window row-pool budget treated "
              f"as intermediary margin rather than Epoch sell-in value")
    if screened:
        ndq = sum(n for n, _ in screened.values())
        mv = sum(u for _, u in screened.values())
        print(f"  [dest-bar] {DEST_QBAR:g}-unit/quarter screen: {ndq} destination-quarters "
              f"below the bar removed ({mv:,.0f} units renormalized onto each quarter's "
              f"survivors)")
    return out


def run_variants(epoch_m, collectors, resid_shares, quarters, non_ci=frozenset(), carry=None,
                 variant_residuals=None):
    """Allocate every share set. Returns ({variant: flows}, the long frame of every variant's
    shares). ``non_ci`` variants are reported (in_ci=False) but stay out of the band envelope;
    ``carry`` lists destination-quarters whose books stop early."""
    flows, share_rows = {}, []
    for vname, D in collectors.items():
        Dm = clean_D(D, (variant_residuals or {}).get(vname, resid_shares), carry)
        flows[vname] = allocate(epoch_m, Dm, quarters)
        sh = Dm.div(Dm.sum(axis=1).where(Dm.sum(axis=1) > 0), axis=0)
        long = Dm.stack().rename("D_usd").to_frame()
        long["share"] = sh.stack()
        long = long.reset_index()
        long.columns = ["q", "dest", "D_usd", "share"]
        long.insert(0, "variant", vname)
        long.insert(1, "in_ci", vname not in non_ci)
        share_rows.append(long[long["D_usd"] > 0])
    return flows, pd.concat(share_rows, ignore_index=True)


def band_frames(epoch_m, flows, headline_name, non_ci=frozenset()):
    """Flows and cumulative stocks with lo/hi bands, plus coherent country bands.

    Per destination x quarter x model, lo is the minimum over the band variants (``non_ci``
    excluded) of the variant's value x Epoch's 5th-percentile factor for that model and quarter,
    hi the maximum at the 95th. Stocks cumulate each variant first, then take the envelope.
    Stocks are a dense panel: every destination x model pair runs over the whole quarter axis.
    The per-cell bands are not additive across models; ``country_bands`` gives destination
    totals.
    """
    fac = epoch_m.set_index(["model", "q"])
    f_lo = (fac["units_lo"] / fac["units"].where(fac["units"] > 0)).fillna(1.0)
    f_hi = (fac["units_hi"] / fac["units"].where(fac["units"] > 0)).fillna(1.0)
    h_lo = (fac["h100e_lo"] / fac["h100e"].where(fac["h100e"] > 0)).fillna(1.0)
    h_hi = (fac["h100e_hi"] / fac["h100e"].where(fac["h100e"] > 0)).fillna(1.0)

    qs = sorted({q for fl in flows.values() for q in fl["q"].unique()})
    meta = epoch_m.drop_duplicates("model").set_index("model")[["designer", "pool"]]
    prov_q = pd.concat(flows.values()).groupby("q")["provisional"].any()
    MEAS = ["units", "units_lo", "units_hi", "h100e", "h100e_lo", "h100e_hi", "usd"]

    per_var_flow, per_var_stock = {}, {}
    for v, fl in flows.items():
        d = fl.set_index(["dest", "q", "model"])
        qkey = d.index.get_level_values("q")
        mi = pd.MultiIndex.from_arrays([d.index.get_level_values("model"), qkey])
        d = d.assign(units_lo=d["units"] * f_lo.reindex(mi).to_numpy(),
                     units_hi=d["units"] * f_hi.reindex(mi).to_numpy(),
                     h100e_lo=d["h100e"] * h_lo.reindex(mi).to_numpy(),
                     h100e_hi=d["h100e"] * h_hi.reindex(mi).to_numpy())
        per_var_flow[v] = d
        s = d.reset_index()
        frames = {}
        for c in MEAS:
            w = s.pivot_table(index=["dest", "model"], columns="q", values=c, aggfunc="sum")
            frames[c] = w.reindex(columns=qs).fillna(0.0).cumsum(axis=1).stack()
        per_var_stock[v] = pd.DataFrame(frames)         # index (dest, model, q), dense

    def envelope(per_var, cols):
        # The union of all variants' cells: a cell only some variant reaches reads 0 in the
        # headline's value columns while the band spans it.
        ci_vars = {v: d for v, d in per_var.items() if v not in non_ci}
        allv = pd.concat({v: d[cols] for v, d in ci_vars.items()}, axis=1)
        head = per_var[headline_name].reindex(allv.index)
        added = allv.index.difference(per_var[headline_name].index)
        if len(added):
            num = head.select_dtypes(include=[np.number]).columns
            head.loc[added, num] = head.loc[added, num].fillna(0.0)
        for c in cols:
            sub = allv.xs(c, axis=1, level=1).fillna(0.0)
            head[c] = sub.min(axis=1) if c.endswith("_lo") else sub.max(axis=1)
        return head.reset_index()

    band_cols = ["units_lo", "units_hi", "h100e_lo", "h100e_hi"]
    flow = envelope(per_var_flow, band_cols)
    # union-added rows carry no designer/pool/provisional from the headline
    flow["designer"] = flow["model"].map(meta["designer"])
    flow["pool"] = flow["pool"].where(flow["pool"].notna(), flow["model"].map(meta["pool"]))
    flow["provisional"] = flow["q"].map(prov_q).fillna(False).astype(bool)
    stock = envelope(per_var_stock, band_cols)
    stock["designer"] = stock["model"].map(meta["designer"])
    stock["pool"] = stock["model"].map(meta["pool"])
    stock["provisional"] = stock["q"].map(prov_q).fillna(False)
    stock = stock.rename(columns={c: c.replace("units", "units_cum").replace("h100e", "h100e_cum")
                                  for c in ["units", "units_lo", "units_hi",
                                            "h100e", "h100e_lo", "h100e_hi"]}
                         ).rename(columns={"usd": "usd_cum"})
    return flow, stock, country_bands(per_var_flow, per_var_stock, headline_name, non_ci)


def country_bands(per_var_flow, per_var_stock, headline_name, non_ci=frozenset()):
    """Destination bands: each variant's destination total (summed over models within the
    variant, for flows and cumulative stocks) at Epoch's 5th/95th percentiles, then the min/max
    across variants. Summing the per-model extremes instead would mix variants across models.
    World rows sum destinations within each variant, reproducing Epoch's own percentiles."""
    meas = ["units", "units_lo", "units_hi", "h100e", "h100e_lo", "h100e_hi"]

    def totals(per_var):
        out = {}
        for v, d in per_var.items():
            t = d[meas].groupby(level=["dest", "q"]).sum()
            w = t.groupby(level="q").sum()
            w.index = pd.MultiIndex.from_product([["World"], w.index], names=["dest", "q"])
            out[v] = pd.concat([t, w])
        return out

    parts = []
    for per_var, suffix in ((per_var_flow, ""), (per_var_stock, "_cum")):
        tot = totals(per_var)
        ci = {v: t for v, t in tot.items() if v not in non_ci}
        allv = pd.concat(ci, axis=1)
        head = tot[headline_name].reindex(allv.index).fillna(0.0)
        f = pd.DataFrame(index=allv.index)
        for m in ("units", "h100e"):
            f[m + suffix] = head[m]
            f[m + suffix + "_lo"] = allv.xs(m + "_lo", axis=1, level=1).fillna(0.0).min(axis=1)
            f[m + suffix + "_hi"] = allv.xs(m + "_hi", axis=1, level=1).fillna(0.0).max(axis=1)
        parts.append(f)
    out = parts[0].join(parts[1], how="outer").fillna(0.0).reset_index()
    return out.sort_values(["q", "dest"]).reset_index(drop=True)


def _tier(dest_name):
    return "measured" if dest_name in HEADLINE_DESTS else "thin"


def report(stock, flow, epoch_m, fell_back, eu_sh, quarters, bands):
    """Print the allocation's summary: stocks through the last quarter, China's pools, checks."""
    last = max(stock["q"])
    print(f"\n=== Allocation: arrival quarters {quarters[0]}..{quarters[-1]} "
          f"(calendar customs quarters; sales spread within quarters, {timing.LAG_MONTHS}-month lag) ===")
    s = stock[stock["q"] == last]
    tab = (s.groupby("dest")
           .agg(units_M=("units_cum", "sum"), h100e_M=("h100e_cum", "sum"))
           .sort_values("h100e_M", ascending=False))
    bl = bands[bands["q"] == last].set_index("dest")
    tab["h100e_lo"] = bl["h100e_cum_lo"].reindex(tab.index)
    tab["h100e_hi"] = bl["h100e_cum_hi"].reindex(tab.index)
    tab = tab / 1e6
    tab["tier"] = [_tier(d) for d in tab.index]
    print(f"\n--- cumulative arrivals through {last} (millions; band = assumption envelope) ---")
    print(tab.head(16).to_string(formatters={c: "{:.3f}".format
                                             for c in ("units_M", "h100e_M", "h100e_lo", "h100e_hi")}))
    thin_tot = tab[tab["tier"] == "thin"][["units_M", "h100e_M"]].sum()
    print(f"  thin tier: {thin_tot['units_M']:.2f}M units / {thin_tot['h100e_M']:.2f}M H100e across "
          f"{int((tab['tier'] == 'thin').sum())} countries")
    cn = stock[(stock["dest"] == "China") & (stock["q"] == last)].groupby("pool")[
        ["units_cum", "h100e_cum"]].sum() / 1e6
    print(f"\n--- China by pool through {last} (millions) ---")
    print(cn.to_string(formatters={c: "{:.2f}".format for c in cn.columns}))
    ep_dom = epoch_m[epoch_m["pool"] == "cn_domestic"].groupby("q")["units"].sum()
    al_dom = flow[flow["pool"] == "cn_domestic"].groupby("q")["units"].sum()
    assert np.allclose(ep_dom.reindex(al_dom.index).fillna(0.0), al_dom), \
        "cn_domestic allocation must equal Epoch's China-fabricated rows"
    print(f"  residual split fallback (pooled shares) quarters: {fell_back or 'none'}")
    nl = eu_sh["Netherlands"].iloc[-4:].mean() if "Netherlands" in eu_sh.columns else np.nan
    print(f"  EU annex: {eu_sh.shape[1]} members; Netherlands trailing-year share {nl:.0%} "
          f"(upper bound: a distribution hub)")


def run(U, est):
    """Allocate every share set from the estimator (``estimator.run``) and write the outputs."""
    res, cm = U["res"], U["cmodel"]
    coC = est["coC"]
    collectors = {f"{b}|{c}": s for (b, c), s in est["collectors"].items()}
    head_D = collectors[HEADLINE_NAME]
    for vtag, s in est["variants"].items():
        collectors[f"{vtag}|corridor"] = s
    # the Malaysia endpoints carry their own composite index into the residual split too
    variant_residuals = {
        f"{vtag}|corridor": residual_country_split(res, U["r_resid"], bindex=bi, cmodel=cm)[0]
        for vtag, bi in est["mirror_indices"].items()}
    # a timing sensitivity, outside the band: shares smoothed over three quarters
    sm = (head_D.unstack("dest").fillna(0.0)
          .rolling(SMOOTH_Q, center=True, min_periods=1).mean()
          .stack().swaplevel().sort_index())
    sm.index = sm.index.set_names(["dest", "q"])
    collectors["headline|smoothshares"] = sm
    print(f"[alloc] {len(collectors)} share sets, {len(collectors) - len(NON_CI_VARIANTS)} in the band")

    epoch_m = epoch.models_by_arrival_quarter()
    quarters = allocation_quarters(epoch_m, head_D, res)
    carry = incomplete_dest_quarters(U, quarters)
    resid_shares, fell_back, resid_co = residual_country_split(res, U["r_resid"],
                                                              bindex=U["bindex"], cmodel=cm)
    imputation = exporter_content.country_exposure(coC, resid_co, cm, res, resid_shares)
    flows, shares_long = run_variants(epoch_m, collectors, resid_shares, quarters,
                                      non_ci=NON_CI_VARIANTS, carry=carry,
                                      variant_residuals=variant_residuals)
    D = clean_D(head_D, resid_shares, carry)
    flows[f"{HEADLINE_NAME}|uniformmix"] = allocate(epoch_m, D, quarters, eligibility=(set(), set()))
    flows[f"{HEADLINE_NAME}|captiveus"] = allocate(epoch_m, D, quarters, captive="us_only")
    flows[f"{HEADLINE_NAME}|nopremium"] = allocate(epoch_m, D, quarters, premium=False)
    flow, stock, bands = band_frames(epoch_m, flows, HEADLINE_NAME, non_ci=NON_CI_VARIANTS)
    for df in (flow, stock):
        df.insert(1, "tier", df["dest"].map(_tier))
    _warn_label_collisions(set(flow.loc[flow["tier"] == "thin", "dest"]))
    ev_q, ev_c = destination_evidence(coC, resid_co)
    flow = flow.merge(ev_q, on=["dest", "q"], how="left")
    stock = stock.merge(ev_c, on=["dest", "q"], how="left")
    stock["evidence"] = stock["evidence"].fillna("n/a")
    carried = sorted(D.attrs.get("carried_forward", []))
    cf = set(carried)
    flow["share_carried_forward"] = [(d_, q_) in cf for d_, q_ in zip(flow["dest"], flow["q"])]

    # EU member annex (indicative split of the EU row)
    eu_sh = eu_member_shares(res)
    eu_flow = flow[flow["dest"] == "European Union"].groupby("q")[["units", "h100e"]].sum()
    annex = eu_sh.reindex(eu_flow.index).ffill().mul(1.0)
    ann = annex.stack().rename("share").reset_index()
    ann.columns = ["q", "member", "share"]
    ann = ann.merge(eu_flow.reset_index(), on="q")
    ann["units"] = ann["share"] * ann.pop("units")
    ann["h100e"] = ann["share"] * ann.pop("h100e")
    qi = {q: i for i, q in enumerate(sorted(eu_flow.index))}
    ann = ann.sort_values("q", key=lambda c: c.map(qi))
    ann[["units_cum", "h100e_cum"]] = ann.groupby("member")[["units", "h100e"]].cumsum()
    ann["nl_flag"] = ann["member"] == "Netherlands"

    report(stock, flow, epoch_m, fell_back, eu_sh, quarters, bands)

    coverage = epoch_m.groupby("q").partial_sales_coverage.any()
    last_window_q = head_D.index.get_level_values("q").max()
    for output in (flow, stock, shares_long, ann):
        meta = timing.metadata(output["q"])
        for col in meta:
            output[col] = meta[col].to_numpy()
        output["partial_sales_coverage"] = output.q.map(coverage).fillna(True)
        output["shares_window_end"] = output.q.map(
            lambda q: str(pd.Period(min(q, last_window_q), freq="Q").asfreq("M", "end")))
        output["timing_basis"] = "calendar customs quarters; uniform within-quarter sales"
    meta = timing.metadata(bands["q"])
    for col in meta:
        bands[col] = meta[col].to_numpy()
    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    flow.to_csv(paths.output_path("gpu_country_flows"), index=False)
    bands.to_csv(paths.output_path("gpu_country_bands"), index=False)
    stock.to_csv(paths.output_path("gpu_country_stocks"), index=False)
    shares_long.to_csv(paths.output_path("gpu_country_shares"), index=False)
    ann.to_csv(paths.output_path("gpu_country_eu_members"), index=False)
    imputation.to_csv(paths.output_path("gpu_country_imputation"), index=False)
    cells = (coC[coC["q"].map(lambda q: q >= AI_LO)]
             .groupby(["dest", "code_hs6", "q"])["chiponly_corr"].sum())
    origin_mix(res, bindex=U["bindex"], cmodel=cm, cells=cells).to_csv(
        paths.output_path("gpu_country_origin_mix"), index=False)
    from .estimator import write_manifest
    write_manifest("allocator", variants=sorted(flows), quarter_axis=[quarters[0], quarters[-1]],
                   shares_carried_forward=[list(x) for x in carried],
                   premium_series=[[q0, fac] for q0, fac, _ in PREMIUM_SERIES],
                   captive_geography={m: sorted(d) for m, d in load_captive_geography().items()})
