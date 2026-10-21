"""The estimator: accelerator chip dollars per destination, from customs, compared with Epoch.

Per destination x HS-6 code x month:

1. the loaded books on the FOB basis (``books``, ``fob``);
2. the single-counted destination universe, with the 2021 history attached (``netting``);
3. counted excess over the measured non-AI reference -- 2021 trade / 12 x the baseline index
   normalized to 2021 -- through the netting ledger (``ledger``, ``baseline``, ``history``);
4. chip dollars = excess x accelerator content (``content``; destinations without a book via
   their exporters' records, ``exporter_content``).

The customs series is compared with Epoch's sales moved to calendar arrival quarters
(``timing``). Besides the headline (indexed baseline, measured content), an assumption band
crosses the baseline {indexed, grown 5% a year} with content {flat per-code constants
("mfg"), measured ("corridor"), flat 0.85 on systems ("high085")}; the k corners and the
Malaysia mirror endpoints enter as further complete universes. Every combination's
per-destination chip dollars are collected for the allocator.
"""
import json
import subprocess
from copy import deepcopy

import numpy as np
import pandas as pd

from . import baseline, books, codes, content, epoch, exporter_content, fob, history, ledger, netting, paths, timing

AI_LO = codes.AI_LO
CARVE_WINDOW = ("2023Q3", "2025Q4")     # quarters whose excess splits the 2022 carve across codes
HEADLINE = ("indexed", "corridor")
BAND_CONTENTS = ("mfg", "corridor", "high085")
HIGH_SYSTEM_CONTENT = 0.85


def load_res():
    """Every book loaded, canonicalized, trailing months dropped, CIF imports put on FOB."""
    print("[load] reading every book ...")
    res = books.drop_provisional(books.load_all(), books.DROP_TRAILING_MONTHS)
    res["rep_c"] = res["reporter"].map(books.canon)
    res["par_c"] = res["partner"].map(books.canon)
    return fob.fobize(res)


def mirror_baseline_index(bindex, fa):
    """A copy of ``bindex`` whose world composite is re-weighted by universe ``fa``'s 2022 trade
    (each structural scenario carries its own composite)."""
    result = deepcopy(bindex)
    weights = fa[fa.period.str.startswith("2022")].groupby(
        ["dest", "code_hs6"]).value_usd.mean().to_dict()
    result.set_composite(weights)
    return result


def _complete_quarters(periods):
    q = pd.Series(sorted(set(periods)))
    q = q.groupby(q.map(codes.quarter)).nunique()
    return set(q[q == 3].index)


def build_universe(res=None, index_options=None, content_model=None, mirror_share=None):
    """Everything the estimator and allocator run on.

    Returns a dict: ``res`` (loaded books), ``t`` and ``epoch_all`` (Epoch timelines and
    quarterly sales), ``axis`` (Epoch quarters that are complete quarters of Taiwan's export
    book), ``epoch_q``, ``bindex``, ``cmodel``, ``r_resid`` (the residual bucket's uplift),
    ``residC`` (Taiwan's residual bucket), ``faC_full`` (the headline universe), ``faK`` (k
    corners), ``faMirror`` (Malaysia endpoints). Pass a loaded ``res`` to skip the reload.

    The structural cases of the uncertainty run pass ``index_options`` (keyword arguments of
    ``baseline.BaselineIndex``), ``content_model`` (a ``content.ContentModel`` subclass) and
    ``mirror_share`` (the headline universe's Malaysia deduction share).
    """
    if res is None:
        res = load_res()
    t = epoch.load_timelines()
    epoch_all = epoch.quarterly_sales(t)
    world_basket = books.load_taiwan_exports()

    # The residual bucket is Taiwan's exports at Taiwan's declared values. Importers declare the
    # same hardware higher; the uplift is importers' Taiwan-origin imports (net of hub re-exports)
    # over Taiwan's exports, pooled over the four system codes on common months, because Taiwan
    # and its importers file the same hardware under different system codes.
    v2a = ledger.v2a_importer_netted(res)
    common = set(v2a["period"]) & set(world_basket["period"])

    def code_sums(df):
        d = df[df["period"].isin(common)]
        return d[d["period"].map(codes.quarter) >= AI_LO].groupby("code_hs6")["value_usd"].sum()

    num, den = code_sums(v2a), code_sums(world_basket)
    r_code = (num / den).to_dict()
    fam = [c for c in codes.BASKET if c not in codes.CHIPS]
    pooled = float(num.reindex(fam).sum() / den.reindex(fam).sum())
    r_resid = {c: (pooled if c in fam else r_code.get(c, 1.0)) for c in codes.BASKET}

    tw_quarters = _complete_quarters(world_basket["period"])
    axis = [q for q in epoch_all["quarter"].tolist() if q in tw_quarters]
    epoch_q = epoch_all.set_index("quarter")["epoch_cost_usd"].reindex(axis)
    epoch_q.attrs["incomplete_quarters"] = epoch_all.loc[epoch_all.epoch_incomplete, "quarter"].tolist()

    baseline.FRONTIER = res["period"].max()        # the index extensions stop where the trade data stop
    bindex = baseline.BaselineIndex(**(index_options or {}))
    thb = netting.thailand_books()
    extra = {"Thailand": thb} if thb is not None else None
    cm = (content_model or content.ContentModel)(res, t, bindex=bindex, cn_excluded=netting.CN_EXCLUDED_ORIGINS,
                                                 extra_books=extra)
    residC = netting.residual_value(res, r_resid, exclude=netting.ASSEMBLY_NODES)
    faC = netting.uniform_universe(res, content_model=cm, mirror_share=mirror_share)
    faC_full = pd.concat([faC, residC], ignore_index=True)
    faK = {}
    for ktag, kanch in netting.K_BRACKET.items():
        faK[ktag] = pd.concat([netting.uniform_universe(res, k_anch=kanch, content_model=cm, tag=ktag),
                               residC], ignore_index=True)

    def nochips(df):
        return df[~df["code_hs6"].isin(codes.CHIPS)].copy()
    dropped = faC_full[faC_full["code_hs6"].isin(codes.CHIPS)]["value_usd"].sum()
    faC_full = nochips(faC_full)
    faK = {k: nochips(v) for k, v in faK.items()}
    faMirror = {}
    for name, share in (("my_server_none", 0.0), ("my_server_full", 1.0)):
        faMirror[name] = nochips(pd.concat([
            netting.uniform_universe(res, content_model=cm, tag=name, mirror_share=share),
            residC], ignore_index=True))
    print(f"[chips] bare chips (854231) count toward no destination: "
          f"${dropped/1e9:.0f}B gross dropped")
    # Destinations without their own index follow the world composite, weighted by 2022 trade.
    w22 = (faC_full[faC_full["period"].str[:4] == "2022"]
           .groupby(["dest", "code_hs6"])["value_usd"].mean().to_dict())
    bindex.set_composite(w22)
    # The content model's cells use the destination's baseline index; a classified pair without
    # its own index would be inconsistent with its ledger.
    unindexed = sorted(p for p in cm._cellcls_pairs if bindex.factor(*p) is None)
    if unindexed:
        raise ValueError(f"[content-model] classified pairs without their own baseline index: {unindexed}")
    return {"res": res, "t": t, "epoch_all": epoch_all, "axis": axis, "epoch_q": epoch_q,
            "bindex": bindex, "cmodel": cm, "r_resid": r_resid, "residC": residC,
            "faC_full": faC_full, "faK": faK, "faMirror": faMirror}


def _check_history(fa):
    """The universe must carry the 2021 history, complete for every counted destination x code."""
    if not fa.period.str.startswith("2021-").any():
        raise ValueError("the universe lacks its 2021 history")
    covered = fa[fa.code_hs6.isin(history.CODES)]
    pairs = covered.groupby(["dest", "code_hs6"]).ngroups
    history_pairs = covered[covered.period.str.startswith("2021-")].groupby(
        ["dest", "code_hs6"]).period.nunique()
    if len(history_pairs) != pairs or history_pairs.ne(12).any():
        raise ValueError("incomplete 2021 history in the universe")


def carve_2022(fa, epoch_all, bindex=None):
    """The A100-era AI content of 2022, {code: $/month}, carved out of the 2022 reference.

    Used only on the 2022 reference (``history.REFERENCE_YEAR``). The level is Epoch's 2022
    sales per month, taken directly as border dollars; it is split across codes by their shares
    of world excess over CARVE_WINDOW in a first pass without a carve, with the baseline grown by
    ``bindex``'s world composite (weighted by ``fa``'s 2022 trade) or, without an index, at the
    parametric rate.
    """
    w = fa.groupby(["code_hs6", "period"], as_index=False)["value_usd"].sum()
    w["year"] = w["period"].str[:4]
    base = w[w["year"] == "2022"].groupby("code_hs6").agg(mean2022=("value_usd", "mean"))
    w = w.merge(base, on="code_hs6", how="left")
    w["gf"] = (1.0 + w["code_hs6"].map(lambda c: codes.NONAI_GROWTH[codes.channel(c)])) ** \
        w["period"].map(codes.years_since_mid2022)
    if bindex is not None:
        weights = fa[fa["period"].str[:4] == "2022"].groupby(["dest", "code_hs6"])["value_usd"].mean().to_dict()
        for code, f in bindex.aggregate(weights).items():
            m = (w["code_hs6"] == code).to_numpy()
            if not m.any():
                continue
            ext = f.reindex(sorted(set(f.index) | set(w.loc[m, "period"]))).ffill()
            vals = ext.reindex(w.loc[m, "period"]).to_numpy()
            w.loc[m, "gf"] = np.where(np.isnan(vals), w.loc[m, "gf"].to_numpy(), vals)
    w["excess"] = (w["value_usd"] - w["mean2022"].clip(lower=0.0) * w["gf"]).clip(lower=0.0)
    calib = w[w["period"].map(codes.quarter).between(*CARVE_WINDOW)]
    shares = calib.groupby("code_hs6")["excess"].sum().pipe(lambda s: s / s.sum()).to_dict()
    monthly = epoch_all.loc[epoch_all["quarter"].str.startswith("2022"), "epoch_cost_usd"].sum() / 12.0
    return {c: monthly * shares.get(c, 0.0) for c in codes.BASKET}


def content_weights(co, cmodel, rack=None):
    """Measured content per row of the excess frame ``co``: the content model for books, the
    exporters' evidence for the thin tier."""
    fb = (co["code_hs6"].map(content.SALE_FLAT)
          .fillna(co["code_hs6"].map(codes.CHIP_CONTENT)).to_numpy(dtype=float))
    w = cmodel.lookup(co["dest"], co["code_hs6"], co["q"], fb, rack=rack, periods=co["period"])
    return exporter_content.weights(co, cmodel, w, rack=rack)


def estimate(U, fa, bindex, full_band=True, dest_collector=None, weigh=None):
    """Chip dollars for universe ``fa`` under the band's baseline x content combinations.

    ``full_band=False`` computes the headline combination only. ``dest_collector`` (a dict)
    receives each combination's chip dollars per destination x calendar quarter, keyed
    (baseline, content). ``weigh(co, cmodel)`` replaces ``content_weights`` for the measured
    content (the uncertainty run's selection rules). Returns ``(headline_q, headline_co, band,
    headline_lag)``: the headline chip dollars per quarter, its row-level excess frame (with
    ``chiponly_corr``), one band row per combination with its median ratio to Epoch, and the
    headline per arrival quarter.
    """
    cmodel, epoch_all, epoch_q = U["cmodel"], U["epoch_all"], U["epoch_q"]
    exporter_content.evidence(cmodel, U["res"])
    if history.ENABLED:
        _check_history(fa)
    ai = [q for q in U["axis"] if q >= AI_LO]
    variants = [("indexed", bindex, 0.05)]
    if full_band:
        variants.append(("grown+5%", None, 0.05))
    contents = list(BAND_CONTENTS) if full_band else [HEADLINE[1]]
    lag_epoch = timing.matched_sales(timing.sales_from_epoch(epoch_all))
    rows = []
    headline_q = headline_co = headline_lag = None
    for vname, bidx, g in variants:
        growth = {"systems": g, "chips": min(g, codes.NONAI_GROWTH["chips"]) if g == 0 else g * 0.6}
        carve = None if history.ENABLED else carve_2022(fa, epoch_all, bidx)
        co = ledger.corridor_chiponly(fa, "dest", growth=growth, baseline_index=bidx,
                                      historical=history.ENABLED, ai2022=carve)
        for rule in contents:
            if rule == "mfg":
                w = co["code_hs6"].map(codes.CHIP_CONTENT).to_numpy()
            elif rule == "corridor":
                w = (weigh or content_weights)(co, cmodel)
            else:
                w = np.where(co["code_hs6"].isin(codes.SERVER_CODES), HIGH_SYSTEM_CONTENT,
                             co["code_hs6"].map(codes.CHIP_CONTENT))
            chip = co["excess"].to_numpy() * w
            aq = pd.Series(chip, index=co.index).groupby(co["q"]).sum()
            if dest_collector is not None:
                dest_collector[(vname, rule)] = timing.aggregate(co, chip, by=("dest",))
            med = float((aq.reindex(ai) / epoch_q.reindex(ai)).median())
            sl = timing.aggregate(co, chip)
            lag_ai = [q for q in sl.index if q >= AI_LO]
            med_lag = float((sl.reindex(lag_ai) / lag_epoch.reindex(lag_ai)).median())
            rows.append({"baseline": vname, "growth": g, "content": rule,
                         "median_ratio": med, "median_ratio_lag": med_lag})
            if (vname, rule) == HEADLINE:
                headline_q, headline_lag = aq, sl
                headline_co = co.assign(chiponly_corr=chip)
    return headline_q, headline_co, pd.DataFrame(rows), headline_lag


def run(U):
    """Run every universe through ``estimate``, write the estimator's outputs, and return what
    the allocator needs: the headline excess frame and every share set."""
    res, cm, axis, epoch_q = U["res"], U["cmodel"], U["axis"], U["epoch_q"]
    collectors = {}
    headline_q, coC, band, lag = estimate(U, U["faC_full"], U["bindex"], dest_collector=collectors)
    bands = [band]
    variants = {}
    for vtag, fa_k in U["faK"].items():
        c = {}
        estimate(U, fa_k, U["bindex"], full_band=False, dest_collector=c)
        variants[vtag] = c[HEADLINE]
    mirror_indices = {}
    for vtag, fa_m in U["faMirror"].items():
        bi = mirror_baseline_index(U["bindex"], fa_m)
        mirror_indices[vtag] = bi
        c = {}
        _, _, b, _ = estimate(U, fa_m, bi, full_band=False, dest_collector=c)
        bands.append(b.assign(baseline=vtag))
        variants[vtag] = c[HEADLINE]
    band = pd.concat(bands, ignore_index=True)

    matched_epoch = timing.matched_sales(timing.sales_from_epoch(U["epoch_all"]))
    lag_ai = [q for q in lag.index if q >= AI_LO]
    lag_path = (lag.reindex(lag_ai) / matched_epoch.reindex(lag_ai)).dropna()
    mq = lag_path.index
    compare = timing.metadata(mq)
    compare["epoch_arrivals"] = matched_epoch.reindex(mq).to_numpy()
    compare["matched_customs"] = lag.reindex(mq).to_numpy()
    compare["match_pct"] = 100 * lag_path.to_numpy()
    compare["sales_interpolation"] = "uniform within each reported sales quarter"
    cumulative = float(lag.reindex(mq).sum() / matched_epoch.reindex(mq).sum())

    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"quarter": axis, "epoch_sellin": epoch_q.values,
                  "approach2_corrected": headline_q.reindex(axis).values,
                  "epoch_arrivals_interpolated": timing.matched_sales(epoch_q).reindex(axis).to_numpy(),
                  }).to_csv(paths.output_path("gpu_finalprice_quarterly"), index=False)
    band.to_csv(paths.output_path("gpu_finalprice_band"), index=False)
    (coC.groupby(["dest", "code_hs6", "q"], as_index=False)[["excess", "chiponly_corr"]].sum()
        .to_csv(paths.output_path("gpu_finalprice_dest"), index=False))
    compare.to_csv(paths.output_path("gpu_epoch_match"), index=False)
    exporter_content.evidence(cm).to_csv(paths.output_path("gpu_exporter_content"), index=False)
    exporter_content.exposure(coC, cm).to_csv(paths.output_path("gpu_exporter_imputation"), index=False)
    cm.report().to_csv(paths.output_path("gpu_content_cells"), index=False)
    pd.Series(cm.monthly_cells, name="content").rename_axis(
        ["dest", "code_hs6", "period"]).reset_index().to_csv(
            paths.output_path("gpu_content_monthly"), index=False)

    print(f"\n=== Customs chip dollars vs Epoch sales (arrival quarters, {timing.LAG_MONTHS}-month lag) ===")
    print(f"  {mq[0]}..{mq[-1]}: cumulative {cumulative:.1%}, median quarter "
          f"{lag_path.median():.1%}")
    print("  " + compare[["arrival_quarter", "epoch_arrivals", "matched_customs", "match_pct"]]
          .round(1).to_string(index=False).replace("\n", "\n  "))
    print(f"  assumption band, median quarterly ratio: {band['median_ratio_lag'].min():.2f}"
          f"..{band['median_ratio_lag'].max():.2f}")
    write_manifest("estimator", lagmatched_cumulative=round(cumulative, 4),
                   lagmatched_median=round(float(lag_path.median()), 4),
                   fob_basis=fob.get_model(verbose=False).manifest(),
                   reexport_shares={"eu_weighted": round(netting.reexport_shares(res).eu_rx_wtd, 4),
                                    "eu_members": {k: round(v, 4) for k, v in
                                                   netting.reexport_shares(res).eu_rx.items()},
                                    "other": dict(netting.UN_LEG_RX)},
                   netting_ledger=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                                   for r in netting.UN_LEDGER["headline"].to_dict("records")],
                   content_cells=[f"{d} {c} [{f}] {n} cells, prior {p:.3f}"
                                  for (d, c), (f, n, p) in cm._cellcls_pairs.items()])
    return {"coC": coC, "collectors": collectors, "variants": variants,
            "mirror_indices": mirror_indices}


def write_manifest(section, **payload):
    """Record the run's provenance in ``gpu_run_manifest_<stamp>.json`` (one section per stage)."""
    path = paths.output_path("gpu_run_manifest", "json")
    doc = {}
    if path.exists():
        try:
            doc = json.loads(path.read_text())
        except ValueError:
            doc = {}
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.PACKAGE_DIR,
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:               # noqa: BLE001 -- not a git checkout
        rev = "unknown"
    doc[section] = {"git_rev": rev,
                    "parameters": {"netting_quarters": ledger.NETTING_QUARTERS,
                                   "lag_months": timing.LAG_MONTHS,
                                   "malaysia_mirror_share": netting.MY_SERVER_MIRROR_SHARE},
                    **payload}
    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
