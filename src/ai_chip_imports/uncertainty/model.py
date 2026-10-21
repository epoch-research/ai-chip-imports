"""The model pieces a draw changes: the baseline index, the content model and the selection rule.

``BaselineStress`` scales a baseline index from January 2023. ``CaseContent`` is the content
model of a structural case (merchant-only prices, the low-content adjustments). ``DrawContent``
reclassifies every physical cell under a draw's parameters and the case's content treatment.

The trade observations behind the cells never change between draws, so ``DrawContent`` reads
them once per worker (``ObservationCache``) and recomputes only what depends on the draw: the
reference prices, the content readings and the cells' baselines. ``cached=False`` recomputes
the cells from the rows instead; the runner's probe checks that both agree.
"""
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import books, content, netting
from ..exporter_content import Perturbation
from .cases import SERVER_CODES, SYSTEM, mixture_content, mixture_price, selected_chip_value, transform

NESOI = "8473301180"      # US printed circuit assemblies, not elsewhere specified
CAPTIVE_PATTERN = r"^(TPU|Trainium|Inferentia)"
STRESS_FROM = "2023-01"


class BaselineStress:
    """A baseline index whose values from STRESS_FROM are multiplied by ``scale`` (``by_leg``
    overrides it per destination). Exposes the interface the pipeline uses."""

    def __init__(self, base, scale=1., by_leg=None):
        self.base = base
        self.scale = scale
        self.by_leg = by_leg or {}

    def factor(self, dest, code):
        return self.base.factor(dest, code)

    @property
    def composite(self):
        return self.base.composite

    def multiplier(self, dest):
        return self.by_leg.get(dest, self.scale)

    def lookup(self, dests, codes, periods, fallback):
        a = self.base.lookup(dests, codes, periods, fallback)
        mult = np.array([self.multiplier(d) if str(p) >= STRESS_FROM else 1. for d, p in zip(dests, periods)])
        return a * mult

    def aggregate(self, weights):
        b = deepcopy(self.base)
        b.idx = {key: s * np.where(s.index >= STRESS_FROM, self.multiplier(key[0]), 1.)
                 for key, s in b.idx.items()}
        return b.aggregate(weights)


# --- structural content adjustments ------------------------------------------------------------
def merchant_asp(model):
    """The rest-of-world reference price from merchant models only (captive TPU, Trainium and
    Inferentia excluded)."""
    key = ("row", "merchant")
    if key not in model._asp_cache:
        d = model._ep[~model._ep["cn_spec"] & ~model._ep["base"].str.match(CAPTIVE_PATTERN)]
        s = d.groupby("quarter").apply(
            lambda g: g["cost"].sum() / g["units"].sum() if g["units"].sum() > 0 else np.nan,
            include_groups=False)
        model._asp_cache[key] = s.reindex(model._quarters).ffill().bfill()
    return model._asp_cache[key]


def reinstall_quarterly(model, frame):
    """Rebuild the quarterly cells and priors of the pairs in a modified cell ``frame`` the way the
    classifier builds them: excess-weighted content per quarter, three-quarter median, gated
    quarters, AI-window prior."""
    for (dest, code), oq in frame.groupby(["dest", "code_hs6"]):
        cl = oq[oq["c"].notna() & (oq["excess"] > 0)]
        if cl.empty:
            continue
        g = (cl.assign(wc=cl["excess"] * cl["c"]).groupby("q")
             .agg(w=("excess", "sum"), wc=("wc", "sum")).sort_index())
        g["c_raw"] = g["wc"] / g["w"]
        g["c_smooth"] = content._median3(g["c_raw"], g.index)
        pw = g[g.index >= content.AI_LO]
        if pw["w"].sum() <= 0:
            continue
        for k in [k for k in model.cells if k[:2] == (dest, code)]:
            del model.cells[k]
        for q, row in g[g["w"] >= content.MIN_EXCESS_Q].iterrows():
            model.cells[(dest, code, q)] = float(row["c_smooth"])
        model.prior[(dest, code)] = float((pw["c_smooth"] * pw["w"]).sum() / pw["w"].sum())


def cap_servers(model):
    """Cap the books' unit-value server cells at the eligible system mix (quarterly and monthly)."""
    for attr, monthly in (("_cellcls_frame", False), ("_monthly_frame", True)):
        f = getattr(model, attr)
        if f is None or f.empty:
            continue
        f = f.copy()
        m = f["code_hs6"].isin(SERVER_CODES) & f["field"].eq("qty") & f["c"].notna()
        mix = np.array([model.mix(model._side(d)).get(q, np.nan) for d, q in zip(f["dest"], f["q"])])
        f.loc[m, "c"] = np.minimum(f.loc[m, "c"].to_numpy(), mix[m.to_numpy()])
        if monthly:
            model.install_monthly(f[f["code_hs6"].isin(SERVER_CODES)])
            model._monthly_frame = f
        else:
            reinstall_quarterly(model, f[f["code_hs6"].isin(SERVER_CODES)])
            model._cellcls_frame = f


def nesoi_density(model):
    """Read the US NESOI boards line by its own $/kg on the boards ramp (gross shipping weight)
    instead of the board tier."""
    path = books.find_national_csv("us")
    d = pd.read_csv(path, usecols=["flow", "partner", "code_native", "period", "value_usd", "weight_kg"],
                    dtype={"code_native": str})
    d = d[d.flow.eq("M") & d.code_native.eq(NESOI) & ~d.partner.map(books.NORM).isin({"world", "total"})]
    d = d[pd.to_numeric(d.weight_kg, errors="coerce").fillna(0) > 0]
    lo, hi = content.CELL_DENS["boards"]

    def ramp(v, kg):
        w = np.clip((np.log(np.maximum(v / kg, 1e-9)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
        return content.TIER_C["low"] + (content.TIER_C["board"] - content.TIER_C["low"]) * w
    monthly = d.groupby("period")[["value_usd", "weight_kg"]].sum()
    c_month = ramp(monthly.value_usd, monthly.weight_kg)
    d["q"] = d.period.map(lambda p: str(pd.Period(p, freq="M").asfreq("Q")))
    quarterly = d.groupby("q")[["value_usd", "weight_kg"]].sum()
    c_quarter = ramp(quarterly.value_usd, quarterly.weight_kg)
    n = model._native_monthly_frame.copy()
    sel = n.dest.eq("United States") & n.code_native.eq(NESOI)
    n.loc[sel, "c_line"] = n.loc[sel, "period"].map(c_month).fillna(content.TIER_C["board"]).to_numpy()
    model._native_monthly_frame = n
    us = n[n.dest.eq("United States") & n.code_hs6.eq("847330")].rename(columns={"c_line": "c"})
    model.install_monthly(us.assign(coverage=1.0))
    nv = model._native_frame.copy()
    sel = nv.dest.eq("United States") & nv.code_native.eq(NESOI)
    nv.loc[sel, "c_line"] = nv.loc[sel, "q"].map(c_quarter).fillna(content.TIER_C["board"]).to_numpy()
    model._native_frame = nv
    reinstall_quarterly(model, nv[nv.dest.eq("United States") & nv.code_hs6.eq("847330")]
                        .rename(columns={"c_line": "c"}))


class CaseContent(content.ContentModel):
    """The content model of a structural case: merchant-only reference prices for
    ``case.merchant_prices``; capped server cells and the NESOI line by density for
    ``case.low_content``. Otherwise identical to ``content.ContentModel``."""

    def __init__(self, res, t, case, **kwargs):
        self.case = case
        super().__init__(res, t, **kwargs)
        if case.low_content:
            cap_servers(self)
            nesoi_density(self)

    def asp(self, side):
        if self.case.merchant_prices and side == "row":
            return merchant_asp(self)
        return super().asp(side)


# --- the per-draw content model ------------------------------------------------------------------
@dataclass
class ObservationCache:
    """Trade observations that are the same in every draw of a worker: the content model's cell
    rows and cell tables (read from ``base``, the case's own content model), native CSVs and the
    slice classifier's reporter rows."""
    base: content.ContentModel
    rows: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    native: dict = field(default_factory=dict)
    reporters: dict = field(default_factory=dict)


def ceiling(model, case, dests, quarters):
    """The chip-share ceiling of measured server content: the HGX-server content in a mixture
    case, else the eligible system mix of the quarter."""
    if case.content.startswith("server_mix"):
        return np.full(len(dests), SYSTEM)
    return np.asarray([model.mix(model._side(d)).get(q, SYSTEM) for d, q in zip(dests, quarters)])


class DrawContent(CaseContent):
    """The case's content model under one draw ``spec``.

    Every cell is read again with the draw's reference-price and density anchors; the server
    mixture replaces the ladder on the server lines it covers (mixture cases); the category shift
    moves the other readings; selection cases cap server content at the ceiling, the low-content
    case at the eligible mix; the composition transform runs last, and Japan's memory cells stay
    at zero. Cell baselines scale with the draw's baseline multiplier.
    """

    def __init__(self, res, t, case, spec, cache, cached=True, **kwargs):
        self.spec = spec
        self.cache = cache
        self.cached = cached
        super().__init__(res, t, case, **kwargs)
        self.exporter_perturbation = Perturbation(
            price_scale=spec["price_anchor"], density_scale=spec["density_anchor"],
            category=spec["category"], transform=lambda v, code: transform(v, code, case.composition_mode),
            cap_at_mix=case.low_content)

    # cached observations
    def _read_native(self, path):
        if path not in self.cache.native:
            self.cache.native[path] = content.ContentModel._read_native(path)
        return self.cache.native[path].copy(deep=True)

    def _build_evidence(self, res, cn_excluded):
        key = id(res)
        if key not in self.cache.reporters:
            reporters = set(content.MEASURED_DESTS)
            if "European Union" in reporters:
                reporters.update(books.EU_MEMBERS)
            self.cache.reporters[key] = (res, res[res.rep_c.isin(reporters)])
        return super()._build_evidence(self.cache.reporters[key][1], cn_excluded)

    def _cell_rows(self, res, dest, code, cn_excluded):
        key = (dest, code, tuple(cn_excluded))
        if key not in self.cache.rows:
            self.cache.rows[key] = content.ContentModel._cell_rows(self.cache.base, res, dest, code, cn_excluded)
        return self.cache.rows[key]

    def _observed_table(self, res, dest, code, field_, cn_excluded, monthly):
        """The case's cell table, with this draw's baselines and reference prices."""
        if not self.cached:
            return content.ContentModel._cell_table(self, res, dest, code, field_, cn_excluded, monthly=monthly)
        key = (dest, code, field_, tuple(cn_excluded), monthly)
        if key not in self.cache.tables:
            self.cache.tables[key] = content.ContentModel._cell_table(
                self.cache.base, res, dest, code, field_, cn_excluded, monthly=monthly)
        oq = self.cache.tables[key]
        if oq is None:
            return None
        oq = oq.copy()
        scale = self.bindex.multiplier(dest) if isinstance(self.bindex, BaselineStress) else 1.
        oq["bv"] *= np.where(oq.q.ge("2023Q1"), scale, 1.)
        side = self._side(dest)
        srv = self.mix(side).reindex(oq.q).to_numpy()
        if field_ == "qty":
            asp = self.asp(side).reindex(oq.q).to_numpy()
            oq["asp_ref"] = asp
            oq["ratio"] = oq.uv / asp
            oq["c"] = [content._tier_content(r, s) for r, s in zip(oq.ratio, srv)]
        else:
            lo, hi = content.CELL_DENS["boards" if code == "847330" else "server"]
            w = np.clip((np.log(oq.uv.clip(lower=1e-9)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
            oq["c"] = .05 + ((.93 if code == "847330" else srv) - .05) * w
        if dest == "Japan" and code == "847330":
            oq.loc[oq.cell.str.endswith("|memory"), "c"] = 0.
        oq["excess"] = (oq.v - oq.bv).clip(lower=0.)
        return oq

    def _cell_table(self, res, dest, code, field_, cn_excluded, monthly=False):
        f = self._observed_table(res, dest, code, field_, cn_excluded, monthly)
        if f is None:
            return None
        f = f.copy()
        spec, case = self.spec, self.case
        srv = self.mix(self._side(dest)).reindex(f.q).to_numpy()
        if field_ == "qty":
            f["c"] = [content._tier_content(r / spec["price_anchor"], s) for r, s in zip(f.ratio, srv)]
        else:
            lo, hi = content.CELL_DENS["boards" if code == "847330" else "server"]
            ramp = np.clip(np.log(f.uv.clip(lower=1e-9) / (lo * spec["density_anchor"])) / np.log(hi / lo), 0, 1)
            f["c"] = .05 + ((.93 if code == "847330" else srv) - .05) * ramp
        modelled = np.zeros(len(f), dtype=bool)
        if case.content.startswith("server_mix") and code in SERVER_CODES and field_ == "qty":
            candidate = mixture_content(f.uv, f.asp_ref * spec["price_anchor"], mixture_price(case.content))
            modelled = np.isfinite(candidate)
            f.loc[modelled, "c"] = candidate[modelled]
        delta = spec["category"].get(code, 0.)
        f.loc[~modelled, "c"] = (f.loc[~modelled, "c"] + delta).clip(.05, .93)
        if code in SERVER_CODES:
            if case.capped:
                f["c"] = np.minimum(f.c, ceiling(self, case, [dest] * len(f), f.q))
            elif case.low_content and field_ == "qty":
                f["c"] = np.minimum(f.c, srv)
        f["c"] = transform(f.c, code, case.composition_mode)
        if dest == "Japan" and code == "847330":
            f.loc[f.cell.str.endswith("|memory"), "c"] = 0.
        return f

    def lookup(self, dests, code_values, quarters, fallback, rack=None, periods=None):
        w = super().lookup(dests, code_values, quarters, fallback, rack=rack, periods=periods)
        mode = self.case.composition_mode
        for code in ("847330", "847180"):
            mask = np.asarray([c == code and (d, c) not in content.NATIVE_TIERS and (d, c) not in self._cellcls_pairs
                               for d, c in zip(dests, code_values)])
            w[mask] = transform(w[mask], code, mode)
        if self.case.capped:
            legs = netting._un_legs()
            mask = np.asarray([d in legs and c in SERVER_CODES for d, c in zip(dests, code_values)])
            w[mask] = np.minimum(w[mask], ceiling(self, self.case, np.asarray(dests)[mask],
                                                  np.asarray(quarters)[mask]))
        return w


def selection_weigh(case):
    """Content weights under the case's selection rule (``cases.selected_chip_value``) on the
    books' measured server lines with positive excess; ``estimator.content_weights`` elsewhere."""
    from .. import estimator
    legs = set(netting._un_legs())

    def weigh(co, cmodel):
        w = estimator.content_weights(co, cmodel)
        mask = co.dest.isin(legs) & co.code_hs6.isin(SERVER_CODES) & co.excess.gt(0)
        t = co.loc[mask]
        if t.empty:
            return w
        m = mask.to_numpy()
        cap = ceiling(cmodel, case, t.dest, t.q)
        new = selected_chip_value(t.value_usd.to_numpy(), t.excess.to_numpy(), w[m], cap, case.preference)
        w = w.copy()
        w[m] = new / t.excess.to_numpy()
        return w
    return weigh
