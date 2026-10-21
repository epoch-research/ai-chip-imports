"""One draw of one case: the whole pipeline under perturbed parameters and sales.

``prepare_universe`` builds the central universe once per run. ``setup`` turns it into a case's
universe (rebuilt when the case changes the reference or the content model). ``run_draw`` then
perturbs Epoch's sales, the baseline, k and the content model, recomputes the netting, the
composite reference, the estimator's headline and the allocation, and checks that every model
and quarter's sampled supply is allocated exactly.
"""
import contextlib
import os
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd

from .. import allocator, codes, epoch, estimator, history, netting, timing
from . import sampling
from .cases import CASES
from .model import BaselineStress, CaseContent, DrawContent, ObservationCache, selection_weigh

AI_FROM = "2023Q3"        # first quarter of the reported chip dollars (``chip_B``)


def prepare_universe():
    """The central universe every case starts from."""
    return estimator.build_universe()


@contextlib.contextmanager
def frozen_thailand(book):
    """Read Thailand's book from ``book`` (the prepared universe's copy) instead of the file, so a
    rebuilt case sees exactly the observations of the prepared central universe."""
    original_books, original_reporter = netting.thailand_books, netting._un_thailand

    def books(basket_only=True):
        return book.copy() if basket_only else original_books(False)

    def reporter(res):
        th = book[book.period.le(res.period.max())].copy()
        th["rep_c"] = "Thailand"
        th["par_c"] = th.partner.map(netting.books.canon)
        return th
    netting.thailand_books, netting._un_thailand = books, reporter
    try:
        yield
    finally:
        netting.thailand_books, netting._un_thailand = original_books, original_reporter


class Case:
    """A structural case ready to draw: its universe, its observation cache and its selection
    rule. Build inside ``frozen_thailand``."""

    def __init__(self, name, central):
        self.case = CASES[name]
        if self.case.reference_year != history.REFERENCE_YEAR:
            raise RuntimeError(f"{name} needs AI_CHIP_IMPORTS_REFERENCE_YEAR={self.case.reference_year}")
        u = central
        if self.case.rebuild:
            u = estimator.build_universe(res=central["res"], index_options=self.case.index_options,
                                         content_model=partial(CaseContent, case=self.case),
                                         mirror_share=self.case.mirror_share)
        self.universe = u
        self.cache = ObservationCache(base=u["cmodel"])
        self.weigh = selection_weigh(self.case) if self.case.capped else None
        self.cached = True

    def draw(self, spec):
        """Cumulative arrivals per destination under draw ``spec``: (stock frame, metadata)."""
        u, case = self.universe, self.case
        t, _ = sampling.apply_sales(u["t"], spec, "timelines")
        models, factors = sampling.apply_sales(epoch.models_by_sales_quarter(), spec, "models")
        models = timing.arrival_models(models)
        bi = BaselineStress(deepcopy(u["bindex"]), spec["baseline"], spec["baseline_by_leg"])
        cm = DrawContent(u["res"], t, case, spec, self.cache, cached=self.cached, bindex=bi,
                         extra_books=u["cmodel"]._extra_books, cn_excluded=netting.CN_EXCLUDED_ORIGINS,
                         verbose=False)
        fa = pd.concat([netting.uniform_universe(u["res"], content_model=cm, tag="draw",
                                                 mirror_share=case.mirror_share,
                                                 k_shift=lambda leg: spec["k_by_leg"].get(leg, spec["k_shift"])),
                        u["residC"]], ignore_index=True)
        fa = fa[~fa.code_hs6.isin(codes.CHIPS)]
        bi.base.set_composite(fa[fa.period.str.startswith("2022")].groupby(["dest", "code_hs6"])
                              .value_usd.mean().to_dict())
        sales = epoch.quarterly_sales(t)
        sales_q = sales.set_index("quarter").epoch_cost_usd.reindex(u["epoch_q"].index)
        sales_q.attrs["incomplete_quarters"] = sales.loc[sales.epoch_incomplete, "quarter"].tolist()
        dollars = {}
        _, co, _, _ = estimator.estimate({**u, "cmodel": cm, "epoch_all": sales, "epoch_q": sales_q}, fa, bi,
                                         full_band=False, dest_collector=dollars, weigh=self.weigh)
        head = dollars[estimator.HEADLINE]
        quarters = allocator.allocation_quarters(models, head, u["res"], verbose=False)
        carry = allocator.incomplete_dest_quarters(u, quarters)
        residual = allocator.residual_country_split(u["res"], u["r_resid"], bindex=bi, cmodel=cm)[0]
        flow = allocator.allocate(models, allocator.clean_D(head, residual, carry), quarters, premium=case.premium)
        expected = models[models.q.isin(quarters)].groupby(["model", "q"])[["units", "h100e"]].sum()
        actual = flow.groupby(["model", "q"])[["units", "h100e"]].sum().reindex(expected.index).fillna(0)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-4)
        if not (np.isfinite(flow[["units", "h100e"]]).all().all() and flow[["units", "h100e"]].ge(0).all().all()):
            raise AssertionError("negative or non-finite allocation")
        stock = flow.groupby("dest")[["units", "h100e"]].sum()
        stock["share_pct"] = 100 * stock.h100e / stock.h100e.sum()
        meta = dict(quarters=quarters, carried_forward=sorted(map(list, carry)),
                    total_h100e=float(stock.h100e.sum()),
                    chip_B=float(co.loc[co.q.ge(AI_FROM), "chiponly_corr"].sum() / 1e9))
        netting.UN_LEDGER.clear()
        return stock, meta, factors


def deterministic_stock(U, fa, bindex, premium=None):
    """Cumulative arrivals per destination of the unperturbed pipeline on universe ``fa``."""
    dollars = {}
    estimator.estimate(U, fa, bindex, full_band=False, dest_collector=dollars)
    head = dollars[estimator.HEADLINE]
    models = epoch.models_by_arrival_quarter()
    quarters = allocator.allocation_quarters(models, head, U["res"], verbose=False)
    carry = allocator.incomplete_dest_quarters(U, quarters)
    residual = allocator.residual_country_split(U["res"], U["r_resid"], bindex=bindex, cmodel=U["cmodel"])[0]
    flow = allocator.allocate(models, allocator.clean_D(head, residual, carry), quarters, premium=premium)
    return flow.groupby("dest")[["units", "h100e"]].sum()


def controls(central):
    """Deterministic results the neutral draws must reproduce: the central universe at each
    Malaysia deduction, and the merchant-price universe rebuilt from the same books."""
    out = {"central__central": deterministic_stock(central, central["faC_full"], central["bindex"])}
    for name, suffix in (("my_server_none", "__my_none"), ("my_server_full", "__my_full")):
        fa = central["faMirror"][name]
        U = {**central, "faC_full": fa, "bindex": estimator.mirror_baseline_index(central["bindex"], fa)}
        out["central__central" + suffix] = deterministic_stock(U, fa, U["bindex"])
    merchant = estimator.build_universe(res=central["res"],
                                        content_model=partial(CaseContent, case=CASES["asp_merchant"]))
    for suffix, fa in (("", merchant["faC_full"]), ("__my_none", merchant["faMirror"]["my_server_none"]),
                       ("__my_full", merchant["faMirror"]["my_server_full"])):
        U = {**merchant, "faC_full": fa, "bindex": estimator.mirror_baseline_index(merchant["bindex"], fa)}
        out["asp_merchant" + suffix] = deterministic_stock(U, fa, U["bindex"])
    return out


def single_threaded_env():
    """Environment for case processes: one BLAS thread each (the runner parallelizes cases)."""
    return {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
