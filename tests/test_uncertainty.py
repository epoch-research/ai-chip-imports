"""Uncertainty run: the case set, the content treatments, the sales and parameter perturbations."""
import numpy as np
import pandas as pd
import pytest

from ai_chip_imports import exporter_content, ledger, netting
from ai_chip_imports.exporter_content import Perturbation
from ai_chip_imports.uncertainty import cases, sampling
from ai_chip_imports.uncertainty.model import BaselineStress


def test_case_set():
    """48 specifications x three Malaysia deductions; the case properties drive the pipeline."""
    assert len(cases.CASES) == 144
    assert {c.mirror_share for c in cases.CASES.values()} == {0., .5, 1.}
    central = cases.CASES[cases.CENTRAL]
    assert (central.rebuild, central.capped, central.composition_mode) == (False, False, "control")
    assert central.premium is None
    assert cases.CASES["named__ref2022__my_full"].reference_year == 2022
    high = cases.CASES["named__china_high"]
    assert high.premium is False and high.index_options == {"reference": "adopted", "origin_overrides": {}}
    sel = cases.CASES["boards_client_it__capped_ladder__ordinary_first"]
    assert sel.capped and sel.preference == 1. and sel.rebuild and sel.composition_mode == "control"
    assert cases.CASES["named__parts_low"].composition_mode == "parts_low"
    assert cases.CASES["central__parts_mixed"].composition_mode == "parts_mixed"
    assert cases.CASES["asp_merchant"].merchant_prices and not cases.CASES["asp_merchant"].rebuild


def test_transform_keeps_low_and_native_readings():
    """Readings at or below the floor and NaNs survive every composition stress."""
    v = np.array([np.nan, 0., .05, .5, .93])
    assert np.array_equal(cases.transform(v, "847330", "parts_low"), [np.nan, 0, .05, .05, .05], equal_nan=True)
    np.testing.assert_allclose(cases.transform(v, "847330", "parts_mixed")[3:], [.275, .49])
    assert cases.transform(v, "847330", "parts_high")[3] == .93
    assert cases.transform(v, "847180", "units_assembly")[4] == .73
    assert np.array_equal(cases.transform(v, "847150", "joint_mixed"), v, equal_nan=True)
    assert np.array_equal(cases.transform(v, "847330", "control"), v, equal_nan=True)


def test_mixture_content():
    """Between the two prices the chip share rises from 0 to the system content; outside it is NaN."""
    p_ai = 8 * 30000. / .73
    got = cases.mixture_content([5000., p_ai, (5000. + p_ai) / 2, 4000., 2 * p_ai], 30000., 5000.)
    assert got[0] == 0 and got[1] == pytest.approx(.73)
    assert got[2] == pytest.approx(.5 * p_ai * .73 / ((5000. + p_ai) / 2))
    assert np.isnan(got[4]) and got[3] == 0


def test_selection_rules():
    """Proportional keeps cE; ordinary-first keeps min(cN, sE); halfway averages them."""
    n, e, c, s = 100., 40., .5, .73
    assert cases.selected_chip_value(n, e, c, s, 0.) == pytest.approx(20.)
    assert cases.selected_chip_value(n, e, c, s, 1.) == pytest.approx(min(50., 29.2))
    assert cases.selected_chip_value(n, e, c, s, .5) == pytest.approx((20. + 29.2) / 2)
    with pytest.raises(ValueError):
        cases.selected_chip_value(n, e, .9, s, 0.)          # content above the ceiling


def test_quantile_factor_hits_the_percentiles():
    z = sampling.Z90
    f = sampling.quantile_factor([100.] * 4, [50., 50., 0., 50.], [300., 300., 300., 0.], [-z, z, -z, z])
    np.testing.assert_allclose(f, [.5, 3., 1., 1.])        # missing bounds keep the median


def models():
    return pd.DataFrame(dict(model=["H100", "MI300X", "Gaudi 3"], designer=["Nvidia", "AMD", "Intel"],
                             units=[100.] * 3, units_lo=[50.] * 3, units_hi=[200.] * 3,
                             h100e=[100.] * 3, h100e_lo=[50.] * 3, h100e_hi=[200.] * 3,
                             asp=[30000., 15000., 10000.], cost=[3e6, 1.5e6, 1e6]))


def test_sales_draw_couples_price_and_quantity():
    """Nvidia keeps its spending at r = 1; AMD splits the log-volume move between price and
    spending; a neutral draw changes nothing."""
    m = models()
    neutral = dict(z={k: 0. for k in m.model}, zp={k: 0. for k in m.model}, zr={k: 0. for k in m.designer})
    same, _ = sampling.apply_sales(m, neutral, "models")
    pd.testing.assert_frame_equal(same, m)
    up = dict(neutral, z={k: sampling.Z90 for k in m.model})
    e, f = sampling.apply_sales(m, up, "models")
    assert e.units.tolist() == pytest.approx([200.] * 3)
    assert e.cost[0] == pytest.approx(3e6)                  # Nvidia: same spending, lower price
    assert e.cost[1] == pytest.approx(1.5e6 * np.sqrt(2))   # AMD: half to spending
    assert f.price_factor[2] == 1.                           # independent price prior at zp = 0


class Index:
    """A baseline index with one destination x code, flat at 1."""

    def __init__(self):
        p = [f"2022-{m:02d}" for m in range(1, 13)] + ["2023-01", "2023-02"]
        self.idx = {("A", "847150"): pd.Series(1., index=p)}

    def lookup(self, dests, codes, periods, fallback):
        return np.ones(len(dests))

    def aggregate(self, weights):
        return {"847150": self.idx[("A", "847150")]}


def test_baseline_stress_scales_from_2023():
    b = BaselineStress(Index(), scale=1.1, by_leg={"A": .9})
    got = b.lookup(["A", "A", "B"], ["847150"] * 3, ["2022-12", "2023-01", "2023-01"], np.ones(3))
    np.testing.assert_allclose(got, [1., .9, 1.1])
    agg = b.aggregate({("A", "847150"): 1.})["847150"]
    assert agg["2022-12"] == 1. and agg["2023-02"] == pytest.approx(.9)


def test_k_shift_moves_only_k_read_from_the_book():
    """The shift applies to server rows whose k comes from the book's own fields, clipped to
    [0, 1]; rows that take the import content are left alone."""
    rows = pd.DataFrame(dict(partner=["P", "P", "Q"], code_hs6=["847150", "847330", "847150"],
                             period=["2024-03"] * 3, value_usd=[2000., 2000., 5000.],
                             weight_kg=[1., 1., 0.], qty=[0., 0., 0.]))
    danch, uanch = netting._k_anchors(None)

    def k(shift):
        return netting._un_k_rows(rows, "L", danch, uanch, lambda *a: .5, k_shift=shift)
    base, moved = k(None), k(lambda leg: .2)
    assert moved[0] == pytest.approx(min(base[0] + .2, 1.))   # server, own field
    assert moved[1] == base[1]                                 # boards are not shifted
    assert moved[2] == base[2] == .5                           # import content
    assert k(lambda leg: 2.)[0] == 1.


def test_carve_is_split_by_2022_means():
    """On the 2022 reference the code's carve comes off each destination's 2022 mean pro rata."""
    p = [f"2022-{m:02d}" for m in range(1, 13)] + ["2024-01"]
    df = pd.DataFrame(dict(dest=["A"] * 13 + ["B"] * 13, code_hs6="847150", period=p * 2,
                           value_usd=[30.] * 13 + [10.] * 13))
    co = ledger.corridor_chiponly(df, "dest", growth={"systems": 0., "chips": 0.}, ai2022={"847150": 8.})
    base = co[co.period.eq("2024-01")].set_index("dest").baseline
    assert base.to_dict() == pytest.approx({"A": 24., "B": 8.})


def test_neutral_perturbation_leaves_the_exporter_classifier_unchanged():
    from test_exporter_content import Model, rows
    plain = exporter_content.classify(rows(), Model())
    m = Model()
    m.exporter_perturbation = Perturbation()
    pd.testing.assert_frame_equal(exporter_content.classify(rows(), m), plain)
    m.exporter_perturbation = Perturbation(price_scale=10.)    # a board at 1x the ASP reads as 0.1x
    assert exporter_content.classify(rows(), m).observed_content[2] == .05
