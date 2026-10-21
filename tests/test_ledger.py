"""The netting ledger and the per-corridor excess over the 2021-anchored baseline."""
import numpy as np
import pandas as pd

from ai_chip_imports import codes, history, ledger

RESIDUAL = "(residual: no importer leg)"


def test_charge_reduces_original_quarter():
    """A shortfall charges back the surplus it offsets in the quarter where that surplus was booked."""
    np.testing.assert_allclose(history.retrospective_ledger([100, 50, -120], 2), [30, 0, 0])
    np.testing.assert_allclose(history.retrospective_ledger([-100, 40, 90], 2), [0, 0, 30])


def test_expiry_prevents_unbounded_revisions():
    """Entries older than the memory expire: they neither charge back nor repay."""
    np.testing.assert_allclose(history.retrospective_ledger([100, 0, 0, -100], 2), [100, 0, 0, 0])
    np.testing.assert_allclose(history.retrospective_ledger([-100, 0, 0, 100], 2), [0, 0, 0, 100])


def test_total_conservation_against_signed_ledger():
    """Charging the original quarter keeps every quarter nonnegative and the total equal to the signed ledger."""
    rng = np.random.default_rng(2021)
    for _ in range(100):
        gaps = rng.normal(0, 100, 24)
        new = history.retrospective_ledger(gaps, 2)
        assert (new >= 0).all()
        np.testing.assert_allclose(new.sum(), sum(ledger._net_ledger(gaps, 2)), rtol=0, atol=1e-8)


def test_monthly_chargeback_is_spread_over_original_surplus():
    """A quarter's charge-back is spread over the months of the surplus it reduces, by their surplus."""
    d = pd.DataFrame({"dest": ["A"] * 4, "code_hs6": ["847150"] * 4,
                      "period": ["2021-10", "2021-11", "2021-12", "2022-01"],
                      "value_usd": [160.0, 140.0, 100.0, 30.0], "baseline": [100.0] * 4})
    np.testing.assert_allclose(ledger._ledger_excess(d, "dest", retrospective=True), [18, 12, 0, 0])


def test_missing_quarters_still_age_the_ledger():
    """Quarters without rows still count toward expiry: the ledger runs on calendar quarters."""
    d = pd.DataFrame({"dest": ["A", "A"], "code_hs6": ["847150", "847150"],
                      "period": ["2021-01", "2021-10"],
                      "value_usd": [200.0, 0.0], "baseline": [100.0, 100.0]})
    np.testing.assert_allclose(ledger._ledger_excess(d, "dest", retrospective=True), [100, 0])


def test_partial_history_leaves_unwarmed_corridor_on_its_2022_basis():
    """A corridor without 2021 rows keeps its 2022-mean baseline and excess when others carry history."""
    periods = pd.period_range("2021-01", "2022-12", freq="M").astype(str)
    exporter = pd.DataFrame({"dest": "Brazil", "code_hs6": "847150", "period": periods,
                             "value_usd": 120.0})
    taiwan = exporter[exporter.period.ge("2022-01")].assign(dest=RESIDUAL)
    old = ledger.corridor_chiponly(pd.concat([exporter[exporter.period.ge("2022-01")], taiwan]),
                                   "dest", historical=True)
    new = ledger.corridor_chiponly(pd.concat([exporter, taiwan]), "dest", historical=True)
    fields = ["period", "baseline", "excess", "chiponly"]
    pd.testing.assert_frame_equal(old[old.dest.eq(RESIDUAL)][fields].reset_index(drop=True),
                                  new[new.dest.eq(RESIDUAL)][fields].reset_index(drop=True))
    assert not new.period.lt("2022-01").any()


def test_complete_history_anchors_the_baseline_and_drops_warmup_output():
    """With full 2021 history the baseline is the 2021 level grown from its 2021 mean; 2021 is not output."""
    periods = pd.period_range("2021-01", "2022-12", freq="M").astype(str)
    d = pd.DataFrame({"dest": RESIDUAL, "code_hs6": "847150", "period": periods, "value_usd": 120.0})
    co = ledger.corridor_chiponly(d, "dest", historical=True).sort_values("period")
    assert not co.period.lt("2022-01").any()
    growth = 1.0 + codes.NONAI_GROWTH["systems"]
    gf = growth ** np.array([codes.years_since_mid2022(p) for p in periods])
    np.testing.assert_allclose(co.baseline, 120.0 * gf[12:] / gf[:12].mean())
