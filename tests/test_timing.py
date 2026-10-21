"""Timing: calendar customs quarters, the sales-to-arrival lag, and conservation of supply."""
import numpy as np
import pandas as pd
import pytest

from ai_chip_imports import timing


def months_2024():
    return pd.DataFrame({"period": pd.period_range("2024-01", "2024-12", freq="M").astype(str),
                         "dest": "A", "dollars": range(1, 13)})


def test_calendar_totals_and_partial_frontier():
    """Months sum into calendar quarters; a trailing partial quarter is not reported."""
    f = pd.DataFrame({"period": pd.period_range("2024-01", "2024-07", freq="M").astype(str),
                      "value": [1, 2, 3, 4, 5, 6, 999]})
    assert timing.aggregate(f, f.value).to_dict() == {"2024Q1": 6, "2024Q2": 15}
    meta = timing.metadata(["2024Q1"]).iloc[0]
    assert meta.sales_window_start == "2023-11"
    assert meta.sales_window_end == "2024-01"
    assert meta.arrival_window_end == "2024-03"


def test_missing_month_is_not_zero_filled():
    """A month missing from the whole frame drops its quarter rather than reading as zero."""
    f = months_2024().query("period != '2024-04'")
    s = timing.aggregate(f, f.dollars)
    assert "2024Q2" not in s.index
    assert {"2024Q1", "2024Q3", "2024Q4"} <= set(s.index)


def test_sparse_destination_contributes_zero_for_months_it_lacks():
    """A destination with a single month still lands in that quarter, alongside the others."""
    f = months_2024()
    f = pd.concat([f, pd.DataFrame({"period": ["2024-04"], "dest": ["B"], "dollars": [100]})])
    s = timing.aggregate(f, f.dollars, by=("dest",))
    assert s.loc[("B", "2024Q2")] == 100
    assert s.xs("2024Q2", level="q").sum() == 4 + 5 + 6 + 100


def test_sales_interpolation_and_missing_quarters():
    """Arrivals blend adjacent sales quarters; a missing or incomplete source quarter gives NaN."""
    s = pd.Series([300.0, 600.0, 1200.0], index=["2023Q4", "2024Q1", "2024Q3"])
    out = timing.interpolate_sales(s)
    assert out["2024Q1"] == pytest.approx(400.0)
    assert pd.isna(out["2024Q2"])
    assert pd.isna(out["2024Q3"])
    s.attrs["incomplete_quarters"] = ["2024Q1"]
    assert pd.isna(timing.matched_sales(s)["2024Q1"])


def test_match_uses_sales_frontier_and_excludes_incomplete_sources():
    """Matched sales stop where they would draw on an incomplete Epoch quarter."""
    e = pd.DataFrame({"quarter": ["2026Q1", "2026Q2", "2026Q3"],
                      "epoch_cost_usd": [300.0, 600.0, 900.0],
                      "epoch_incomplete": [False, False, True]})
    matched = timing.matched_sales(timing.sales_from_epoch(e))
    assert matched["2026Q2"] == pytest.approx(400.0)
    assert pd.isna(matched["2026Q3"])


def model_sales(**columns):
    d = pd.DataFrame({"designer": "Nvidia", "pool": "row", "cn_spec": False, **columns})
    for suffix in ["lo", "hi"]:
        for field in ["units", "h100e"]:
            d[field + "_" + suffix] = d[field]
    return d


def test_model_conservation_prices_and_launch_boundary():
    """Moving sales to arrivals conserves each model's totals, recomputes ASP and flags partial sources."""
    d = model_sales(model=["A", "A", "B"], q=["2023Q4", "2024Q1", "2024Q1"],
                    incomplete=[False, False, True], units=[300.0, 600.0, 90.0],
                    h100e=[150.0, 300.0, 180.0], cost=[3000.0, 12000.0, 900.0])
    out = timing.arrival_models(d)
    np.testing.assert_allclose(out.groupby("model")[["units", "h100e", "cost"]].sum(),
                               d.groupby("model")[["units", "h100e", "cost"]].sum())
    a = out.set_index(["model", "q"]).loc[("A", "2024Q1")]
    assert a.units == pytest.approx(400.0)
    assert a.asp == pytest.approx(15.0)
    assert ("B", "2023Q4") not in out.set_index(["model", "q"]).index
    assert out[out.q.eq("2024Q2")].partial_sales_coverage.all()
    assert out[out.q.eq("2024Q1")].partial_sales_coverage.all()


def test_other_lags_conserve_supply_and_shift_sources():
    """Any lag splits each sales quarter between two arrival quarters and conserves supply."""
    s = pd.Series([300.0, 600.0, 900.0, 1200.0], index=["2023Q3", "2023Q4", "2024Q1", "2024Q2"])
    one = timing.interpolate_sales(s, months=1)           # 1/3 of q-1 + 2/3 of q
    assert one["2024Q1"] == pytest.approx(600.0 / 3 + 900.0 * 2 / 3)
    four = timing.interpolate_sales(s, months=4)          # 2/3 of q-1 + 1/3 of q-2
    assert four["2024Q2"] == pytest.approx(900.0 * 2 / 3 + 600.0 / 3)
    assert pd.isna(four["2023Q4"])
    pd.testing.assert_series_equal(timing.interpolate_sales(s, months=3), s.shift(1))
    d = model_sales(model="A", q=["2023Q4", "2024Q1"], incomplete=False, units=[300.0, 600.0],
                    h100e=[150.0, 300.0], cost=[3000.0, 12000.0])
    for months in (1, 4, 5):
        out = timing.arrival_models(d, months=months)
        np.testing.assert_allclose(out[["units", "h100e", "cost"]].sum(),
                                   d[["units", "h100e", "cost"]].sum())
    out = timing.arrival_models(d, months=4).set_index("q")
    assert out.loc["2024Q2", "units"] == pytest.approx(600.0 * 2 / 3 + 300.0 / 3)
    assert sorted(out.index) == ["2024Q1", "2024Q2", "2024Q3"]
