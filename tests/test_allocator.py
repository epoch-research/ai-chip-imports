"""Allocator: carried-forward shares for incomplete books and coherent country bands."""
import pandas as pd
import pytest

from ai_chip_imports import allocator


def test_lagging_destination_keeps_its_previous_share():
    """A destination whose book stops early keeps its last complete share; others keep their dollars."""
    m = pd.DataFrame({"United States": [60.0, 120.0], "Saudi Arabia": [20.0, 5.0],
                      "Brazil": [20.0, 40.0]}, index=["2026Q1", "2026Q2"])
    out = allocator.carry_forward_shares(m, {("Saudi Arabia", "2026Q2")})
    q2 = out.loc["2026Q2"]
    assert q2["Saudi Arabia"] / q2.sum() == pytest.approx(0.20)
    assert q2["United States"] == 120.0
    assert q2["Brazil"] == 40.0
    assert out.attrs["carried_forward"] == [("Saudi Arabia", "2026Q2")]
    pd.testing.assert_series_equal(out.loc["2026Q1"], m.loc["2026Q1"])


def test_thin_mark_carries_every_thin_destination():
    """An incomplete thin tier carries forward the share of every destination without a book."""
    m = pd.DataFrame({"United States": [80.0, 160.0], "Brazil": [10.0, 1.0], "Chile": [10.0, 1.0]},
                     index=["2026Q1", "2026Q2"])
    out = allocator.carry_forward_shares(m, {(allocator.THIN_MARK, "2026Q2")})
    q2 = out.loc["2026Q2"]
    assert q2["Brazil"] / q2.sum() == pytest.approx(0.10)
    assert q2["Chile"] / q2.sum() == pytest.approx(0.10)
    assert q2["United States"] == 160.0


def test_bands_take_the_envelope_of_country_totals():
    """Country bands span variants' destination totals, not sums of per-model extremes."""
    def variant(us_a, us_b):
        idx = pd.MultiIndex.from_tuples([("US", "2026Q1", "A"), ("US", "2026Q1", "B"),
                                         ("CN", "2026Q1", "A"), ("CN", "2026Q1", "B")],
                                        names=["dest", "q", "model"])
        d = pd.DataFrame({"units": [us_a, us_b, 10 - us_a, 10 - us_b]}, index=idx)
        for c in ["units_lo", "units_hi", "h100e", "h100e_lo", "h100e_hi"]:
            d[c] = d["units"]
        return d
    flows = {"h": variant(5.0, 5.0), "v1": variant(8.0, 2.0), "v2": variant(2.0, 8.0)}
    stocks = {k: v.swaplevel("q", "model") for k, v in flows.items()}
    b = allocator.country_bands(flows, stocks, "h").set_index(["dest", "q"])
    # every variant gives the US 10 units: summing per-cell extremes would claim 4..16
    assert b.loc[("US", "2026Q1"), "units_lo"] == 10.0
    assert b.loc[("US", "2026Q1"), "units_hi"] == 10.0
    assert b.loc[("World", "2026Q1"), "units_cum"] == 20.0
