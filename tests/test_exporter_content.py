"""Exporter-side content for the thin tier: field coverage, donor imputation and conservation."""
import pandas as pd
import pytest

from ai_chip_imports import exporter_content


class Model:
    """Epoch references for one quarter: a 0.73 system mix and an eligible ASP of ``asp``."""

    def __init__(self, asp=10000.0):
        self._asp = asp

    def mix(self, side, rack=None):
        return pd.Series({"2025Q1": 0.73})

    def asp(self, side):
        return pd.Series({"2025Q1": self._asp})


def rows():
    """One exporter's server rows to four destinations: two at $1k/unit, one at $10k/unit, one
    with no physical field."""
    return pd.DataFrame(dict(
        exporter=["X"] * 4, dest=["A", "B", "C", "D"], taiwan=[False] * 4, counted=[True] * 4,
        code_hs6=["847150"] * 4, period=["2025-01"] * 4, value_usd=[10000.0] * 4,
        qty=[10.0, 10.0, 1.0, 0.0], qty_value=[10000.0, 10000.0, 10000.0, 0.0],
        weight_kg=[0.0] * 4, kg_value=[0.0] * 4))


def test_donors_include_ordinary_goods_and_missingness_is_visible():
    """Low-content rows are donors like any other; an unread row shows zero coverage and a donor range."""
    d = exporter_content.classify(rows(), Model()).set_index("dest")
    assert d.loc["D", "content"] == pytest.approx((0.05 + 0.05 + 0.93) / 3)
    assert d.loc["D", "coverage"] == 0
    assert d.loc["A", "content"] == 0.05
    assert d.loc["D", "content_low"] == 0.05
    assert d.loc["D", "content_high"] == 0.93
    assert d.loc["A", "content_low"] == d.loc["A", "content_high"]


def test_uncovered_fraction_is_imputed_not_treated_as_measured():
    """Value outside a row's field coverage takes the donor content, not the row's own reading."""
    r = rows()
    r.loc[3, ["qty", "qty_value"]] = [9.0, 9000.0]
    d = exporter_content.classify(r, Model()).set_index("dest")
    assert d.loc["D", "coverage"] == pytest.approx(0.9)
    assert d.loc["D", "content"] > 0.05


def test_boards_cannot_be_classified_from_piece_count():
    """Board rows are read by weight only; a board row with pieces but no weight is unread."""
    r = rows()
    r["code_hs6"] = "847330"
    r.loc[:2, "weight_kg"] = 100
    r.loc[:2, "kg_value"] = 10000
    d = exporter_content.classify(r, Model()).set_index("dest")
    assert d.loc["A", "field"] == "weight_kg"
    assert d.loc["D", "field"] == "missing"
    assert d.loc["D", "content"] == pytest.approx(0.05)


def test_no_donors_fails_instead_of_assuming_accelerator_systems():
    """Without any observed donor the classification fails rather than assume system content."""
    r = rows()
    r["qty"] = 0.0
    r["qty_value"] = 0.0
    with pytest.raises(ValueError, match="No empirical"):
        exporter_content.classify(r, Model())


def test_sparse_exporter_borrows_same_code_quarter_not_system_constant():
    """An exporter with too few destinations borrows the code x quarter pool of other exporters."""
    r = rows()
    r.loc[3, "exporter"] = "Sparse"
    d = exporter_content.classify(r, Model()).set_index("dest")
    assert d.loc["D", "donor_level"] == "code_hs6/q"
    assert d.loc["D", "content"] == pytest.approx((0.05 + 0.05 + 0.93) / 3)


def test_eligible_price_reaches_exporter_classification():
    """A higher eligible accelerator price lowers the content read from, and imputed to, rows."""
    a = exporter_content.classify(rows(), Model())
    b = exporter_content.classify(rows(), Model(asp=100000.0))
    assert b.loc[2, "content"] < a.loc[2, "content"]
    assert b.loc[3, "content"] < a.loc[3, "content"]


def test_china_reference_is_not_borrowed_for_row_destinations():
    """China's rows, priced against the China-specific mix, are never donors for other destinations."""
    r = rows()
    extra = r.iloc[[2]].copy()
    extra["dest"] = "China"
    extra["value_usd"] = 1e9
    extra["qty_value"] = 1e9
    extra["qty"] = 1e5
    d = exporter_content.classify(pd.concat([r, extra], ignore_index=True), Model()).set_index("dest")
    assert d.loc["D", "content"] == pytest.approx((0.05 + 0.05 + 0.93) / 3)
    assert not d.loc["China", "eligible_donor"]


def test_pooling_conserves_trade_and_imputed_dollars():
    """Pooling rows into a destination conserves trade value, chip dollars and the imputed share."""
    d = exporter_content.classify(rows(), Model())
    d["dest"] = "A"
    a = exporter_content.aggregate(d).iloc[0]
    assert a.value_usd == 40000.0
    assert a.content * 40000 == pytest.approx((d.value_usd * d.content).sum())
    assert a.imputed_value_fraction == 0.25
    # Applying this weight to one aggregate excess cannot manufacture excess.
    assert 1000 * a.content <= 1000
