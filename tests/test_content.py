"""Accelerator content: quarterly smoothing, monthly readings and piece-covered unit values."""
import pandas as pd
import pytest

from ai_chip_imports import content


def bare_model():
    """A ContentModel shell with no evidence, for exercising one method at a time."""
    return content.ContentModel.__new__(content.ContentModel)


def test_median_treats_a_missing_quarter_as_a_gap():
    """The centered median skips a missing quarter instead of reaching to the next one present."""
    c = pd.Series([0.1, 0.9, 0.2], index=[0, 1, 2])
    out = content._median3(c, ["2024Q1", "2024Q2", "2024Q4"])      # 2024Q3 missing
    assert out[1] == pytest.approx(0.5)       # median of 2024Q1-Q3 = {0.1, 0.9}
    assert out[2] == pytest.approx(0.2)       # 2024Q4's neighbours: Q3 (gap) and Q5 (none)


def test_months_are_independent_and_gates_remove_old_readings():
    """Each month passes the excess and coverage gates on its own; a pair's old readings are replaced."""
    cm = bare_model()
    cm.monthly_cells = {("X", "847150", "2025-02"): 0.9}
    f = pd.DataFrame(dict(dest=["X"] * 4, code_hs6=["847150"] * 4,
                          period=["2025-01", "2025-02", "2025-03", "2025-04"],
                          excess=[30e6, 30e6, 1e6, 30e6], coverage=[1, 0.8, 1, 1],
                          c=[0.05, 0.93, 0.93, 0.93]))
    cm.install_monthly(f)
    assert cm.monthly_cells == {("X", "847150", "2025-01"): 0.05, ("X", "847150", "2025-04"): 0.93}


def test_zero_content_is_in_the_denominator_and_other_pairs_are_kept():
    """Zero-content cells dilute the month's reading; readings of other pairs are untouched."""
    cm = bare_model()
    cm.monthly_cells = {("X", "847150", "2025-01"): 0.7}
    f = pd.DataFrame(dict(dest=["United States"] * 2, code_hs6=["847330"] * 2,
                          period=["2025-01"] * 2, excess=[90e6, 10e6], coverage=[1, 1], c=[0, 0.93]))
    cm.install_monthly(f)
    assert cm.monthly_cells[("United States", "847330", "2025-01")] == pytest.approx(0.093)
    assert cm.monthly_cells[("X", "847150", "2025-01")] == 0.7


def piece_rows():
    """An HS-6 line whose pieces sit on 60% of its value (the rest is a kilogram-quantity line)."""
    return pd.DataFrame({
        "rep_c": ["Singapore"] * 2, "partner": ["Taiwan"] * 2, "code_hs6": ["847180"] * 2,
        "period": ["2025-04", "2025-05"], "value_usd": [100.0, 100.0], "qty": [2.0, 2.0],
        "value_qtycov": [60.0, 60.0], "weight_kg": [0.0, 0.0]})


def piece_model(rows):
    """A ContentModel shell that classifies ``rows`` as Singapore's 847180 line by pieces."""
    cm = bare_model()
    cm.bindex = None
    cm._cell_rows = lambda res, dest, code, cn_excluded: rows
    cm._side = lambda dest: "row"
    cm.mix = lambda side, rack=None: {"2025Q2": 0.7}
    cm.asp = lambda side: {"2025Q2": 10000.0}
    cm.monthly_cells = {}
    cm._cellcls_pairs = {("Singapore", "847180"): ("qty", 1, 0.5)}
    cm._native_monthly_frame = pd.DataFrame()
    return cm


def test_coverage_gate_reads_piece_covered_value():
    """Pieces classify a line only where the value they cover reaches the coverage threshold."""
    rows = piece_rows()
    assert content.ContentModel._cell_field(rows, "847180") is None        # 60% < 90%
    assert content.ContentModel._cell_field(rows.assign(value_qtycov=95.0), "847180") == "qty"


def test_unit_value_on_covered_value_and_excess_on_whole_value():
    """$/unit is read over the piece-covered value; the cell's excess stays on its whole value."""
    oq = piece_model(piece_rows())._cell_table(None, "Singapore", "847180", "qty", ())
    assert "vq" not in oq.columns
    assert oq.uv[0] == pytest.approx(30.0)          # 120 covered / 4 pieces in the quarter
    assert oq.excess[0] == pytest.approx(200.0)     # no 2022 pace: the whole value


def test_books_without_the_coverage_column_keep_the_whole_row_reading():
    """A book without piece-covered values reads pieces against the whole row value."""
    rows = piece_rows().drop(columns="value_qtycov")
    assert content.ContentModel._cell_field(rows, "847180") == "qty"
    oq = piece_model(rows)._cell_table(None, "Singapore", "847180", "qty", ())
    assert oq.uv[0] == pytest.approx(50.0)


def test_monthly_gate_coverage():
    """Monthly readings carry the piece-covered share of value as their coverage."""
    cm = piece_model(piece_rows())
    cm._monthly_classifier(None, ())
    assert cm._monthly_frame.coverage.tolist() == [0.6, 0.6]
    assert cm._monthly_frame.uv[0] == pytest.approx(30.0)
