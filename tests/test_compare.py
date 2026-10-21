"""Run comparison: verbatim, order-free cell matching between a reference and a candidate run."""
import gzip

import pandas as pd

from ai_chip_imports import compare


def frame(rows, columns=("dest", "q", "usd")):
    """A canonical frame: every cell as CSV text."""
    return pd.DataFrame(rows, columns=list(columns), dtype=str)


REF = frame([["Brazil", "2024Q1", "1.0"], ["Chile", "2024Q1", "2.0"], ["Peru", "2024Q2", "3.0"]])


def test_identical_frames_match():
    """Identical frames match and the report gives their shape."""
    ok, lines = compare.compare_frames(REF, REF.copy())
    assert ok
    assert lines == ["  identical: 3 rows x 3 columns"]


def test_row_and_column_order_do_not_matter():
    """Rows are compared as a multiset and columns by name."""
    shuffled = REF.iloc[[2, 0, 1]][["usd", "dest", "q"]].reset_index(drop=True)
    ok, _ = compare.compare_frames(REF, shuffled)
    assert ok


def test_a_changed_cell_is_reported_with_its_numeric_gap():
    """One changed value fails the comparison and is named with its column, gap and example."""
    changed = REF.copy()
    changed.loc[1, "usd"] = "2.5"
    ok, lines = compare.compare_frames(REF, changed)
    assert not ok
    assert lines == ["  column 'usd': 1 cells differ (max |diff| 0.5), e.g. '2.0' -> '2.5'"]


def test_text_differences_count_even_when_numerically_equal():
    """Cells are compared as text, so a reformatted number is a difference."""
    reformatted = REF.copy()
    reformatted.loc[0, "usd"] = "1.00"
    ok, _ = compare.compare_frames(REF, reformatted)
    assert not ok


def test_row_count_mismatch_fails():
    """A candidate with fewer rows fails outside subset mode."""
    ok, lines = compare.compare_frames(REF, REF.iloc[:2])
    assert not ok
    assert lines == ["  row count: reference 3 vs candidate 2"]


def test_column_differences_fail_unless_comparing_shared_columns():
    """An extra column fails by default; shared-columns mode compares the rest and still reports it."""
    wider = REF.assign(note="x")
    ok, lines = compare.compare_frames(REF, wider)
    assert not ok
    assert "  columns only in candidate: ['note']" in lines
    ok, lines = compare.compare_frames(REF, wider, shared_columns=True)
    assert ok
    assert lines == ["  columns only in candidate: ['note']", "  identical: 3 rows x 3 columns"]


def test_subset_mode_accepts_reproduced_rows_only():
    """Subset mode accepts a candidate whose rows all appear in the reference, and no other."""
    ok, lines = compare.compare_frames(REF, REF.iloc[[2, 0]], subset=True)
    assert ok
    assert lines == ["  2 of 3 reference rows reproduced exactly"]
    extra = pd.concat([REF.iloc[[0]], frame([["Peru", "2024Q2", "3.5"]])], ignore_index=True)
    ok, lines = compare.compare_frames(REF, extra, subset=True)
    assert not ok
    assert lines == ["  1 candidate rows not in the reference"]


def test_newest_stamp_wins_and_dated_outputs_beat_undated_archives(tmp_path):
    """Each output name resolves to its newest dated file; an undated .csv.gz is the fallback."""
    for name in ["flows_20260101.csv", "flows_20260301.csv", "flows.csv.gz", "stocks.csv.gz",
                 "notes.txt", "flows_2026.csv"]:
        (tmp_path / name).write_text("a\n1\n")
    assert compare.newest_outputs(tmp_path) == {"flows": tmp_path / "flows_20260301.csv",
                                                "stocks": tmp_path / "stocks.csv.gz"}


def test_read_canonical_keeps_text_and_sorts(tmp_path):
    """CSV cells are read verbatim (no float round trip, empty stays empty), sorted by column and row."""
    path = tmp_path / "flows.csv.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("usd,dest\n2.50,Chile\n,Brazil\n1e3,Argentina\n")
    d = compare.read_canonical(path)
    assert list(d.columns) == ["dest", "usd"]
    assert d.values.tolist() == [["Argentina", "1e3"], ["Brazil", ""], ["Chile", "2.50"]]


def test_main_exit_status_reflects_the_comparison(tmp_path, capsys):
    """The command exits 0 when every output matches and 1 when any differs or is missing."""
    ref, same, diff = (tmp_path / n for n in ("ref", "same", "diff"))
    for d in (ref, same, diff):
        d.mkdir()
    (ref / "flows_20260101.csv").write_text("dest,usd\nBrazil,1.0\nChile,2.0\n")
    (same / "flows_20260301.csv").write_text("dest,usd\nChile,2.0\nBrazil,1.0\n")
    (diff / "flows_20260301.csv").write_text("dest,usd\nChile,2.0\nBrazil,1.5\n")
    assert compare.main([str(ref), str(same)]) == 0
    assert compare.main([str(ref), str(diff)]) == 1
    assert compare.main([str(ref), str(same), "--files", "stocks"]) == 1
    out = capsys.readouterr().out
    assert "MATCH  flows" in out and "DIFFER flows" in out and "MISSING stocks" in out
