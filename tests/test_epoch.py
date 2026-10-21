"""Epoch's Chip Sales timelines: cost imputation for rows with units but no cost."""
import numpy as np
import pandas as pd
import pytest

from ai_chip_imports import epoch

COLUMNS = ["Name", "Chip manufacturer", "Start date", "Number of Units",
           "Compute estimate in H100e (median)", "Chip type (linked)", "Chip Cost (USD)",
           "Cost Estimate (USD)"]


@pytest.fixture(autouse=True)
def catalogue(tmp_path, monkeypatch):
    """A chip_types.csv that prices TPU v8i at $17,500 and leaves Trainium1 unpriced."""
    pd.DataFrame({"Name": ["TPU v8i", "Trainium1"],
                  "Cost per chip (approx.)": [17500.0, np.nan]}).to_csv(tmp_path / "chip_types.csv",
                                                                       index=False)
    monkeypatch.setattr(epoch, "EPOCH_DIR", str(tmp_path))


def timeline(rows):
    t = pd.DataFrame(rows, columns=COLUMNS)
    t["Cost Estimate (USD)"] = t["Cost Estimate (USD)"].astype("int64")
    return t


def test_missing_cost_takes_catalogue_price_then_designer_rate():
    """A zero-cost row is priced from the catalogue, else at its designer's $/H100e that quarter."""
    t = timeline([
        ["Q1 2024 Trainium2", "Amazon", "2024-01-01", 100, 50.0, "Trainium2", 4000.0, 400000],
        ["Q1 2024 Trainium1/Inferentia", "Amazon", "2024-01-01", 1000, 200.0, "Trainium1", np.nan, 0],
        ["Q2 2026 - TPU v8i", "Google", "2026-04-01", 10, 50.0, "TPU v8i", np.nan, 0],
    ])
    out = epoch.impute_missing_costs(t).set_index("Chip type (linked)")
    assert out.loc["TPU v8i", "Cost Estimate (USD)"] == 10 * 17500
    assert out.loc["TPU v8i", "cost_source"] == "catalogue"
    # Amazon's priced rate that quarter: $400k / 50 H100e = $8k per H100e, x 200 H100e
    assert out.loc["Trainium1", "Cost Estimate (USD)"] == 1_600_000
    assert out.loc["Trainium1", "cost_source"] == "designer_h100e"
    assert out.loc["Trainium2", "Cost Estimate (USD)"] == 400000
    assert not out.loc["Trainium2", "cost_imputed"]
    assert out["Cost Estimate (USD)"].dtype.kind == "i"


def test_designer_without_a_priced_row_takes_its_group_rate():
    """A designer with no priced row that quarter falls back to its group's $/H100e."""
    t = timeline([
        ["Q1 2024 Trainium2", "Amazon", "2024-01-01", 100, 50.0, "Trainium2", 4000.0, 400000],
        ["Q1 2024 TPU v5e", "Google", "2024-01-01", 30, 10.0, "TPU v5e", np.nan, 0],
    ])
    out = epoch.impute_missing_costs(t).set_index("Chip type (linked)")
    assert out.loc["TPU v5e", "Cost Estimate (USD)"] == 80000
    assert out.loc["TPU v5e", "cost_source"] == "group_h100e"
    assert out.loc["TPU v5e", "Chip Cost (USD)"] == pytest.approx(80000 / 30)
