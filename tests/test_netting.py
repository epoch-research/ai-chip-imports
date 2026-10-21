"""Netting: the negative carry and Malaysia's deduction for undeclared server exports to China."""
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_chip_imports import netting


def books(rows):
    return pd.DataFrame(rows, columns=["rep_c", "flow", "partner", "code_hs6", "period", "value_usd"])


def test_carry_runs_over_the_union_of_months_and_returns_import_cells():
    """A deduction in a month without imports is carried into the next import month, not dropped."""
    M = pd.DataFrame({"code_hs6": ["847150"] * 2, "period": ["2024-01", "2024-03"],
                      "value_usd": [10.0, 50.0]})
    OUT = pd.DataFrame({"code_hs6": ["847150"] * 2, "period": ["2024-01", "2024-02"],
                        "out": [30.0, 20.0]})
    net = netting._carry_net(M, OUT).set_index("period").value_usd
    assert net.to_dict() == {"2024-01": 0.0, "2024-03": 10.0}


def test_quarterly_gap_not_sum_of_positive_monthly_gaps():
    """The mirror gap is quarterly, server codes only, China's imports from Malaysia only, spread by month."""
    r = books([
        ["China", "M", "Malaysia", "847150", "2021-01", 80.0],
        ["China", "M", "Malaysia", "847150", "2021-02", 20.0],
        ["Malaysia", "X", "China", "847150", "2021-02", 60.0],
        # Neither boards, another origin, nor an export to another destination counts.
        ["China", "M", "Malaysia", "847330", "2021-01", 900.0],
        ["China", "M", "Singapore", "847150", "2021-01", 900.0],
        ["Malaysia", "X", "Singapore", "847150", "2021-01", 900.0],
    ])
    got = netting.malaysia_server_mirror(r, 0.5).set_index("period")["out"]
    assert got.to_dict() == {"2021-01": 16.0, "2021-02": 4.0}
    assert netting.malaysia_server_mirror(r, 1).out.sum() == 40.0
    assert netting.malaysia_server_mirror(r, 0).empty


def test_negative_gap_never_adds_imports():
    """Declared exports above China's imports give no deduction; shares outside [0, 1] are rejected."""
    r = books([
        ["China", "M", "Malaysia", "847149", "2024-01", 20.0],
        ["Malaysia", "X", "China", "847149", "2024-03", 30.0],
    ])
    assert netting.malaysia_server_mirror(r, 1).out.sum() == 0.0
    for value in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            netting.malaysia_server_mirror(r, value)


def test_declared_split_stands_in_where_total_exports_are_absent():
    """Months without Malaysia's total exports read DX + RX; months with it never add the split."""
    r = books([
        ["China", "M", "Malaysia", "847150", "2021-01", 100.0],
        ["Malaysia", "DX", "China", "847150", "2021-01", 30.0],
        ["Malaysia", "RX", "China", "847150", "2021-01", 10.0],
        ["China", "M", "Malaysia", "847150", "2024-01", 100.0],
        ["Malaysia", "X", "China", "847150", "2024-01", 50.0],
        ["Malaysia", "DX", "China", "847150", "2024-01", 40.0],
        ["Malaysia", "RX", "China", "847150", "2024-01", 10.0],
    ])
    got = netting.malaysia_server_mirror(r, 1).set_index("period")["out"]
    assert got.to_dict() == {"2021-01": 60.0, "2024-01": 50.0}   # never X + DX + RX


def test_deduction_enters_malaysias_carry_ledger_only():
    """The mirror deduction runs through Malaysia's negative carry; China's book is unaffected."""
    r = books([
        ["Malaysia", "M", "Taiwan", "847150", "2024-01", 10.0],
        ["Malaysia", "M", "Taiwan", "847150", "2024-02", 100.0],
        ["China", "M", "Malaysia", "847150", "2024-01", 60.0],
    ])
    r["qty"] = 0.0
    r["weight_kg"] = 0.0
    r["par_c"] = r.partner
    shares = SimpleNamespace(leg_rx={})

    def run(leg, share):
        out = netting._un_leg_rows(r, None, shares, leg, (), (), lambda *a: 0.4, [], mirror_share=share)
        return out.set_index("period").value_usd

    assert run("Malaysia", 0).to_dict() == {"2024-01": 10.0, "2024-02": 100.0}
    assert run("Malaysia", 0.5).to_dict() == {"2024-01": 0.0, "2024-02": 80.0}
    assert run("China", 0.5).to_dict() == {"2024-01": 60.0}
    pd.testing.assert_series_equal(run("China", 0.5), run("China", 0))
