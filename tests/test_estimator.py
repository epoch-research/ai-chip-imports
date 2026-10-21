"""Estimator guards: the universe must carry a complete 2021 history."""
import pandas as pd
import pytest

from ai_chip_imports import estimator

RESIDUAL = "(residual: no importer leg)"


def year(dest, y, value=1.0):
    return pd.DataFrame({"dest": dest, "code_hs6": "847150",
                         "period": [f"{y}-{m:02}" for m in range(1, 13)], "value_usd": value})


def test_universe_without_2021_history_is_rejected():
    """A universe with no 2021 rows cannot anchor the baseline and is rejected."""
    with pytest.raises(ValueError, match="lacks its 2021 history"):
        estimator._check_history(year("Brazil", 2022))


def test_destination_without_2021_history_is_rejected():
    """Every counted destination x code needs all twelve 2021 months, the residual bucket included."""
    h = year("Brazil", 2021)
    fa = pd.concat([h, h.iloc[:1].assign(dest=RESIDUAL, period="2022-01")])
    with pytest.raises(ValueError, match="incomplete 2021 history"):
        estimator._check_history(fa)


def test_complete_history_is_accepted():
    """A universe whose every corridor carries twelve 2021 months passes the guard."""
    fa = pd.concat([year("Brazil", 2021), year("Brazil", 2022), year(RESIDUAL, 2021), year(RESIDUAL, 2022)])
    estimator._check_history(fa)
