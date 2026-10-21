"""The UK's 2021 imports from the EU, rebuilt from EU members' exports."""
import numpy as np
import pandas as pd
import pytest

from ai_chip_imports.prepare import uk_reconstruction as uk


def test_uk_reconstruction_pools_machines_and_preserves_other_flows():
    """Machine codes share one calibration ratio; only EU-origin 2021 imports change; gaps fail."""
    calibration = pd.period_range("2022-07", "2023-06", freq="M").astype(str)
    imp = pd.DataFrame([dict(period=p, code_hs6=c, value_usd=200.0 if c == "847149" else 100.0)
                        for p in calibration for c in uk.GROUPS])
    exp = imp.assign(value_usd=100.0)
    rows = [dict(period=f"2021-{m:02}", code_hs6=c, value_usd=10.0, qty=2.0,
                 weight_kg=3.0, value_qtycov=10.0, flow=f, par_c=partner)
            for m in range(1, 13) for c in uk.GROUPS
            for f, partner in [("M", "France"), ("M", "United States"), ("X", "France")]]
    h = pd.DataFrame(rows)
    x = h[h.flow.eq("M") & h.par_c.eq("France")].set_index(["code_hs6", "period"]).value_usd * 2
    corrected, ratios, _ = uk.reconstruct(h, imp, exp, x, ["France"])
    assert ratios["machines"] == 1.5
    eligible = h.flow.eq("M") & h.par_c.eq("France")
    np.testing.assert_allclose(
        corrected.loc[eligible, "value_usd"],
        h.loc[eligible, "code_hs6"].map(lambda c: 30.0 if uk.GROUPS[c] == "machines" else 20.0))
    pd.testing.assert_frame_equal(corrected[~eligible], h[~eligible])
    with pytest.raises(ValueError, match="Incomplete or invalid"):
        uk.reconstruct(h, imp[imp.period.ne("2022-07")], exp, x, ["France"])
