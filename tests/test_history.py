"""The 2021 reference year: partner alignment, thin-tier export history, padding, baselines, FOB basis."""
import numpy as np
import pandas as pd
import pytest

from ai_chip_imports import fob, history, netting

SPLIT_HUBS = ("United States", "Malaysia", "Singapore", "Hong Kong")
EXPORTERS = SPLIT_HUBS + ("United Kingdom", "Indonesia", "India")


def test_historical_names_match_native_destinations_without_changing_current_rows():
    """2021 export partners take the label their exporter uses today; the current book is untouched."""
    h = pd.DataFrame({"rep_c": ["United States"] * 3, "par_c": [np.nan] * 3,
                      "partner": ["China, Macao SAR", "Dominican Rep.", "Viet Nam"]})
    current = pd.DataFrame({"rep_c": ["United States"] * 3, "par_c": [np.nan] * 3,
                            "partner": ["MACAU", "DOMINICAN REPUBLIC", "VIETNAM"], "flow": "X"})
    original = current.copy(deep=True)
    aligned = history.align_export_partners(h, current)
    assert aligned.partner.tolist() == ["Macau", "Dominican Republic", "Vietnam"]
    assert "China" not in aligned.partner.tolist()
    pd.testing.assert_frame_equal(current, original)


def export_history():
    """Every thin-tier exporter's 2021 exports to Brazil: the split hubs as RX ($40/month) and
    DX ($60/month), the others as total X ($60/month)."""
    rows = []
    for leg in EXPORTERS:
        flows = ["RX", "DX"] if leg in SPLIT_HUBS else ["X"]
        for flow in flows:
            for month in range(1, 13):
                rows.append(dict(reporter=leg, rep_c=leg, partner="Brazil", par_c=np.nan,
                                 code_hs6="847150", period=f"2021-{month:02d}", flow=flow,
                                 value_usd=40.0 if flow == "RX" else 60.0,
                                 weight_kg=1.0, qty=1.0, value_qtycov=0.0))
    return pd.DataFrame(rows)


def current_book(h):
    """Malaysia's current exports: one new destination (Norway) at $200/month through 2022."""
    current = h[h.rep_c.eq("Malaysia") & h.flow.eq("DX")].copy()
    current["flow"], current["partner"] = "X", "Norway"
    current["period"] = current.period.str.replace("2021", "2022")
    current["value_usd"] = 200.0
    return current


@pytest.fixture
def no_thin_cache(monkeypatch):
    monkeypatch.setattr(history, "_THIN_SOURCE", None)
    monkeypatch.setattr(history, "_THIN_FRAME", None)


def test_export_history_totals_coverage_and_new_destinations(monkeypatch, no_thin_cache):
    """Hubs' 2021 totals are rebuilt from their split, current rows are kept, and new corridors get zero history."""
    h = export_history()
    current = current_book(h)
    monkeypatch.setattr(history, "history", lambda: h)
    extended = history.thin_res(current, [])
    assert history.thin_res(current, []) is extended
    x = extended[extended.rep_c.eq("Malaysia") & extended.flow.eq("X")]
    assert x[x.period.str.startswith("2021")].value_usd.sum() == 1200.0
    pd.testing.assert_frame_equal(extended[extended.period.ge("2022-01")].reset_index(drop=True),
                                  current.reset_index(drop=True))
    readd = netting.hub_x_readd_rows(extended, "Malaysia", netting.MY_FINISHED)
    norway = readd[readd.dest.eq("Norway")]
    assert len(norway) == 24
    assert norway[norway.period.str.startswith("2021")].value_usd.sum() == 0.0
    assert readd[readd.dest.eq("Brazil") & readd.period.ge("2022-01")].value_usd.sum() == 0.0
    assert not extended.rep_c.eq("Taiwan").any()


def test_incomplete_export_history_is_rejected(monkeypatch, no_thin_cache):
    """An exporter missing a 2021 month fails instead of reading the gap as zero trade."""
    h = export_history()
    monkeypatch.setattr(history, "history", lambda: h[h.period.ne("2021-01")])
    with pytest.raises(ValueError, match="Incomplete 2021 export source"):
        history.thin_res(current_book(h), [])


def test_padding_uses_twelve_months_and_preserves_observations():
    """Every destination x code gets all twelve 2021 months, zeros where absent; padding is idempotent."""
    d = pd.DataFrame({"dest": ["Brazil", "Brazil", "UAE"],
                      "code_hs6": ["847150"] * 3,
                      "period": ["2021-02", "2022-01", "2022-01"],
                      "value_usd": [120.0, 200.0, 300.0]})
    padded = history.pad_history(d)
    pd.testing.assert_frame_equal(padded.iloc[:len(d)], d)
    h = padded[padded.period.str.startswith("2021")]
    assert h.groupby("dest").period.nunique().eq(12).all()
    assert h[h.dest.eq("Brazil")].value_usd.mean() == 10.0
    assert h[h.dest.eq("UAE")].value_usd.sum() == 0.0
    pd.testing.assert_frame_equal(history.pad_history(padded), padded)


def test_baseline_is_2021_level_times_index_normalized_to_2021():
    """The baseline is 2021 trade / 12 x the index over its 2021 mean; books without 2021 are untouched."""
    periods = pd.period_range("2021-01", "2022-12", freq="M").astype(str)
    d = pd.DataFrame({"dest": "A", "code_hs6": "847150", "period": periods, "value_usd": 120.0,
                      "gf": np.r_[np.ones(12) * 2, np.ones(12) * 3], "baseline": 99.0})

    class Index:
        composite = None

        def factor(self, leg, code):
            return pd.Series(d.gf.to_numpy(), index=periods)

    result = history.override_baselines(d.copy(), "dest", Index())
    np.testing.assert_allclose(result.baseline, np.r_[np.ones(12) * 120, np.ones(12) * 180])
    thin = d[d.period.ge("2022-01")].copy()
    pd.testing.assert_frame_equal(history.override_baselines(thin.copy(), "dest", Index()), thin)
    with pytest.raises(ValueError):
        history.override_baselines(d.iloc[1:].copy(), "dest", Index())


class StubFob:
    """A margin curve of 1% at $400/kg, falling with density."""
    code_dens, median_dens = {}, 1000.0

    @staticmethod
    def tau(q, dens):
        return 0.01 * (np.asarray(dens, dtype=float) / 400.0) ** -0.39


def pack():
    cols = ["reporter", "flow", "code_hs6", "period", "value_usd", "weight_kg", "value_qtycov"]
    return pd.DataFrame([
        ["australia", "M", "847150", "2021-01", 1000.0, 0.0, 1000.0],
        ["china", "M", "847150", "2021-02", 4000.0, 10.0, 0.0],     # pooled density $400/kg
        ["us", "M", "847150", "2021-01", 500.0, 0.0, 0.0],           # FOB-type: untouched
        ["australia", "X", "847150", "2021-01", 300.0, 0.0, 0.0],    # exports: untouched
    ], columns=cols)


def test_staged_fob_type_imports_now_cif_are_deflated(monkeypatch):
    """Pack imports staged as FOB but valued CIF are deflated at the pooled density's margin."""
    monkeypatch.setattr(fob, "get_model", lambda verbose=True: StubFob())
    monkeypatch.setattr(fob, "FOB_IMPORT_REPORTERS", {"us", "mexico", "canada"})
    d = pack()
    out = history.fob_pack_imports(d)
    assert out.value_usd[0] == pytest.approx(1000.0 / 1.01)
    assert out.value_qtycov[0] == pytest.approx(1000.0 / 1.01)
    pd.testing.assert_frame_equal(out.iloc[1:], d.iloc[1:])


def test_fob_type_australia_keeps_staged_values(monkeypatch):
    """When Australia's imports are FOB-type, the pack's values stand as staged."""
    monkeypatch.setattr(fob, "get_model", lambda verbose=True: StubFob())
    monkeypatch.setattr(fob, "FOB_IMPORT_REPORTERS", {"us", "mexico", "canada", "australia"})
    d = pack()
    pd.testing.assert_frame_equal(history.fob_pack_imports(d), d)
