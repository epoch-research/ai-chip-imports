"""Reconstruct the UK's 2021 imports from the EU, from EU members' export records.

The UK's 2021 imports from the EU are not comparable with later years (EU-origin goods were
recorded differently in the first year after the UK left the single market). Each 2021 month's
UK imports from the EU, per code, are replaced by EU members' exports to the UK in that month,
scaled by the ratio of UK imports from the EU to EU exports to the UK over July 2022-June 2023.
The machine codes 847149/847150 share one ratio. Non-EU imports and all exports are untouched.
"""
import numpy as np
import pandas as pd

CALIBRATION_START = '2022-07'
CALIBRATION_END = '2023-06'
GROUPS = {'847149': 'machines', '847150': 'machines',
          '847180': '847180', '847330': '847330'}


def require_months(series, start, end, label):
    """Reject incomplete or invalid calibration totals rather than compare unequal windows."""
    expected = pd.period_range(start, end, freq='M').astype(str)
    selected = series.reindex(expected)
    if not np.isfinite(selected).all() or selected.le(0).any():
        raise ValueError(f'Incomplete or invalid {label}: expected {start} through {end}')
    return selected


def reconstruct(history, uk_imports, eu_exports, eu_exports_2021, eu_members):
    """Return the corrected UK history, the calibration ratios and the monthly targets.

    ``uk_imports`` and ``eu_exports`` (calibration window) must be on comparable value bases;
    ``eu_exports_2021`` is a Series of export dollars indexed by (code_hs6, period). Quantity and
    weight are scaled with value, preserving the observed composition.
    """
    totals = []
    for frame in (uk_imports, eu_exports):
        d = frame[frame.period.between(CALIBRATION_START, CALIBRATION_END)
                  & frame.code_hs6.isin(GROUPS)].copy()
        d['group'] = d.code_hs6.map(GROUPS)
        for group in set(GROUPS.values()):
            require_months(d[d.group.eq(group)].groupby('period').value_usd.sum(),
                           CALIBRATION_START, CALIBRATION_END, f'{group} calibration')
        totals.append(d.groupby('group').value_usd.sum())
    ratios = totals[0] / totals[1]
    if set(ratios.index) != set(GROUPS.values()) or not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError('Incomplete or invalid calibration groups')
    out = history.copy()
    eligible = (out.flow.eq('M') & out.par_c.isin(eu_members)
                & out.period.str.startswith('2021') & out.code_hs6.isin(GROUPS))
    cols = ['value_usd', 'qty', 'weight_kg', 'value_qtycov']
    observed = out[eligible].groupby(['code_hs6', 'period']).value_usd.sum()
    expected = pd.MultiIndex.from_product([list(GROUPS), [f'2021-{m:02}' for m in range(1, 13)]],
                                          names=['code_hs6', 'period'])
    exports = eu_exports_2021.reindex(expected)
    if exports.isna().any() or (exports < 0).any() or (observed.reindex(expected).fillna(0) <= 0).any():
        raise ValueError('History and exports must cover every code/month')
    records = []
    for (code, period), value in exports.items():
        target = value * ratios[GROUPS[code]]
        mask = eligible & out.code_hs6.eq(code) & out.period.eq(period)
        out.loc[mask, cols] *= target / observed.loc[(code, period)]
        records.append({'code_hs6': code, 'period': period,
                        'original_usd': observed.loc[(code, period)], 'reconstructed_usd': target})
    pd.testing.assert_frame_equal(out[~eligible], history[~eligible])
    return out, ratios, pd.DataFrame(records)
