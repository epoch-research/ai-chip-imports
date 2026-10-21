"""Timing conventions shared by the estimator and the allocator.

Customs contributions are summed into calendar quarters of arrival. Epoch's sales are
reported per quarter; each quarter's sales are spread uniformly over its three months and
shifted by ``LAG_MONTHS`` (default two) to the months they arrive in, then summed into
calendar arrival quarters. With the default lag, Q2 arrivals receive two-thirds of Q1 sales
and one-third of Q2 sales.
"""
import os

import pandas as pd

LAG_MONTHS = int(os.environ.get("AI_CHIP_IMPORTS_LAG_MONTHS", "2"))
if not 0 <= LAG_MONTHS <= 6:
    raise ValueError("AI_CHIP_IMPORTS_LAG_MONTHS must be between 0 and 6")


def _shift_weights(months):
    """[(quarter shift, weight)] of the sales quarters feeding an arrival quarter: a lag of
    3k + r months draws (1 - r/3) of sales quarter q-k and r/3 of q-k-1."""
    k, r = divmod(months, 3)
    return [(k, 1 - r / 3)] + ([(k + 1, r / 3)] if r else [])


def _months_of(quarter):
    start = pd.Period(quarter, freq="Q").asfreq("M", "start")
    return {str(start + i) for i in range(3)}


def aggregate(frame, values, by=()):
    """Sum monthly contributions into complete calendar quarters.

    ``frame`` supplies ``period`` (YYYY-MM) and the ``by`` columns; ``values`` the amounts.
    A quarter is complete when all three of its months appear anywhere in ``frame`` and lie at
    or before the last complete quarter of the frame's frontier. A destination with no row in
    such a month contributes zero for it; months missing from the whole frame are never filled.
    """
    periods = frame["period"].astype(str)
    frontier = pd.Period(periods.max(), freq="M")
    full_end = frontier.asfreq("Q").asfreq("M", "end")
    if frontier != full_end:
        full_end = (frontier.asfreq("Q") - 1).asfreq("M", "end")
    available = {m for m in periods.unique() if m <= str(full_end)}
    mapping = {m: str(pd.Period(m, freq="M").asfreq("Q")) for m in available}
    complete = {q for q in mapping.values() if _months_of(q) <= available}
    work = frame[list(by)].copy()
    work["q"] = periods.map(mapping)
    work["value"] = list(values)
    work = work[work["q"].isin(complete)]
    return work.groupby(list(by) + ["q"])["value"].sum()


def metadata(quarters, lag=LAG_MONTHS):
    """Arrival and sales window labels for a sequence of arrival quarters."""
    qs = [pd.Period(q, freq="Q") for q in quarters]
    starts = [q.asfreq("M", "start") for q in qs]
    ends = [q.asfreq("M", "end") for q in qs]
    return pd.DataFrame({"arrival_quarter": [str(q) for q in qs],
                         "arrival_window_start": [str(m) for m in starts],
                         "arrival_window_end": [str(m) for m in ends],
                         "sales_window_start": [str(m - lag) for m in starts],
                         "sales_window_end": [str(m - lag) for m in ends]})


def interpolate_sales(series, months=LAG_MONTHS):
    """Calendar-quarter arrivals of a quarterly sales series; missing source quarters stay NaN."""
    axis = pd.period_range(series.index.min(), series.index.max(), freq="Q").astype(str)
    s = series.reindex(axis)
    k, r = divmod(months, 3)
    if r == 0:
        return s.shift(k)
    return s.shift(k + 1) * (r / 3) + s.shift(k) * (1 - r / 3)


def arrival_models(frame, months=LAG_MONTHS):
    """Move per-model quarterly sales to calendar arrival quarters, conserving each total.

    Extensive columns (units, H100e, cost and their percentiles) are split between arrival
    quarters; ASP is recomputed from the moved cost and units, never interpolated. A model
    absent from a populated sales quarter sold zero there. Arrival quarters drawing on a
    missing or incomplete sales quarter are flagged ``partial_sales_coverage``.
    """
    fields = ["units", "units_lo", "units_hi", "h100e", "h100e_lo", "h100e_hi", "cost"]
    parts = []
    for shift, weight in _shift_weights(months):
        if weight == 0:
            continue
        p = frame.copy()
        p["q"] = p.q.map(lambda q: str(pd.Period(q, freq="Q") + shift))
        p[fields] = p[fields] * weight
        parts.append(p)
    g = pd.concat(parts).groupby(["model", "q"], as_index=False).agg(
        **{c: (c, "sum") for c in fields}, designer=("designer", "first"),
        pool=("pool", "first"), cn_spec=("cn_spec", "first"), incomplete=("incomplete", "any"))
    status = frame.groupby("q").incomplete.any()

    def source_status(q):
        sources = [str(pd.Period(q, freq="Q") - k) for k, _ in _shift_weights(months)]
        return any(s not in status.index or bool(status.loc[s]) for s in sources)
    g["partial_sales_coverage"] = g.q.map(source_status)
    g["incomplete"] = g.incomplete | g.partial_sales_coverage
    g["asp"] = (g.cost / g.units.where(g.units.gt(0))).where(g.cost.gt(0))
    return g


def matched_sales(series):
    """Sales moved to arrival quarters; quarters drawing on an incomplete sales quarter
    (listed in ``series.attrs["incomplete_quarters"]``) are NaN."""
    out = interpolate_sales(series)
    incomplete = set(series.attrs.get("incomplete_quarters", []))
    for q in out.index:
        sources = {str(pd.Period(q, freq="Q") - k) for k, _ in _shift_weights(LAG_MONTHS)}
        if sources & incomplete:
            out.loc[q] = float("nan")
    return out


def sales_from_epoch(epoch_all):
    """Epoch's quarterly sell-in dollars, carrying its incomplete quarters in ``attrs``."""
    s = epoch_all.set_index("quarter")["epoch_cost_usd"].copy()
    s.attrs["incomplete_quarters"] = epoch_all.loc[epoch_all.epoch_incomplete, "quarter"].tolist()
    return s
