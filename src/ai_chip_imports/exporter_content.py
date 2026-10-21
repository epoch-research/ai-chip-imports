"""Chip content for destinations without an importer book, from the exporters' own records.

Countries that publish no usable import book (the "thin tier") are measured through the
exports of the books that ship to them: the hubs' re-exports and dispatches, US exports, the
EU members' extra-EU dispatches, and Taiwan's exports (the residual bucket). Their excess over
the reference baseline is computed per destination, code and month from the summed
contributions (``ledger``); this module supplies the share of that excess that is accelerator
content.

Every exporter x destination x code x month row is classified from its own physical fields,
with the importer-side ladders of ``content``: pieces where they cover at least 90% of the
row's value (machine codes only), else net weight at that coverage; boards are read by weight
only. Value not covered by a field is imputed from donors -- the value-weighted mean content of
observed rows in the same exporter/code/quarter, else code/quarter, else code/year, each pool
needing at least three destinations. China and Hong Kong are excluded as donors because their
rows are priced against a different (China-specific) model mix. If no donor exists the run
fails rather than falling back to accelerator-system content.

A destination's monthly excess is attributed to exporters in proportion to their gross
shipment values (the excess is assumed to have the month's observed exporter mix); exporter-
level excesses are never summed. Exporters' physical fields cannot show whether hardware stays
in the destination after arrival: all of it is assumed retained.
"""
import numpy as np
import pandas as pd

from . import books, content, history, ledger, netting, timing

KEYS = ["dest", "code_hs6", "period"]
RESIDUAL = "(residual: no importer leg)"
_SOURCE = _ROWS = None


def source_rows(res):
    """The export rows behind every thin-tier contribution, summed per exporter and destination.

    Exactly the exporters, flows and codes the estimator re-adds (``netting.uniform_universe``)
    plus Taiwan's exports. Every real destination is kept as a potential donor; ``counted``
    marks the destinations without a book of their own, which are the ones re-added. US
    re-exports and domestic exports stay separate, and total exports are never combined with
    their re-export/domestic split.
    """
    global _SOURCE, _ROWS
    if _SOURCE is res:
        return _ROWS
    parts = []
    specs = [("United States", "RX", history.CODES), ("United States", "DX", netting.US_DX_FINISHED)]
    specs += [("Malaysia", "X", netting.MY_FINISHED)]
    specs += [(x, "X", netting.SGHK_FINISHED) for x in netting.SGHK_HUBS]
    specs += [(x, "X", netting.NEWLEG_FINISHED) for x in netting.NEWLEG_NETTED]
    specs += [("Taiwan", "X", history.CODES)]
    for reporter, flow, codes in specs:
        p = res[res.rep_c.eq(reporter) & res.flow.eq(flow) & res.code_hs6.isin(codes)].copy()
        p["exporter"] = reporter + ":" + flow
        p["taiwan"] = reporter == "Taiwan"
        parts.append(p)
    p = netting.eu_extra_flows(res, "X").copy()
    p = p[p.code_hs6.isin(netting.EU_FINISHED)]
    p["exporter"] = p.rep_c + ":X"
    p["taiwan"] = False
    parts.append(p)
    d = pd.concat(parts, ignore_index=True)
    norm = d.partner.map(books.NORM)
    d = d[~norm.isin({"world", "total"}) & d.value_usd.gt(0)].copy()
    norm = d.partner.map(books.NORM)
    d["counted"] = d.par_c.isna() & ~norm.isin(netting.ASSEMBLY_NODES)
    d["dest"] = norm.str.title().map(ledger.clean_thin)
    d["qty"] = pd.to_numeric(d.qty, errors="coerce").fillna(0).clip(lower=0)
    d["weight_kg"] = pd.to_numeric(d.weight_kg, errors="coerce").fillna(0).clip(lower=0)
    d["qty_value"] = np.where(d.qty.gt(0), d.value_qtycov.fillna(0).clip(lower=0), 0.)
    d["qty_value"] = np.minimum(d.qty_value, d.value_usd)
    d["kg_value"] = np.where(d.weight_kg.gt(0), d.value_usd, 0.)
    d = d.groupby(["exporter", "taiwan", "counted", *KEYS], as_index=False)[
        ["value_usd", "qty", "weight_kg", "qty_value", "kg_value"]].sum()
    _SOURCE, _ROWS = res, d
    return d


def weighted_quantile(values, weights, q):
    """The ``q`` quantile of ``values`` under ``weights`` (lower empirical quantile)."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(v[min(np.searchsorted(np.cumsum(w), q * w.sum()), len(v) - 1)])


def classify(rows, model, rack=None):
    """Content per export row: observed where a physical field covers the row, imputed otherwise.

    Adds ``field``/``coverage`` (the field used and the share of value it covers),
    ``observed_content``, the donor pool level, and ``content`` -- the coverage-weighted blend
    of the observed and the imputed content. ``content_low``/``content_high`` blend in the
    donor pool's 10th/90th percentiles instead (widened to include the mean): imputation
    scenarios, not probability bounds.
    """
    d = rows.copy()
    d["q"] = pd.PeriodIndex(d.period, freq="M").asfreq("Q").astype(str)
    d["year"] = d.period.str[:4]
    covq = d.qty_value / d.value_usd
    covk = d.kg_value / d.value_usd
    qok = covq.ge(content.CELL_FIELD_MIN) & d.code_hs6.ne("847330") & d.qty.gt(0)
    kok = ~qok & covk.ge(content.CELL_FIELD_MIN) & d.weight_kg.gt(0)
    d["field"] = np.select([qok, kok], ["qty", "weight_kg"], default="missing")
    d["coverage"] = np.select([qok, kok], [covq, covk], default=0.)
    srv = model.mix("row", rack).reindex(d.q).to_numpy()
    asp = model.asp("row").reindex(d.q).to_numpy()
    if not np.isfinite(srv).all() or not np.isfinite(asp).all():
        raise ValueError("Exporter classifier lacks Epoch ASP/mix references")
    uv = np.divide(d.qty_value, d.qty, out=np.zeros(len(d)), where=d.qty.gt(0))
    uq = np.array([content._tier_content(r, s) for r, s in zip(uv / asp, srv)])
    dens = np.divide(d.kg_value, d.weight_kg, out=np.zeros(len(d)), where=d.weight_kg.gt(0))
    board = d.code_hs6.eq("847330")
    lo = np.where(board, content.CELL_DENS["boards"][0], content.CELL_DENS["server"][0])
    hi = np.where(board, content.CELL_DENS["boards"][1], content.CELL_DENS["server"][1])
    ramp = np.clip((np.log(np.maximum(dens, 1e-9)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
    uk = content.TIER_C["low"] + (np.where(board, content.TIER_C["board"], srv) - content.TIER_C["low"]) * ramp
    d["observed_content"] = np.select([qok, kok], [uq, uk], default=np.nan)
    d["observed_value"] = d.value_usd * d.coverage
    # Every counted thin destination is priced against the rest-of-world reference, so China and
    # Hong Kong rows (read against the China-specific mix) are not comparable donors.
    d["eligible_donor"] = ~d.dest.isin(content.CN_SIDE)
    donors = d[d.observed_content.notna() & d.observed_value.gt(0) & d.eligible_donor]
    d["donor_level"] = "observed"
    for name in ["imputed_content", "imputed_low", "imputed_high"]:
        d[name] = np.nan
    need = d.coverage.lt(1 - 1e-12)
    for keys in [["exporter", "code_hs6", "q"], ["code_hs6", "q"], ["code_hs6", "year"]]:
        pools = {}
        for key, g in donors.groupby(keys, sort=False):
            if g.dest.nunique() >= 3:
                mean = np.average(g.observed_content, weights=g.observed_value)
                pools[key] = (mean,
                              min(mean, weighted_quantile(g.observed_content, g.observed_value, .1)),
                              max(mean, weighted_quantile(g.observed_content, g.observed_value, .9)))
        for i in d.index[need & d.imputed_content.isna()]:
            val = pools.get(tuple(d.loc[i, keys]))
            if val is not None:
                d.loc[i, ["imputed_content", "imputed_low", "imputed_high"]] = val
                d.loc[i, "donor_level"] = "/".join(keys)
    if (need & d.imputed_content.isna()).any():
        bad = d.loc[need & d.imputed_content.isna(), ["exporter", *KEYS]].head()
        raise ValueError(f"No empirical exporter-content donors:\n{bad}")
    for name, imputed in [("content", "imputed_content"), ("content_low", "imputed_low"),
                          ("content_high", "imputed_high")]:
        d[name] = (d.coverage * d.observed_content.fillna(0)
                   + (1 - d.coverage) * d[imputed].fillna(0))
    return d


def evidence(model, res=None, rack=None):
    """Classified export rows, cached on the content model per rack constant.

    Pass ``res`` once to attach the source rows; later calls reuse them.
    """
    if res is not None:
        rows = source_rows(res)
        if getattr(model, "_exporter_rows", None) is not rows:
            model._exporter_rows = rows
            model._exporter_cache = {}
    if not hasattr(model, "_exporter_rows"):
        raise ValueError("Exporter content requires source rows: call evidence(model, res) first")
    key = model.rack if rack is None else rack
    if key not in model._exporter_cache:
        model._exporter_cache[key] = classify(model._exporter_rows, model, rack)
    return model._exporter_cache[key]


def aggregate(e, residual=False):
    """Value-weighted content per destination, code and month over the counted rows.

    ``residual=True`` pools Taiwan's rows into the single residual bucket; otherwise Taiwan's
    rows are excluded (they reach destinations only through the residual split).
    """
    d = e[e.counted & e.taiwan.eq(residual)].copy()
    if residual:
        d["dest"] = RESIDUAL
    d["cv"] = d.value_usd * d["content"]
    d["iv"] = d.value_usd * (1 - d.coverage)
    d["ic"] = d.iv * d.imputed_content.fillna(0)
    g = d.groupby(KEYS)[["value_usd", "cv", "iv", "ic"]].sum()
    g["content"] = g.cv / g.value_usd
    g["imputed_value_fraction"] = g.iv / g.value_usd
    g["imputed_chip_weight"] = g.ic / g.value_usd
    return g


def weights(co, model, original, rack=None):
    """Content weights for the excess frame ``co``: exporter evidence on thin-tier rows.

    Rows of destinations with their own book keep ``original``. Fails if a thin-tier row with
    positive excess has no export evidence, and checks that the evidence's values reproduce
    the estimator's own thin-tier contributions.
    """
    e = evidence(model, rack=rack)
    g = pd.concat([aggregate(e), aggregate(e, True)])
    idx = pd.MultiIndex.from_frame(co[KEYS])
    mapped = g.reindex(idx)
    thin = ~co.dest.isin(netting._un_legs())
    required = thin & co.excess.gt(0)
    if mapped.content.isna().to_numpy()[required].any():
        raise ValueError("Positive inferred-country excess has no export content evidence")
    direct = thin & ~co.dest.eq(RESIDUAL) & co.value_usd.gt(0)
    np.testing.assert_allclose(mapped.value_usd.to_numpy()[direct], co.loc[direct, "value_usd"],
                               rtol=1e-9, atol=.01)
    out = np.asarray(original).copy()
    out[thin] = mapped.content.fillna(0).to_numpy()[thin]
    return out


def exposure(co, model):
    """Per destination, code and month: the share of trade whose content was imputed, and the
    chip dollars resting on imputed content (in the units of the excess ledger)."""
    e = evidence(model)
    g = pd.concat([aggregate(e), aggregate(e, True)])
    m = g.reindex(pd.MultiIndex.from_frame(co[KEYS]))
    out = co[KEYS + ["excess"]].copy()
    out["imputed_trade_fraction"] = m.imputed_value_fraction.fillna(0).to_numpy()
    out["imputed_chip_dollars"] = co.excess.to_numpy() * m.imputed_chip_weight.fillna(0).to_numpy()
    return out


def residual_weights(co, model, res):
    """Content weights for the per-country split of Taiwan's residual bucket, from Taiwan's own
    classified exports to each destination."""
    e = evidence(model, res)
    e = e[e.taiwan & e.counted].copy()
    e["cv"] = e.value_usd * e["content"]
    g = e.groupby(KEYS)[["value_usd", "cv"]].sum()
    w = (g.cv / g.value_usd).reindex(pd.MultiIndex.from_frame(co[KEYS]))
    if w.isna().to_numpy()[co.excess.gt(0)].any():
        raise ValueError("Taiwan residual excess lacks physical/donor evidence")
    return w.fillna(0).to_numpy()


def country_exposure(co, residual_co, model, res, shares):
    """Per destination and arrival quarter: the chip dollars that depend on imputed content.

    For destinations reached through Taiwan's residual bucket, both the bucket's own imputed
    dollars and the imputation behind each country's splitting key count; their union avoids
    counting a dollar twice. This measures dependence on imputation, not an uncertainty band.
    """
    from . import allocator
    x = exposure(co, model)
    direct = timing.aggregate(co, x.imputed_chip_dollars, by=("dest",))
    bucket = direct.xs(RESIDUAL, level="dest")
    direct = direct.drop(RESIDUAL, level="dest").unstack("dest").fillna(0)
    total = timing.aggregate(co, co.chiponly_corr, by=("dest",))
    bucket_total = total.xs(RESIDUAL, level="dest")
    e = evidence(model, res)
    e = e[e.taiwan].copy()
    e["taiwan"] = False
    g = aggregate(e).reindex(pd.MultiIndex.from_frame(residual_co[KEYS]))
    ip = residual_co.excess.to_numpy() * g.imputed_chip_weight.fillna(0).to_numpy()
    key_imputed = timing.aggregate(residual_co, ip, by=("dest",)).unstack("dest").fillna(0)
    key_total = (timing.aggregate(residual_co, residual_co.chiponly, by=("dest",))
                 .unstack("dest").fillna(0).clip(lower=0))
    # The split keys are clipped at zero before allocation, so only nonnegative dollars count.
    fraction = (key_imputed.div(key_total.replace(0, np.nan)).fillna(0).clip(0, 1)
                .reindex_like(shares).fillna(0))
    # A quarter with no positive key falls back to the pooled split; carry its missingness too.
    pooled_i = (pd.Series(ip).groupby([residual_co.dest.to_numpy(), residual_co.q.to_numpy()])
                .sum().clip(lower=0).groupby(level=0).sum())
    pooled_t = residual_co.groupby(["dest", "q"]).chiponly.sum().clip(lower=0).groupby(level=0).sum()
    fallback = key_total.sum(axis=1).eq(0)
    fraction.loc[fallback] = ((pooled_i / pooled_t.clip(lower=0).replace(0, np.nan)).clip(0, 1)
                              .reindex(fraction.columns).fillna(0).to_numpy())
    aggregate_fraction = (bucket / bucket_total.replace(0, np.nan)).fillna(0)
    union = 1 - (1 - fraction).mul(1 - aggregate_fraction, axis=0)
    indirect = shares.mul(bucket_total, axis=0) * union
    numerator = direct.add(indirect, fill_value=0).fillna(0)
    denominator = allocator.clean_D(total, shares)
    out = denominator.stack().rename("chip_dollars").to_frame()
    out["imputation_dependent_dollars"] = numerator.stack().reindex(out.index).fillna(0)
    out["fraction"] = out.imputation_dependent_dollars / out.chip_dollars.replace(0, np.nan)
    out = out.reset_index()
    meta = timing.metadata(out.q)
    for col in meta:
        out[col] = meta[col].to_numpy()
    return out
