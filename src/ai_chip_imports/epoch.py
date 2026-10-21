"""Epoch AI's Chip Sales data: the supply side the customs shares distribute.

Reads ``timelines_by_chip.csv`` and ``chip_types.csv`` from the Epoch AI "AI Chip Sales"
dataset (``data/external/ai_chip_sales/``; https://epoch.ai/data/ai-chip-sales, CC BY). Epoch's
cost column is units x a fixed per-model sell-in price, so it is a chip-level sales value, not a
system price.
"""
import os

import numpy as np
import pandas as pd

from . import codes, paths, timing

EPOCH_DIR = str(paths.EXTERNAL_DIR / "ai_chip_sales")
# Designers whose accelerators are fabricated in Taiwan and so pass through the customs books.
TW_DESIGNERS = ["Nvidia", "AMD", "Google", "Amazon"]
# China-fabricated accelerators: they never appear in the Taiwan-centred supply chain the
# customs basket observes, and are allocated to China outright.
CN_DOMESTIC_DESIGNERS = {"Huawei", "Cambricon"}


def impute_missing_costs(t):
    """Give timeline rows with units but no cost an imputed cost.

    A model with units and zero cost would take units in the allocation without spending any
    destination's dollar budget, and would drag the reference ASP down. In order of preference:
    the catalogue price (chip_types "Cost per chip (approx.)") x units; else the designer's
    cost-weighted $/H100e that quarter x the row's H100e; else the same rate over the designer's
    group (China-domestic or the rest). Adds ``cost_imputed`` and ``cost_source``.
    """
    t = t.copy()
    units = pd.to_numeric(t["Number of Units"], errors="coerce").fillna(0.0)
    cost = pd.to_numeric(t["Cost Estimate (USD)"], errors="coerce").fillna(0.0)
    h100e = pd.to_numeric(t["Compute estimate in H100e (median)"], errors="coerce").fillna(0.0)
    q = pd.to_datetime(t["Start date"]).dt.to_period("Q").astype(str)
    need = (units > 0) & (cost <= 0)
    t["cost_imputed"] = False
    t["cost_source"] = np.where(cost > 0, "epoch", "")
    if not need.any():
        return t
    ct = pd.read_csv(os.path.join(EPOCH_DIR, "chip_types.csv"))
    catalogue = (pd.to_numeric(ct["Cost per chip (approx.)"], errors="coerce")
                 .groupby(ct["Name"].astype(str).str.strip()).first())
    model = t["Chip type (linked)"].astype(str).str.strip()
    designer = t["Chip manufacturer"].astype(str)
    group = np.where(designer.isin(CN_DOMESTIC_DESIGNERS), "cn_domestic", "other")
    priced = (cost > 0) & (h100e > 0)
    per_h = pd.DataFrame({"q": q, "designer": designer, "group": group, "cost": cost, "h": h100e})[priced]
    by_designer = per_h.groupby(["designer", "q"]).apply(
        lambda g: g["cost"].sum() / g["h"].sum(), include_groups=False)
    by_group = per_h.groupby(["group", "q"]).apply(
        lambda g: g["cost"].sum() / g["h"].sum(), include_groups=False)
    new_cost, source = cost.astype(float).copy(), t["cost_source"].copy()
    for i in t.index[need]:
        price = catalogue.get(model[i], np.nan)
        if np.isfinite(price) and price > 0:
            new_cost[i], source[i] = units[i] * price, "catalogue"
            continue
        rate = by_designer.get((designer[i], q[i]), np.nan)
        src = "designer_h100e"
        if not (np.isfinite(rate) and rate > 0):
            rate, src = by_group.get((group[i], q[i]), np.nan), "group_h100e"
        if np.isfinite(rate) and rate > 0 and h100e[i] > 0:
            new_cost[i], source[i] = rate * h100e[i], src
    t["cost_imputed"] = need & (new_cost > 0)
    imp = t["cost_imputed"].to_numpy()
    # whole dollars, so the source's integer cost column keeps its dtype
    est = t["Cost Estimate (USD)"].copy()
    est.loc[imp] = np.round(new_cost[imp]).astype(est.dtype if est.dtype.kind in "iu" else float)
    t["Cost Estimate (USD)"] = est
    chip = pd.to_numeric(t["Chip Cost (USD)"], errors="coerce")
    chip.loc[imp] = (new_cost[imp] / units[imp]).to_numpy()
    t["Chip Cost (USD)"] = chip
    t["cost_source"] = source
    return t


def load_timelines():
    """Timeline rows of the Taiwan-fabricated designers, costs imputed, with ``quarter`` and
    ``incomplete`` columns. Imputation runs before the designer filter, so designer rates use
    every row."""
    t = pd.read_csv(os.path.join(EPOCH_DIR, "timelines_by_chip.csv"))
    t = impute_missing_costs(t)
    t = t[t["Chip manufacturer"].isin(TW_DESIGNERS)].copy()
    t["quarter"] = t["Start date"].map(lambda s: codes.quarter(s[:7]))
    t["incomplete"] = t["Incomplete"].astype(str).str.lower().eq("true")
    return t


def quarterly_sales(t):
    """Per-quarter totals of ``load_timelines()``: units, H100e (median and 5th/95th
    percentiles), sell-in dollars, and whether any row is flagged incomplete."""
    g = t[t["Chip manufacturer"].isin(TW_DESIGNERS)].groupby("quarter").agg(
        epoch_units=("Number of Units", "sum"),
        epoch_h100e=("Compute estimate in H100e (median)", "sum"),
        epoch_h100e_lo=("H100e (5th percentile)", "sum"),
        epoch_h100e_hi=("H100e (95th percentile)", "sum"),
        epoch_cost_usd=("Cost Estimate (USD)", "sum"),
        epoch_incomplete=("incomplete", "any")).reset_index()
    return g.sort_values("quarter")


def models_by_arrival_quarter():
    """Every designer's sales per model, moved to calendar arrival quarters.

    One row per model x arrival quarter with units, H100e and their 5th/95th percentiles, cost
    and ASP, the designer, ``cn_spec`` (models Epoch flags as primarily for the Chinese market)
    and the allocation ``pool``: ``cn_domestic`` (China-fabricated), ``cn_spec``, or ``row``.
    The model key is ``Chip type (linked)``, which joins ``chip_types.csv`` exactly. Row-level
    H100e columns are used as published. A missing percentile is replaced by the median.
    """
    t = impute_missing_costs(pd.read_csv(os.path.join(EPOCH_DIR, "timelines_by_chip.csv")))
    ct = pd.read_csv(os.path.join(EPOCH_DIR, "chip_types.csv"))
    cn_spec = set(ct.loc[ct["Primarily for Chinese market"].astype(str).str.lower()
                         .eq("true"), "Name"])
    known = set(ct["Name"])
    t["model"] = t["Chip type (linked)"].astype(str).str.strip()
    bad = sorted(set(t["model"]) - known)
    assert not bad, f"unjoined Chip type (linked) values: {bad}"
    t["q"] = t["Start date"].str[:7].map(codes.quarter)
    t["incomplete"] = t["Incomplete"].astype(str).str.lower().eq("true")
    g = (t.groupby(["model", "q"], as_index=False)
         .agg(designer=("Chip manufacturer", "first"),
              units=("Number of Units", "sum"),
              units_lo=("Number of Units (5th percentile)", "sum"),
              units_hi=("Number of Units (95th percentile)", "sum"),
              h100e=("Compute estimate in H100e (median)", "sum"),
              h100e_lo=("H100e (5th percentile)", "sum"),
              h100e_hi=("H100e (95th percentile)", "sum"),
              cost=("Cost Estimate (USD)", "sum"),
              incomplete=("incomplete", "any")))
    g["asp"] = np.where(g["units"] > 0, g["cost"] / g["units"], np.nan)
    g.loc[g["cost"] <= 0, "asp"] = np.nan          # zero-cost rows are unpriced, not free
    for c, m in (("units_lo", "units"), ("units_hi", "units"),
                 ("h100e_lo", "h100e"), ("h100e_hi", "h100e")):
        g[c] = g[c].where(g[c] > 0, g[m])
    g["cn_spec"] = g["model"].isin(cn_spec)
    g["pool"] = np.where(g["designer"].isin(CN_DOMESTIC_DESIGNERS), "cn_domestic",
                         np.where(g["cn_spec"], "cn_spec", "row"))
    return timing.arrival_models(g)
