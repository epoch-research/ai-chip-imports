"""Excess over the non-AI baseline, per destination x code, and the netting ledger.

``corridor_chiponly`` is the kernel shared by the estimator and the allocator. For every
destination x HS-6 code it sets a non-AI baseline, takes the gap between retained trade and that
baseline, and runs the quarterly gaps through a netting ledger that turns them into counted
excess (the AI-attributable dollars before chip content is applied):

* **Baseline.** On the headline basis (``historical=True``): 2021 trade / 12 x a measured index
  normalized to its 2021 mean (``history.override_baselines``). Without history: the 2022
  monthly mean grown by the measured index where one exists, else at a parametric rate.
* **Ledger.** Gaps are netted per quarter with a memory of ``NETTING_QUARTERS`` quarters: a
  shortfall can be repaid by surpluses in the following quarters and a counted surplus can be
  charged back by later shortfalls, within the memory. On the headline basis charges reduce
  the quarter they charge (``history.retrospective_ledger``), so counted excess is never
  negative; otherwise they are booked as later negative amounts (``_net_ledger``).

The module also holds the thin-tier label harmonization (``clean_thin``) and the
Taiwan-origin import view used for the residual bucket's price uplift (``v2a_importer_netted``).
"""
import os

import numpy as np
import pandas as pd

from . import books, codes

# Netting memory in quarters ("inf" = a cumulative floor that never expires).
NETTING_QUARTERS = os.environ.get("AI_CHIP_IMPORTS_NETTING_QUARTERS", "2").strip().lower()
if NETTING_QUARTERS not in ("inf", "infinity") and not NETTING_QUARTERS.isdigit():
    raise ValueError("AI_CHIP_IMPORTS_NETTING_QUARTERS must be a non-negative integer or 'inf'")


# --- thin-tier destination labels ------------------------------------------------------------
# Destinations without a book are named by whichever exporter ships to them, and every exporter
# spells countries its own way. clean_thin() maps the spellings to one label (alias map first,
# then a ", Xxx Of"-style official suffix is stripped). Headline destinations are fixed points.
THIN_ALIASES = {
    "Brunei Darussalam": "Brunei", "Cabo Verde": "Cape Verde",
    "Bosnia And Herzegovina": "Bosnia & Herzegovina",
    "British Indian Ocean Territories": "British Indian Ocean Territory",
    "Burma": "Myanmar", "Russian Federation": "Russia", "Turkiye": "Turkey",
    "Korea, Democratic People'S Republic Of": "North Korea",
    "Lao People'S Democratic Republic": "Laos", "Syrian Arab Republic": "Syria",
    # Keep the two Virgin Islands apart (the comma rule below would merge them) and unify the
    # curly- and straight-apostrophe spellings of Cote d'Ivoire.
    "Virgin Islands, British": "British Virgin Islands",
    "Virgin Islands, United States": "U.S. Virgin Islands",
    "Us Virgin Islands": "U.S. Virgin Islands",
    "Us Minor Outlying Islands": "United States Minor Outlying Islands",
    "French Southern And Antarctic Lands": "French Southern Territories",
    "Côte D’Ivoire": "Cote D'Ivoire",
    "Macedonia": "North Macedonia", "St Lucia": "Saint Lucia",
    "East Timor": "Timor-Leste", "Timor Leste": "Timor-Leste",
    "Western Samoa": "Samoa", "Antigua & Barbuda": "Antigua And Barbuda",
    "Saint Barthelemy": "Saint Barthélemy", "Cote D' Ivoire": "Cote D'Ivoire",
    "Türkiye": "Turkey", "Macao": "Macau", "Côte D'Ivoire": "Cote D'Ivoire",
    "Curaçao": "Curacao", "Islamic Republic Of Iran": "Iran",
    "Central African Rep": "Central African Republic",
    "Northern Mariana Is": "Northern Mariana Islands",
    "St Helena": "Saint Helena", "St Kitts And Nevis": "Saint Kitts And Nevis",
    "St Vincent And The Grenadines": "Saint Vincent And The Grenadines",
    "St. Helena": "Saint Helena", "St. Kitts And Nevis": "Saint Kitts And Nevis",
    "St. Vincent And The Grenadines": "Saint Vincent And The Grenadines",
    "St. Lucia": "Saint Lucia",
    "Trinidad & Tobago": "Trinidad And Tobago", "Swaziland": "Eswatini",
    "Occupied Palestinian Territory": "Palestine",
    "West Bank Administered By Israel": "Palestine",
    "Others": "(unspecified)",
}
THIN_ALIASES_CANON = {
    "Congo, Dem Rep The": "Congo (Dem. Rep)", "Congo, Democratic Republic Of": "Congo (Dem. Rep)",
    "Democratic Republic Of The Congo": "Congo (Dem. Rep)",
    "Congo, The Democratic Republic Of The": "Congo (Dem. Rep)",
    "Dem. Rep. Of The Congo": "Congo (Dem. Rep)", "Congo, Democratic Republic Of The": "Congo (Dem. Rep)",
    # Taiwan customs portal spellings that reach the thin tier through the residual split
    "Congo D. Rep.": "Congo (Dem. Rep)", "Bangladesh Pr": "Bangladesh", "Sri Lanka Dsr": "Sri Lanka",
    "Bosnia & Herz.": "Bosnia & Herzegovina", "Lao Peoples Dem. Rep.": "Laos",
    "Sao Tome-Princ.": "Sao Tome And Principe", "St Kitts & Nevis": "Saint Kitts And Nevis",
    "St Vincent": "Saint Vincent And The Grenadines", "Br Virgin Is": "British Virgin Islands",
    "Antigua:Barbuda": "Antigua And Barbuda", "Surinam": "Suriname",
    "Turks & Caicos": "Turks And Caicos Islands", "Stores & Provis.": "(unspecified)",
    "Unspecified": "(unspecified)", "Uae": "United Arab Emirates", "Baharain Is": "Bahrain",
    # abbreviated exporter spellings (India/UK/Indonesia style) against other exporters' names
    "Burkina": "Burkina Faso", "Cent Afr Rep": "Central African Republic",
    "Dominican Rep": "Dominican Republic", "Egypt A Rp": "Egypt",
    "Equat Guinea": "Equatorial Guinea", "Fiji Is": "Fiji", "Ivory Coast": "Cote D'Ivoire",
    "Kyrghyzstan": "Kyrgyzstan", "Kyrgyz Republic": "Kyrgyzstan",
    "Papua N Gna": "Papua New Guinea", "Solomon Is": "Solomon Islands",
    "Tanzania Rep": "Tanzania", "Trinidad": "Trinidad And Tobago",
    "Trinidad:Tobago": "Trinidad And Tobago", "Us Virgin Is": "U.S. Virgin Islands",
    "Congo (Republic)": "Congo",
}
_ALIASES = {**THIN_ALIASES, **THIN_ALIASES_CANON}
# The map must be idempotent: no alias target may itself be an alias key.
assert not [v for v in _ALIASES.values() if v in _ALIASES], "thin alias map not idempotent"


def clean_thin(name):
    """Harmonize a thin-tier destination label: alias map, then strip a ', Xxx Of' suffix."""
    n = _ALIASES.get(name, name)
    if n in _ALIASES.values():
        return n
    if "," in n:
        head = n.split(",")[0].strip()
        n = _ALIASES.get(head, head)
    return n


# --- the netting ledger -----------------------------------------------------------------------
def _net_ledger(gaps, W):
    """Counted excess per quarter for one corridor's signed quarterly gaps, charges booked late.

    A surplus first repays outstanding shortfalls (newest first); a shortfall charges back
    counted surpluses (newest first) as a negative amount in the current quarter. Entries
    older than ``W`` quarters expire.
    """
    short, surp, out = [], [], []          # ledger entries [amount, age]; oldest first
    for g in gaps:
        for L in (short, surp):
            for e in L:
                e[1] += 1
            while L and L[0][1] > W:
                L.pop(0)
        x = 0.0
        if g > 0:
            rem = g
            while rem > 1e-9 and short:
                pay = min(rem, short[-1][0])
                short[-1][0] -= pay
                rem -= pay
                if short[-1][0] <= 1e-9:
                    short.pop()
            x = rem
            if rem > 1e-9:
                surp.append([rem, 0])
        elif g < 0:
            rem, charged = -g, 0.0
            while rem > 1e-9 and surp:
                c = min(rem, surp[-1][0])
                surp[-1][0] -= c
                rem -= c
                charged += c
                if surp[-1][0] <= 1e-9:
                    surp.pop()
            x = -charged
            if rem > 1e-9:
                short.append([rem, 0])
        out.append(x)
    return out


def _ledger_excess(d, dest_col, retrospective=False):
    """Row-level counted excess under the netting ledger.

    Quarterly gaps per destination x code run through the ledger over a continuous quarter axis
    (entries age in calendar time). Each quarter's counted amount is spread over that quarter's
    rows in proportion to their positive gaps (a surplus) or negative gaps (a charge), so the
    rows sum exactly to the ledger's quarterly figure.
    """
    from .history import retrospective_ledger
    W = 10 ** 6 if NETTING_QUARTERS in ("inf", "infinity") else int(NETTING_QUARTERS)
    gap = (d["value_usd"] - d["baseline"]).fillna(0.0).to_numpy(dtype=float)
    k = pd.DataFrame({"c1": d[dest_col].to_numpy(), "c2": d["code_hs6"].to_numpy(),
                      "q": d["period"].map(codes.quarter).to_numpy(), "gap": gap})
    qaxis = sorted(k["q"].unique())
    if retrospective and qaxis:
        qaxis = list(pd.period_range(qaxis[0], qaxis[-1], freq="Q").astype(str))
    piv = (k.pivot_table(index=["c1", "c2"], columns="q", values="gap", aggfunc="sum")
           .reindex(columns=qaxis).fillna(0.0))
    ledger = retrospective_ledger if retrospective else _net_ledger
    X = pd.DataFrame([ledger(r, W) for r in piv.to_numpy()], index=piv.index, columns=qaxis)
    xrow = (X.stack().reindex(pd.MultiIndex.from_arrays([k["c1"], k["c2"], k["q"]]))
            .to_numpy(dtype=float))
    w = np.where(xrow > 0, np.clip(gap, 0.0, None),
                 np.where(xrow < 0, np.clip(-gap, 0.0, None), 0.0))
    wsum = pd.Series(w).groupby([k["c1"], k["c2"], k["q"]]).transform("sum").to_numpy()
    share = np.divide(w, wsum, out=np.zeros_like(w), where=wsum > 0)
    return xrow * share


def corridor_chiponly(df, dest_col, growth=None, baseline_index=None, historical=False, ai2022=None):
    """Baseline, counted excess and flat-content chip dollars per row of ``df``.

    ``df`` needs ``value_usd``, ``period``, ``code_hs6`` and ``dest_col``. ``growth`` overrides
    the parametric non-AI growth rates (keyed by channel, see ``codes.NONAI_GROWTH``);
    ``baseline_index`` (a ``baseline.BaselineIndex``) replaces the parametric factor with the
    measured index where one exists. ``historical=True`` anchors the baseline on 2021 (the
    rows must include the 2021 history), uses the retrospective ledger and drops the 2021
    warm-up rows from the result. On the 2022 reference, ``ai2022`` ({code: $/month}) carves
    the AI content of 2022 out of the 2022 means, split across destinations by their share of
    the code's 2022 mean. Returns the rows with ``baseline``, ``excess``, ``q``, ``chiponly``
    (excess x the flat per-code content) and ``dest_col`` renamed to ``dest``.
    """
    growth = growth or codes.NONAI_GROWTH
    d = df[df["code_hs6"].isin(codes.BASKET)].copy()
    d["year"] = d["period"].str[:4]
    b = (d[d["year"] == "2022"].groupby([dest_col, "code_hs6"])["value_usd"].mean()
         .rename("mean2022").reset_index())
    if ai2022:
        tot = b.groupby("code_hs6")["mean2022"].transform("sum")
        b["ai2022_d"] = (b["code_hs6"].map(ai2022).fillna(0.0)
                         * np.where(tot > 0, b["mean2022"] / tot, 0.0))
    d = d.merge(b, on=[dest_col, "code_hs6"], how="left")
    d["mean2022"] = d["mean2022"].fillna(0.0)          # a corridor new after 2022 has no floor
    d["g"] = d["code_hs6"].map(lambda c: growth[codes.channel(c)])
    d["gf"] = (1.0 + d["g"]) ** d["period"].map(codes.years_since_mid2022)
    if baseline_index is not None:
        d = d.reset_index(drop=True)
        d["gf"] = baseline_index.lookup(d[dest_col], d["code_hs6"], d["period"], d["gf"])
    if ai2022:
        d["ai2022_d"] = d["ai2022_d"].fillna(0.0)
        d["baseline"] = (d["mean2022"] - d["ai2022_d"]).clip(lower=0.0) * d["gf"]
    else:
        d["baseline"] = d["mean2022"].clip(lower=0.0) * d["gf"]
    if historical:
        from .history import override_baselines
        d = override_baselines(d, dest_col, baseline_index)
    d["excess"] = _ledger_excess(d, dest_col, retrospective=historical)
    if historical:
        # 2021 initializes the baseline and the ledgers; it is never an allocated cohort.
        d = d[d.period.ge("2022-01")].copy()
    d["q"] = d["period"].map(codes.quarter)
    d["chiponly"] = d["excess"] * d["code_hs6"].map(codes.CHIP_CONTENT)
    return d.rename(columns={dest_col: "dest"})


def v2a_importer_netted(res):
    """Importers' imports of Taiwan-origin goods, net of the hubs' re-exports of them.

    Importers book by origin, so this sees Taiwan's hardware where it finally lands. For books
    with a re-export split, re-exports are removed in proportion to Taiwan's share of the
    reporter's recent imports of the code (``books._origin_weights``). The ratio of this to
    Taiwan's own exports measures the price wedge between Taiwan's export declarations and
    importers' declared values, used to lift Taiwan's residual exports.
    """
    M = res[(res["flow"] == "M") & res["rep_c"].notna() & (res["rep_c"] != "Taiwan")
            & res["code_hs6"].isin(codes.BASKET)].copy()
    M["origin"] = M["par_c"].where(M["par_c"].notna(), M["partner"].map(books.NORM))
    splits = books.split_legs(res)
    gross = (M[M["par_c"] == "Taiwan"].groupby(["rep_c", "code_hs6", "period"], as_index=False)
             ["value_usd"].sum())
    rx = (res[res["flow"] == "RX"].groupby(["rep_c", "code_hs6", "period"], as_index=False)["value_usd"]
          .sum().rename(columns={"value_usd": "rx"}))
    parts = []
    for (rep, code), g in gross.groupby(["rep_c", "code_hs6"]):
        g = g.sort_values("period").reset_index(drop=True)
        if rep in splits:
            imp_hc = M[(M["rep_c"] == rep) & (M["code_hs6"] == code)][["origin", "period", "value_usd"]]
            wts = books._origin_weights(imp_hc, g["period"].tolist(), books.ATTR_WINDOW_MONTHS)
            tw_share = np.array([float(wts.get(p, pd.Series(dtype=float)).get("Taiwan", 0.0))
                                 for p in g["period"]])
            rxc = rx[(rx["rep_c"] == rep) & (rx["code_hs6"] == code)].set_index("period")["rx"]
            rx_val = np.array([float(rxc.get(p, 0.0)) for p in g["period"]])
            v = g["value_usd"].to_numpy()
            frac = np.where(v > 0, np.clip((v - rx_val * tw_share) / v, 0.0, 1.0), 0.0)
            g["value_usd"] = v * frac
        parts.append(g)
    return pd.concat(parts, ignore_index=True).rename(columns={"rep_c": "dest"})
