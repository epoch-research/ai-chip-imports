"""The 2021 reference year: historical books that anchor the non-AI baseline and warm the ledgers.

The non-AI reference for every destination x code is its total retained 2021 trade divided by
twelve, scaled by a measured index normalized to its 2021 mean (``override_baselines``). AI
trade in 2021 is assumed negligible. 2021 also warms up both the import/export netting carry
and the excess ledger, so the first published quarter starts from a settled state; 2021 itself
is never part of a published cohort.

Inputs (``data/external/baseline_2021/``, built by ``ai_chip_imports.prepare.history_2021``):

* ``measured_history.csv.gz`` -- the sixteen measured books' 2021 months, normalized and on the
  FOB basis: native Eurostat books for the EU members, HMRC for the UK (with its EU imports
  reconstructed from EU export records, see ``uk_reconstruction``), UN Comtrade monthly
  submissions for the rest. Hong Kong, Singapore and Malaysia carry their re-export /
  domestic-export split, the US and Canada their reported split.
* Taiwan's native 2021 exports (``data/customs/taiwan/clean/taiwan_history_T1_X_2021_*.csv``),
  used for the residual bucket's reference and ledger warm-up only.

Historical rows enter netting only; the content classification keeps its own observations.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths

INPUT = paths.EXTERNAL_DIR / "baseline_2021"
# Reporters whose import rows the 2021 pack holds as FOB-type values. Australia's Comtrade value
# is CIF but the pack was prepared treating it as FOB, so its rows are put on the FOB basis on
# load (``fob_pack_imports``). A pack prepared differently must restate this set.
PACK_FOB_EXEMPT = frozenset({"us", "mexico", "canada", "australia"})
CODES = ("847149", "847150", "847180", "847330")
# Books whose netting reads a declared re-export split, and the flows it needs.
SPLIT_LEGS = {"United States", "Singapore", "Malaysia", "Hong Kong", "Canada"}
SPLIT_FLOWS = ("M", "RX", "DX")
_WARM_SOURCE = None
_WARM_FRAME = None
_THIN_SOURCE = None
_THIN_FRAME = None

# Comtrade and native exporter spellings in the 2021 books, mapped to the labels each exporter
# uses in its current book. Used only to align the 2021 history with existing destination labels.
EXPORT_PARTNER_ALIASES = {
    "China, Macao SAR": "Macau", "Viet Nam": "Vietnam",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Bosnia Herzegovina": "Bosnia & Herzegovina", "Bosnia-Hrzgovin": "Bosnia & Herzegovina",
    "Dominican Rep.": "Dominican Republic", "Dominican Rep": "Dominican Republic",
    "Dominic Rep": "Dominican Republic",
    "United Rep. of Tanzania": "Tanzania", "Tanzania Rep": "Tanzania",
    "Rep. of Moldova": "Moldova", "Lao People's Dem. Rep.": "Laos", "Lao Pd Rp": "Laos",
    "Cayman Isds": "Cayman Islands", "Cayman Is": "Cayman Islands",
    "Br. Virgin Isds": "British Virgin Islands", "Br Virgn Is": "British Virgin Islands",
    "Br. Indian Ocean Terr.": "British Indian Ocean Territory",
    "Br Ind Oc Terr": "British Indian Ocean Territory",
    "Turks and Caicos Isds": "Turks And Caicos Islands", "Turks C Is": "Turks And Caicos Islands",
    "Central African Rep.": "Central African Republic", "Cent Afr Rep": "Central African Republic",
    "C Afri Rep": "Central African Republic", "Marshall Isds": "Marshall Islands",
    "Marshall Island": "Marshall Islands", "Naoero": "Nauru", "Nauru Rp": "Nauru",
    "FS Micronesia": "Micronesia", "Christmas Isds": "Christmas Island",
    "Cocos Isds": "Cocos Islands", "Cocos (Keeling) Islands": "Cocos Islands",
    "Solomon Isds": "Solomon Islands", "Solomon Is": "Solomon Islands",
    "Falkland Isds (Malvinas)": "Falkland Islands",
    "Falkland Islands (Islas Malvinas)": "Falkland Islands",
    "Faroe Isds": "Faroe Islands", "N. Mariana Isds": "Northern Mariana Islands",
    "N Mariana Is": "Northern Mariana Islands", "Cook Isds": "Cook Islands",
    "Norfolk Isds": "Norfolk Island", "State of Palestine": "Palestine",
    "Palestinian Territory Occupied": "Palestine", "Egypt A Rp": "Egypt",
    "Panama Republic": "Panama", "Papua N Gna": "Papua New Guinea",
    "Congo P Rep": "Congo", "Fiji Is": "Fiji", "Trinidad": "Trinidad And Tobago",
    "Trinidad:Tobago": "Trinidad And Tobago", "Yemen Republc": "Yemen",
    "Antigua": "Antigua And Barbuda", "Sao Tome": "Sao Tome And Principe",
    "Us Virgin Is": "U.S. Virgin Islands", "Burkina": "Burkina Faso",
    "Equat Guinea": "Equatorial Guinea", "Equtl Guinea": "Equatorial Guinea",
    "Ivory Coast": "Cote D'Ivoire", "Cote Divoire": "Cote D'Ivoire",
}


def align_export_partners(h, res):
    """Relabel 2021 export destinations with the label the same exporter uses in ``res``."""
    from .books import NORM
    from .ledger import clean_thin
    aliases = {NORM(k): v for k, v in EXPORT_PARTNER_ALIASES.items()}

    def identity(name):
        return clean_thin(aliases.get(NORM(name), NORM(name).title()))
    h = h.copy()
    for leg, g in h[h.par_c.isna()].groupby("rep_c"):
        current = res[res.rep_c.eq(leg) & res.par_c.isna()
                      & res.flow.isin(["X", "RX", "DX"])].partner.drop_duplicates()
        candidates = {}
        for name in current:
            candidates.setdefault(identity(name), set()).add(clean_thin(NORM(name).title()))
        replacements = {}
        for name in g.partner.unique():
            country = identity(name)
            choices = candidates.get(country, set())
            exact = clean_thin(NORM(name).title())
            if exact in choices and NORM(name) not in aliases:
                replacements[name] = exact
            elif len(choices) > 1:
                raise ValueError(f"Ambiguous historical destination: {leg} {name}: {choices}")
            else:
                replacements[name] = next(iter(choices), country)
        h.loc[g.index, "partner"] = g.partner.map(replacements)
    return h


@lru_cache(maxsize=1)
def history():
    """The measured books' 2021 rows, validated and on the FOB basis."""
    path = INPUT / "measured_history.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path}: build it with python -m ai_chip_imports.prepare.history_2021")
    d = pd.read_csv(path, dtype={"code_hs6": str})
    if not d.period.str.startswith("2021-").all():
        raise ValueError("Historical pack must contain only 2021")
    if not np.isfinite(d.value_usd).all() or d.value_usd.lt(0).any():
        raise ValueError("Historical values must be finite and nonnegative")
    # A code-month a book does not report is zero trade (pad_history fills it); a month missing
    # from a whole book-flow is a gap that zero-filling would misread as no trade, so it fails.
    for leg, g in d.groupby("rep_c"):
        for flow in (SPLIT_FLOWS if leg in SPLIT_LEGS else ("M", "X")):
            if g.loc[g.flow.eq(flow), "period"].nunique() != 12:
                raise ValueError(f"Incomplete 2021 history: {leg} {flow} lacks whole months")
    return fob_pack_imports(d)


def fob_pack_imports(d):
    """Put the pack's import rows that were staged as FOB-type, but are CIF, on the FOB basis.

    The freight curve applies at its earliest quarter, as for the rest of the pack. A row's own
    density sets its margin where it carries weight; otherwise the pooled 2021 code x quarter
    density of the pack's weight-bearing CIF books stands in, then the code's, then the curve's
    own fallbacks.
    """
    from . import fob
    stale = PACK_FOB_EXEMPT - fob.FOB_IMPORT_REPORTERS
    m = d.flow.eq("M") & d.reporter.isin(stale) & d.value_usd.gt(0)
    if not m.any():
        return d
    model = fob.get_model(verbose=False)
    weight = d.weight_kg.fillna(0.0)
    cif = (d.flow.eq("M") & ~d.reporter.isin(PACK_FOB_EXEMPT)
           & weight.gt(0) & d.value_usd.gt(0))
    w = d.loc[cif, ["code_hs6", "period", "value_usd"]].assign(kg=weight[cif])
    by_q = w.groupby([w.code_hs6, w.period.map(fob.quarter)])[["value_usd", "kg"]].sum()
    by_c = w.groupby("code_hs6")[["value_usd", "kg"]].sum()
    by_q = (by_q.value_usd / by_q.kg).to_dict()
    by_c = (by_c.value_usd / by_c.kg).to_dict()
    sub = d.loc[m]
    q = sub.period.map(fob.quarter).to_numpy()
    pooled = np.array([by_q.get((c, q_), by_c.get(c, model.code_dens.get(c, model.median_dens)))
                       for c, q_ in zip(sub.code_hs6, q)], dtype=float)
    kg = weight[m].to_numpy()
    dens = np.where(kg > 0, sub.value_usd.to_numpy() / np.where(kg > 0, kg, 1.0), pooled)
    f = 1.0 / (1.0 + model.tau(q, dens))
    d = d.copy()
    d.loc[m, "value_usd"] = sub.value_usd.to_numpy() * f
    if "value_qtycov" in d.columns:
        d.loc[m, "value_qtycov"] = sub.value_qtycov.to_numpy() * f
    return d


def warm_res(res):
    """``res`` extended with the measured books' 2021 rows (Thailand's are added separately).

    Cached by identity of ``res``; callers must treat both frames as immutable.
    """
    global _WARM_SOURCE, _WARM_FRAME
    if _WARM_SOURCE is not res:
        hist = history()
        _WARM_FRAME = pd.concat([res, hist[hist.rep_c.ne("Thailand")]], ignore_index=True)
        _WARM_SOURCE = res
    return _WARM_FRAME


def thin_res(res, eu_members):
    """``res`` extended with the 2021 export books behind the thin-tier contributions.

    Taiwan's history is added separately (``taiwan_history``). Hong Kong, Singapore and Malaysia
    report 2021 exports only as their re-export/domestic split, so their total exports are
    rebuilt as the sum of the two; this synthetic X is used for exporter contributions only,
    never for measured netting, which reads the split itself.
    """
    global _THIN_SOURCE, _THIN_FRAME
    if _THIN_SOURCE is res:
        return _THIN_FRAME
    h = history()
    sources = set(eu_members) | {"United States", "Malaysia", "Singapore",
                                 "Hong Kong", "United Kingdom", "Indonesia", "India"}
    h = h[h.rep_c.isin(sources) & h.flow.isin(["X", "RX", "DX"])].copy()
    h = align_export_partners(h, res)
    synthetic = []
    for leg in ["Malaysia", "Singapore", "Hong Kong"]:
        g = h[h.rep_c.eq(leg)]
        if g.flow.eq("X").any():
            raise ValueError(f"Unexpected historical total X for {leg}; reconcile before combining")
        for flow in ["RX", "DX"]:
            if g[g.flow.eq(flow)].period.nunique() != 12:
                raise ValueError(f"Incomplete 2021 export source: {leg} {flow}")
        keys = ["reporter", "partner", "code_hs6", "period", "rep_c", "par_c"]
        x = g.groupby(keys, dropna=False, as_index=False)[
            ["value_usd", "weight_kg", "qty", "value_qtycov"]].sum()
        x["flow"] = "X"
        synthetic.append(x)
    h = pd.concat([h, *synthetic], ignore_index=True)
    for leg in sorted(sources):
        for flow in (["RX", "DX"] if leg == "United States" else ["X"]):
            months = h.loc[h.rep_c.eq(leg) & h.flow.eq(flow), "period"]
            if set(months) != set(pd.period_range("2021-01", "2021-12", freq="M").astype(str)):
                raise ValueError(f"Incomplete 2021 export source: {leg} {flow}")
    _THIN_FRAME = pd.concat([res, h], ignore_index=True)
    _THIN_SOURCE = res
    return _THIN_FRAME


def override_baselines(d, dest_col, baseline_index):
    """Set each destination x code baseline to (2021 total / 12) x index / (index's 2021 mean).

    Only destinations with 2021 rows change. Missing code-months must already be zero rows
    (``pad_history``); a missing index or index month is an error, never an extrapolation.
    Without a ``baseline_index`` the row's parametric growth factor is normalized instead.
    """
    h = d[d.period.str.startswith("2021-")]
    for (leg, code), rows in h.groupby([dest_col, "code_hs6"]):
        if rows.period.nunique() != 12:
            raise ValueError(f"Incomplete 2021 history: {leg} {code}")
        mask = d[dest_col].eq(leg) & d.code_hs6.eq(code)
        if baseline_index is None:
            norm = d.loc[rows.index, "gf"].mean()
        else:
            ix = baseline_index.factor(leg, code)
            if ix is None and baseline_index.composite is not None:
                ix = baseline_index.composite.get(code)
            if ix is None:
                raise ValueError(f"No historical index for {leg} {code}")
            year = ix.reindex(pd.period_range("2021-01", "2021-12", freq="M").astype(str))
            if year.isna().any():
                raise ValueError(f"Incomplete 2021 index: {leg} {code}")
            norm = year.mean()
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"Invalid 2021 index: {leg} {code}")
        d.loc[mask, "baseline"] = rows.value_usd.sum() / 12 * d.loc[mask, "gf"] / norm
    return d


def retrospective_ledger(gaps, memory):
    """Net one corridor's signed quarterly gaps; charges reduce the quarter they charge.

    A surplus first repays outstanding shortfalls from the previous ``memory`` quarters
    (newest first); what remains is counted. A shortfall charges back counted surpluses from
    the previous ``memory`` quarters (newest first), reducing the quarter in which each was
    booked; what remains stays outstanding. Counted excess is therefore never negative.
    """
    short, surplus = [], []
    out = np.zeros(len(gaps))
    for t, gap in enumerate(gaps):
        short = [e for e in short if t - e[1] <= memory]
        surplus = [e for e in surplus if t - e[1] <= memory]
        if gap > 0:
            rem = gap
            while short and rem > 1e-9:
                pay = min(rem, short[-1][0])
                short[-1][0] -= pay
                rem -= pay
                if short[-1][0] <= 1e-9:
                    short.pop()
            out[t] = rem
            if rem > 1e-9:
                surplus.append([rem, t])
        elif gap < 0:
            rem = -gap
            while surplus and rem > 1e-9:
                pay = min(rem, surplus[-1][0])
                surplus[-1][0] -= pay
                out[surplus[-1][1]] -= pay
                rem -= pay
                if surplus[-1][0] <= 1e-9:
                    surplus.pop()
            if rem > 1e-9:
                short.append([rem, t])
    return np.maximum(out, 0)


@lru_cache(maxsize=1)
def taiwan_history():
    """Taiwan's native 2021 exports by destination, validated (complete, no duplicates)."""
    hits = sorted(paths.clean_dir("taiwan").glob("taiwan_history_T1_X_2021_pulled*.csv"))
    if not hits:
        raise FileNotFoundError("no taiwan_history_T1_X_2021_pulled*.csv in "
                                f"{paths.clean_dir('taiwan')}")
    d = pd.read_csv(hits[-1], dtype={"code_hs6": str})
    expected = {(c, f"2021-{m:02}") for c in CODES for m in range(1, 13)}
    if set(zip(d.code_hs6, d.period)) != expected or not d.flow.eq("X").all():
        raise ValueError("Incomplete or invalid Taiwan 2021 export history")
    if d.duplicated(["partner", "code_hs6", "period"]).any():
        raise ValueError("Duplicate Taiwan historical destination/code/month")
    if not np.isfinite(d.value_usd).all() or d.value_usd.lt(0).any():
        raise ValueError("Taiwan historical values must be finite and nonnegative")
    return d


def pad_history(d):
    """Add zero rows for every 2021 month a destination x code lacks (including corridors that
    first appear after 2021): absent trade is zero trade."""
    keys = ["dest", "code_hs6", "period"]
    pairs = d.loc[d.code_hs6.isin(CODES), keys[:2]].drop_duplicates()
    months = pd.DataFrame({"period": [f"2021-{m:02}" for m in range(1, 13)]})
    grid = pairs.merge(months, how="cross")
    missing = grid.merge(d[keys], on=keys, how="left", indicator=True)
    missing = missing.loc[missing._merge.eq("left_only"), keys].assign(value_usd=0.)
    return pd.concat([d, missing], ignore_index=True)
