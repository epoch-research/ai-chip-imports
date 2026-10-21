"""Accelerator content of customs dollars: the share of each traded dollar that is the chip.

Excess trade over the baseline is in border dollars of whatever crossed: servers, boards, or
ordinary electronics that happen to share an HS code. This module turns it into chip dollars,
per destination x code x month, from each book's own physical evidence.

**The classifier.** Every reporter x origin x quarter cell (member x origin x quarter for the EU)
of a counted import line is read from its own physical fields:

* by unit value, where the book carries pieces on at least 90% of the line's value (machine
  codes only). The cell's $/unit is compared with the eligible accelerator ASP (Epoch's
  sell-in price, cost-weighted over the models the destination can buy): below 0.15x it holds no
  accelerator (content 0.05); 0.35-2.5x is an accelerator board (0.93); above 5x a multi-
  accelerator system (the eligible product mix's system content); log-linear blends in between.
* else by value density, where net weight covers at least 90% of the value: log-linear from the
  low tier at $1,000/kg (boards) or $600/kg (machines) to the board tier or the system mix at
  $3,000/kg or $2,000/kg. Boards are read by weight only: a board's piece count says nothing
  about what is on it.

Cells are weighted by their excess over their own 2022 pace (at the destination's baseline
index) into a content per destination x code x quarter, smoothed by a centered median over three
calendar-adjacent quarters. Monthly readings replace the quarterly ones where a destination x
code x month has at least $50M/3 of classified excess and 90% field coverage; a quarter with
less than $50M of classified excess falls back to the pooled AI-window prior of its line.

**Native tariff lines** keep precedence where a book's own tree separates form factors that HS-6
hides (``NATIVE_TIERS``): US 847180 and 847330, Korea 847330. Explicit memory-module lines get
zero accelerator content (US 8473301140, Korea 8473304060, Japan 847330011; Japan's 847330 is
split into memory and remainder by origin-month before classification). HBM included in an
accelerator's price stays in scope.

**Origin-group slices** (the earlier, piece-based ladder) remain for the China, Hong Kong,
Singapore and EU machine lines, where they serve lines whose piece coverage is 50-90% and feed
the origin-mix diagnostic.

**References.** The eligible ASP and product mix come from Epoch's sales: China and Hong Kong
are priced against models sold primarily for the Chinese market, every other destination
against the rest. Once China-specific sell-in falls below $500M a quarter (with one quarter of
arrival lag), trailing quarters use the rest-of-world reference, so compliant cards that Epoch
does not track are not read as accelerators while diverted flagship hardware still is.

Content constants are on the sale basis (Epoch's prices are market ASPs): merchant systems 0.73,
GB200/GB300 racks 0.79 (0.73-0.86 across the price anchors), MI300A 0.64, boards 0.93, captive
TPU/Trainium 0.70 (priced near production cost, so there is no designer margin).

``ContentModel.lookup`` serves, in order: monthly reading, quarterly cell, line prior, eligible
mix (machine codes), caller's fallback.
"""
import glob
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from . import books, codes, epoch

AI_LO = codes.AI_LO
MEASURED_CODES = tuple(codes.SERVER_CODES)             # 847149/847150/847180
# Destinations with origin-group slices (piece-based) on their machine lines.
MEASURED_DESTS = ("China", "Hong Kong", "Singapore", "European Union")
# Per-destination slice codes (default: the machine codes). The US keeps its native lines for
# 847180/847330, so only its server codes are sliced.
LADDER_CODES = {"United States": ("847149", "847150")}
SALE_FLAT = {"847330": 0.93, "854231": 1.00}   # boards / bare chips at sale basis
CN_SIDE = {"China", "Hong Kong"}               # priced against the China-specific references
# Books classified from their own cells; Thailand's book is read outside the loader and
# handed in as an extra book.
CELL_LEGS = ("United States", "China", "Hong Kong", "Singapore", "Malaysia", "South Korea",
             "Japan", "United Kingdom", "Indonesia", "India", "Mexico", "European Union")
CELL_LEGS = CELL_LEGS + tuple(books.COMTRADE_LEG_NAMES[k] for k in books.COMTRADE_LEGS) + ("Thailand",)
CELL_FIELD_MIN = 0.9                                   # share of the line's value the field must cover
CELL_DENS = {"boards": (1000.0, 3000.0), "server": (600.0, 2000.0)}   # $/kg: low tier -> high tier
TIER_C = {"board": 0.93, "low": 0.05}                  # "server" = eligible mix share
RACK_CONTENT = 0.79                                    # GB300 rack: $3.10M of GPUs in a $3.9M rack
RACK_BAND = (0.73, 0.86)                               # range across the rack price anchors
ORIGIN_VN = {"vietnam", "viet nam"}
PROMOTE_MIN = 0.5e9      # AI-window excess per (dest, code, origin) that earns an origin its
#                          own slice instead of the pooled "other" slice
QTYCOV_MIN = 0.5                                       # min value_qtycov/value to trust a $/unit
MIN_EXCESS_Q = 50e6                                    # $/quarter; thinner cells -> line prior
SMOOTH_Q = 3                                           # centered median window, quarters


def _qty_covered(r, field):
    """The column holding the value that pieces cover (so a line whose pieces sit on part of its
    native detail is priced only over that part), or None for the whole-row reading."""
    return "value_qtycov" if (field == "qty" and "value_qtycov" in r.columns) else None


def _median3(c, quarters):
    """Centered SMOOTH_Q-quarter median of ``c`` (aligned with ``quarters``, 'YYYYQn'); a quarter
    missing from the series is a gap, not a neighbour."""
    idx = pd.PeriodIndex(list(quarters), freq="Q")
    full = pd.period_range(idx.min(), idx.max(), freq="Q")
    sm = (pd.Series(c.to_numpy(dtype=float), index=idx).reindex(full)
          .rolling(SMOOTH_Q, center=True, min_periods=1).median())
    return pd.Series(sm.reindex(idx).to_numpy(), index=c.index)


# Unit-value ladder zones, in multiples of the eligible ASP (log-linear blends between zones).
LOW_MAX, BOARD_LO, BOARD_HI, SERVER_LO = 0.15, 0.35, 2.5, 5.0
DEFAULT_CN_EXCLUDED = {"china"}          # China's own-origin re-imports are excluded
CN_MATERIAL_FLOOR = 500e6   # $/quarter of China-specific sell-in (incl. one quarter of lag)
#                             below which trailing quarters use the rest-of-world reference

# Native tariff lines per (dest, code): "board" = accelerator boards (0.93); "low" = no counted
# accelerator content (memory modules, adapters, CPUs, parts; 0.05); "server" = ambiguous machine
# lines (the eligible mix). Unlisted lines take NATIVE_DEFAULT. Identities from the books' own
# descriptions: US 8473301140 memory modules, 8473301180 other printed circuit assemblies (GPU
# boards), 8471804000 units for physical incorporation (baseboards), 8471801000 control/adapter
# units; Korea 8473304060 DRAM modules, 8473309010 CPUs, 8473302000 main boards (Korea's tree has
# no GPU-board line).
NATIVE_TIERS = {
    ("United States", "847180"): {"8471804000": "board", "8471801000": "low",
                                  "8471809000": "server"},
    ("United States", "847330"): {"8473301180": "board", "8473301140": "low",
                                  "8473302000": "low", "8473305100": "low", "8473309100": "low"},
    ("South Korea", "847330"): {"8473304060": "low", "8473309010": "low", "8473302000": "low",
                                "8473301000": "low", "8473303000": "low", "8473309090": "low",
                                "8473304090": "board"},
}
NATIVE_DEFAULT = {"847180": "server", "847330": "board"}
NATIVE_FILES = {"United States": "us", "South Korea": "korea"}
MEMORY_CODES = {"Japan": "847330011", "United States": "8473301140",
                "South Korea": "8473304060"}


def base_name(name):
    """'Q1 2024 - H100/H200' / 'Q1 2024 Trainium2' / 'Instinct MI308X' -> the bare model name."""
    n = re.sub(r"^(Q[1-4] 20\d\d|FY\d\d Q[1-4])\s*-?\s*", "", str(name)).strip()
    return re.sub(r"^Instinct\s+", "", n)


def fam_content(name, rack=RACK_CONTENT):
    """Sale-basis accelerator content of a model family's systems (see module docstring)."""
    n = base_name(name)
    if "Trainium" in n or "Inferentia" in n:
        return 0.70
    if n.startswith("TPU"):
        return 0.70
    if n.startswith("B200") or n.startswith("B300"):
        return rack
    if n.startswith("MI300A"):
        return 0.64
    return 0.73


def china_spec_bases():
    """Model base names flagged 'Primarily for Chinese market' in Epoch's chip_types.csv."""
    ct = pd.read_csv(os.path.join(epoch.EPOCH_DIR, "chip_types.csv"))
    flag = ct["Primarily for Chinese market"].astype(str).str.lower().eq("true")
    return set(ct.loc[flag, "Name"].map(base_name))


def _tier_content(ratio, server_c):
    """Unit-value ladder: $/unit over the eligible ASP -> content. Non-monotone by design: ~1x
    the ASP is an accelerator board; far above it a multi-accelerator machine, whose non-GPU share
    brings content back to the system mix (``server_c``); near zero no accelerator at all."""
    if not np.isfinite(ratio) or ratio <= 0:
        return np.nan
    lo, bd = TIER_C["low"], TIER_C["board"]
    if ratio < LOW_MAX:
        return lo
    if ratio < BOARD_LO:
        w = (np.log(ratio) - np.log(LOW_MAX)) / (np.log(BOARD_LO) - np.log(LOW_MAX))
        return lo + (bd - lo) * w
    if ratio <= BOARD_HI:
        return bd
    if ratio < SERVER_LO:
        w = (np.log(ratio) - np.log(BOARD_HI)) / (np.log(SERVER_LO) - np.log(BOARD_HI))
        return bd + (server_c - bd) * w
    return server_c


class ContentModel:
    """Accelerator content per destination x code x month/quarter, built from the books.

    ``res`` is the loaded trade frame, ``t`` Epoch's timelines (``epoch.load_timelines``),
    ``bindex`` the baseline index, ``extra_books`` books read outside the loader ({dest: frame},
    e.g. Thailand's). ``lookup`` serves the content values.
    """

    def __init__(self, res, t, bindex=None, rack=RACK_CONTENT,
                 cn_excluded=DEFAULT_CN_EXCLUDED, verbose=True, extra_books=None):
        self.rack = rack
        self._extra_books = extra_books or {}
        self.bindex = bindex
        self.fallback_counts = Counter()
        self.mix_dests = set()          # (dest, code) served by the eligible mix (level 3)
        # Epoch per-model frame: base name, CN flag, quarter, units, cost dollars
        cn = china_spec_bases()
        d = t.copy()
        d["base"] = d["Name"].map(base_name)
        d["cn_spec"] = d["base"].isin(cn)
        d["units"] = d["Number of Units"]
        d["cost"] = d["Cost Estimate (USD)"]
        self._ep = d[["base", "cn_spec", "quarter", "units", "cost"]]
        self._quarters = sorted(self._ep["quarter"].unique())
        # trailing quarters with immaterial China-specific sell-in use the rest-of-world reference
        cn_cost = self._ep[self._ep["cn_spec"]].groupby("quarter")["cost"].sum()
        alive = {q: (float(cn_cost.get(q, 0.0))
                     + (float(cn_cost.get(self._quarters[i - 1], 0.0)) if i else 0.0))
                 >= CN_MATERIAL_FLOOR for i, q in enumerate(self._quarters)}
        last_alive = max((q for q, a in alive.items() if a), default=None)
        self._cn_dead = ({q for q in self._quarters if q > last_alive}
                         if last_alive is not None else set())
        self._mix_cache, self._asp_cache = {}, {}
        self.evidence = self._build_evidence(res, cn_excluded)
        self.cells, self.prior = self._classify(self.evidence)
        self.evidence = self._slice_frame        # classified slice grain (uv/ratio/tier attached)
        self._native_cells()
        self._cellcls_frame, self._cellcls_pairs = None, {}
        self._cell_classifier(res, cn_excluded)
        self.monthly_cells = {}
        self._monthly_frame = None
        self._monthly_classifier(res, cn_excluded)
        if verbose:
            n_cells = len(self.cells)
            dests = sorted({d_ for d_, _, _ in self.cells})
            promo = sorted({o for _, _, o in self.promoted})
            print(f"[content-model] rack={self.rack:.2f}: {n_cells} measured (dest, code, quarter) "
                  f"cells for {dests}; priors for {sorted(self.prior)}; "
                  f"promoted origins (>=${PROMOTE_MIN/1e9:.1f}B excess): {promo}")
            print(f"[content-model] cell classifier: {len(self._cellcls_pairs)} "
                  f"(dest, code) pairs measured from their own cells; the slice ladder serves "
                  f"the rest")
            for (d_, c), (f, n, pr) in self._cellcls_pairs.items():
                print(f"    {d_} {c} [{'units' if f == 'qty' else 'kg'}]: {n} cells, "
                      f"AI-window content {pr:.2f}")
            eu_cells = sorted({c for d_, c, _ in self.cells if d_ == "European Union"})
            print(f"[content-model] EU in the ladder: "
                  f"measured codes {eu_cells or 'NONE -- all cells failed the gates'}")
            if self._cn_dead:
                dd = sorted(self._cn_dead)
                print(f"[content-model] China-specific reference: "
                      f"ROW-reference fallback in {dd[0]}..{dd[-1]} "
                      f"(CN-spec sell-in < ${CN_MATERIAL_FLOOR/1e6:.0f}M/q incl. 1-q lag)")

    # --- Epoch-side references -------------------------------------------------------------------
    def _side(self, dest):
        return "cn" if dest in CN_SIDE else "row"

    def mix(self, side, rack=None):
        """System content of the eligible product mix per quarter: cost-weighted ``fam_content``
        over the models a side can buy ('cn': China-specific; 'row': the rest). Gaps are filled
        forward and backward."""
        rack = self.rack if rack is None else rack
        key = (side, round(rack, 4))
        if key not in self._mix_cache:
            d = self._ep[self._ep["cn_spec"] == (side == "cn")].copy()
            d["c"] = d["base"].map(lambda b: fam_content(b, rack))
            s = d.groupby("quarter").apply(
                lambda g: (g["c"] * g["cost"]).sum() / g["cost"].sum() if g["cost"].sum() > 0
                else np.nan, include_groups=False)
            s = s.reindex(self._quarters).ffill().bfill()
            if side == "cn" and self._cn_dead:
                row = self.mix("row", rack)
                s = s.where(~s.index.isin(self._cn_dead), row.reindex(s.index))
            self._mix_cache[key] = s
        return self._mix_cache[key]

    def asp(self, side):
        """Eligible accelerator ASP per quarter (sell-in $ per unit), same eligibility rule."""
        if side not in self._asp_cache:
            d = self._ep[self._ep["cn_spec"] == (side == "cn")]
            s = d.groupby("quarter").apply(
                lambda g: g["cost"].sum() / g["units"].sum() if g["units"].sum() > 0 else np.nan,
                include_groups=False)
            s = s.reindex(self._quarters).ffill().bfill()
            if side == "cn" and self._cn_dead:
                row = self.asp("row")
                s = s.where(~s.index.isin(self._cn_dead), row.reindex(s.index))
            self._asp_cache[side] = s
        return self._asp_cache[side]

    # --- trade-side evidence -----------------------------------------------------------------
    def _build_evidence(self, res, cn_excluded):
        """Quarterly origin-group slices for MEASURED_DESTS x their LADDER_CODES: value, pieces,
        piece-covered value, and excess over the slice's own 2022 mean grown by the baseline
        index (parametric growth where there is none)."""
        frames = []
        for d_ in MEASURED_DESTS:
            line_codes = LADDER_CODES.get(d_, MEASURED_CODES)
            if d_ == "European Union":
                # the EU is the sum of member books' extra-EU imports
                sub = res[res["rep_c"].isin(books.EU_MEMBERS) & (res["flow"] == "M")
                          & res["code_hs6"].isin(line_codes)
                          & ~res["par_c"].isin(books.EU_MEMBERS)].copy()
                sub["rep_c"] = d_
            else:
                sub = res[(res["rep_c"] == d_) & (res["flow"] == "M")
                          & res["code_hs6"].isin(line_codes)].copy()
            frames.append(sub)
        m = pd.concat(frames, ignore_index=True)
        norm = m["partner"].map(books.NORM)
        m = m[~norm.isin({"world", "total"})]
        m = m[~((m["rep_c"] == "China") & m["partner"].map(books.NORM).isin(set(cn_excluded)))]
        origin = m["par_c"].where(m["par_c"].notna(), m["partner"].map(books.NORM))
        m["origin"] = origin
        base = np.select([origin.eq("Taiwan"),
                          m["partner"].map(books.NORM).isin(ORIGIN_VN)],
                         ["taiwan", "vietnam"], default="other")
        # Large origins get their own slice: the pooled "other" slice mixes board-priced origins
        # with cheap electronics, and a pooled $/unit would read all of it as low content.
        self.promoted = self._promoted_origins(m[base == "other"])
        key = pd.MultiIndex.from_arrays([m["rep_c"], m["code_hs6"], m["origin"]])
        m["grp"] = np.where((base == "other") & key.isin(self.promoted),
                            origin.str.lower(), base)
        g = (m.groupby(["rep_c", "code_hs6", "grp", "period"], as_index=False)
             .agg(value=("value_usd", "sum"), qty=("qty", "sum"),
                  vqcov=("value_qtycov", "sum")).rename(columns={"rep_c": "dest"}))
        # slice baseline: 2022 monthly mean x the destination's growth factor
        b = (g[g["period"].str[:4] == "2022"].groupby(["dest", "code_hs6", "grp"])["value"]
             .mean().rename("mean2022").reset_index())
        g = g.merge(b, on=["dest", "code_hs6", "grp"], how="left")
        g["mean2022"] = g["mean2022"].fillna(0.0)
        gf = (1.0 + codes.NONAI_GROWTH["systems"]) ** g["period"].map(codes.years_since_mid2022)
        if self.bindex is not None:
            gf = self.bindex.lookup(g["dest"], g["code_hs6"], g["period"], gf)
        g["excess"] = (g["value"] - g["mean2022"] * gf).clip(lower=0.0)
        g["q"] = g["period"].map(codes.quarter)
        return (g.groupby(["dest", "code_hs6", "grp", "q"], as_index=False)
                [["value", "qty", "vqcov", "excess"]].sum())

    def _promoted_origins(self, m):
        """Origins in the pooled slice whose AI-window excess over their own 2022 pace reaches
        PROMOTE_MIN within a (dest, code)."""
        g = (m.groupby(["rep_c", "code_hs6", "origin", "period"], as_index=False)
             .agg(value=("value_usd", "sum")))
        b = (g[g["period"].str[:4] == "2022"].groupby(["rep_c", "code_hs6", "origin"])["value"]
             .mean().rename("mean2022").reset_index())
        g = g.merge(b, on=["rep_c", "code_hs6", "origin"], how="left")
        g["mean2022"] = g["mean2022"].fillna(0.0)
        gf = (1.0 + codes.NONAI_GROWTH["systems"]) ** g["period"].map(codes.years_since_mid2022)
        if self.bindex is not None:
            gf = self.bindex.lookup(g["rep_c"], g["code_hs6"], g["period"], gf)
        g["excess"] = (g["value"] - g["mean2022"] * gf).clip(lower=0.0)
        ai = g[g["period"].map(codes.quarter) >= AI_LO]
        tot = ai.groupby(["rep_c", "code_hs6", "origin"])["excess"].sum()
        return set(tot[tot >= PROMOTE_MIN].index)

    def _classify(self, ev):
        """Slice tiers -> excess-weighted content per (dest, code, quarter), median-smoothed;
        thin quarters fall back to the line's pooled AI-window prior."""
        ev = ev.copy()
        # $/unit: quarterly where covered; else trailing-4Q; else pooled AI-window (per slice)
        ev = ev.sort_values(["dest", "code_hs6", "grp", "q"]).reset_index(drop=True)
        grp = ev.groupby(["dest", "code_hs6", "grp"], sort=False)
        roll_v = grp["vqcov"].transform(lambda s: s.rolling(4, min_periods=1).sum())
        roll_q = grp["qty"].transform(lambda s: s.rolling(4, min_periods=1).sum())
        roll_val = grp["value"].transform(lambda s: s.rolling(4, min_periods=1).sum())
        cov_q = np.where(ev["value"] > 0, ev["vqcov"] / ev["value"], 0.0)
        cov_r = np.where(roll_val > 0, roll_v / roll_val, 0.0)
        pool = (ev[ev["q"] >= AI_LO].groupby(["dest", "code_hs6", "grp"])
                .agg(pv=("vqcov", "sum"), pq=("qty", "sum"), pval=("value", "sum")))
        pool["puv"] = np.where(pool["pq"] > 0, pool["pv"] / pool["pq"], np.nan)
        pool["pcov"] = np.where(pool["pval"] > 0, pool["pv"] / pool["pval"], 0.0)
        uv, basis = np.full(len(ev), np.nan), np.full(len(ev), "none", dtype=object)
        q_ok = (cov_q >= QTYCOV_MIN) & (ev["qty"] > 0)
        r_ok = ~q_ok & (cov_r >= QTYCOV_MIN) & (roll_q > 0)
        uv[q_ok] = (ev.loc[q_ok, "vqcov"] / ev.loc[q_ok, "qty"])
        basis[q_ok] = "qty"
        uv[r_ok] = (roll_v[r_ok] / roll_q[r_ok])
        basis[r_ok] = "qty4q"
        p_ok = ~q_ok & ~r_ok
        pk = pd.MultiIndex.from_frame(ev.loc[p_ok, ["dest", "code_hs6", "grp"]])
        puv = pool["puv"].reindex(pk).to_numpy()
        pcv = pool["pcov"].reindex(pk).to_numpy()
        good = np.isfinite(puv) & (pcv >= QTYCOV_MIN)
        uv[np.flatnonzero(p_ok)[good]] = puv[good]
        basis[np.flatnonzero(p_ok)[good]] = "pooled"
        ev["uv"], ev["uv_basis"] = uv, basis
        # ladder classification vs the corridor-eligible ASP
        side = ev["dest"].map(self._side)
        asp = np.array([self.asp(s).get(q, np.nan) for s, q in zip(side, ev["q"])])
        srv = np.array([self.mix(s).get(q, np.nan) for s, q in zip(side, ev["q"])])
        ev["asp_ref"] = asp
        ev["ratio"] = ev["uv"] / asp
        ev["c_grp"] = [_tier_content(r, sc) for r, sc in zip(ev["ratio"], srv)]
        # cell content: excess-share-weighted over classified slices
        cl = ev[ev["c_grp"].notna() & (ev["excess"] > 0)]
        cell = (cl.groupby(["dest", "code_hs6", "q"])
                .apply(lambda g: pd.Series({
                    "c_raw": (g["c_grp"] * g["excess"]).sum() / g["excess"].sum(),
                    "excess_cl": g["excess"].sum()}), include_groups=False).reset_index())
        tot = (ev.groupby(["dest", "code_hs6", "q"])["excess"].sum().rename("excess_tot")
               .reset_index())
        cell = cell.merge(tot, on=["dest", "code_hs6", "q"], how="left")
        # classified coverage of the cell's excess; unclassified remainder -> prior treatment
        cell["cl_cov"] = np.where(cell["excess_tot"] > 0,
                                  cell["excess_cl"] / cell["excess_tot"], 0.0)
        cell = cell.sort_values(["dest", "code_hs6", "q"]).reset_index(drop=True)
        cell["c_smooth"] = (cell.groupby(["dest", "code_hs6"], sort=False)["c_raw"]
                            .transform(lambda s: _median3(s, cell.loc[s.index, "q"])))
        # corridor prior: pooled AI-window excess-weighted content per (dest, code)
        pw = cell[cell["q"] >= AI_LO]
        prior = (pw.groupby(["dest", "code_hs6"])
                 .apply(lambda g: (g["c_smooth"] * g["excess_cl"]).sum() / g["excess_cl"].sum()
                        if g["excess_cl"].sum() > 0 else np.nan, include_groups=False).dropna()
                 .to_dict())
        ok = (cell["excess_tot"] >= MIN_EXCESS_Q) & (cell["cl_cov"] >= QTYCOV_MIN)
        cells = {(r.dest, r.code_hs6, r.q): float(r.c_smooth)
                 for r in cell[ok].itertuples()}
        self._cell_frame = cell
        self._slice_frame = ev
        return cells, prior

    def _native_cells(self):
        """Contents from native tariff lines for the (dest, code) pairs in NATIVE_TIERS: each
        line's tier comes from its identity, weighted by the line's excess. Reads the clean CSVs
        directly (the loader keeps HS-6 only); native rows are those with more than six digits."""
        frames, monthly_frames = [], []
        for dest, repdir in NATIVE_FILES.items():
            native_codes = sorted({c for d, c in NATIVE_TIERS if d == dest})
            path = sorted(glob.glob(os.path.join(books.CUSTOMS_DIR, repdir, "clean",
                                                 f"{repdir}_T1-T2_*.csv")))[-1]
            df = pd.read_csv(path, usecols=["flow", "partner", "code_native", "code_hs6",
                                            "period", "value_usd"],
                             dtype={"code_hs6": str, "code_native": str})
            d = df[(df["flow"] == "M") & df["code_hs6"].isin(native_codes)
                   & (df["code_native"].str.len() > 6)].copy()
            d = d[~d["partner"].map(books.NORM).isin({"world", "total"})]
            g = (d.groupby(["code_hs6", "code_native", "period"], as_index=False)
                 ["value_usd"].sum())
            b = (g[g["period"].str[:4] == "2022"].groupby(["code_hs6", "code_native"])
                 ["value_usd"].mean().rename("mean2022").reset_index())
            g = g.merge(b, on=["code_hs6", "code_native"], how="left")
            g["mean2022"] = g["mean2022"].fillna(0.0)
            gf = (1.0 + codes.NONAI_GROWTH["systems"]) ** g["period"].map(codes.years_since_mid2022)
            if self.bindex is not None:
                gf = self.bindex.lookup(np.repeat(dest, len(g)), g["code_hs6"], g["period"], gf)
            g["excess"] = (g["value_usd"] - g["mean2022"] * gf).clip(lower=0.0)
            g["q"] = g["period"].map(codes.quarter)
            g["dest"] = dest
            monthly_frames.append(g.copy())
            frames.append(g.groupby(["dest", "code_hs6", "code_native", "q"], as_index=False)
                          [["value_usd", "excess"]].sum())
        nv = pd.concat(frames, ignore_index=True)
        side = nv["dest"].map(self._side)
        tier = [NATIVE_TIERS[(d, c)].get(n, NATIVE_DEFAULT[c])
                for d, c, n in zip(nv["dest"], nv["code_hs6"], nv["code_native"])]
        srv = np.array([self.mix(s).get(q, np.nan) for s, q in zip(side, nv["q"])])
        nv["tier"] = tier
        nv["c_line"] = np.select([nv["tier"] == "board", nv["tier"] == "low"],
                                 [TIER_C["board"], TIER_C["low"]], default=srv)
        nv.loc[nv["code_native"].eq(nv["dest"].map(MEMORY_CODES)), "c_line"] = 0.
        cl = nv[nv["excess"] > 0]
        cell = (cl.groupby(["dest", "code_hs6", "q"])
                .apply(lambda g_: pd.Series({
                    "c_raw": (g_["c_line"] * g_["excess"]).sum() / g_["excess"].sum(),
                    "excess_tot": g_["excess"].sum()}), include_groups=False).reset_index()
                .sort_values(["dest", "code_hs6", "q"]).reset_index(drop=True))
        cell["c_smooth"] = (cell.groupby(["dest", "code_hs6"], sort=False)["c_raw"]
                            .transform(lambda s: _median3(s, cell.loc[s.index, "q"])))
        ok = cell["excess_tot"] >= MIN_EXCESS_Q
        for r in cell[ok].itertuples():
            self.cells[(r.dest, r.code_hs6, r.q)] = float(r.c_smooth)
        pw = cell[(cell["q"] >= AI_LO) & ok]
        for (d, c), g_ in pw.groupby(["dest", "code_hs6"]):
            self.prior[(d, c)] = float((g_["c_smooth"] * g_["excess_tot"]).sum()
                                       / g_["excess_tot"].sum())
        self._native_frame = nv
        self._native_monthly_frame = pd.concat(monthly_frames, ignore_index=True).merge(
            nv[["dest", "code_hs6", "code_native", "q", "c_line"]],
            on=["dest", "code_hs6", "code_native", "q"], validate="many_to_one")

    # --- the cell classifier -------------------------------------------------------------------
    def _cell_rows(self, res, dest, code, cn_excluded):
        """A book's counted import rows on one code: the members' extra-EU rows for the EU, the
        reporter's own rows otherwise; world totals and the reporter's own-origin re-imports out."""
        if dest == "European Union":
            r = res[res["rep_c"].isin(books.EU_MEMBERS) & (res["flow"] == "M")
                    & (res["code_hs6"] == code) & ~res["par_c"].isin(books.EU_MEMBERS)]
            norm = r["partner"].map(books.NORM)
            return r[~norm.isin({"world", "total"})]
        if dest in self._extra_books:
            b = self._extra_books[dest]
            r = b[(b["flow"] == "M") & (b["code_hs6"] == code)].copy()
            for col in ("value_usd", "qty", "weight_kg"):
                r[col] = pd.to_numeric(r[col], errors="coerce") if col in r.columns else np.nan
            norm = r["partner"].map(books.NORM)
            return r[~norm.isin({"world", "total"}) & ~(norm == books.NORM(dest))]
        r = res[(res["rep_c"] == dest) & (res["flow"] == "M") & (res["code_hs6"] == code)]
        norm = r["partner"].map(books.NORM)
        drop = norm.isin({"world", "total"}) | (r["par_c"] == dest) | (norm == books.NORM(dest))
        if dest == "China":
            drop |= norm.isin(set(cn_excluded))
        r = r[~drop]
        if dest == "Japan" and code == "847330":
            return self._japan_memory_rows(r)
        return r

    def _japan_memory_rows(self, parent):
        """Japan's 847330 rows split by origin-month into memory modules and the remainder.

        The native lines' declared-value shares apportion the parent's FOB value, so totals and
        physical fields are preserved; each part keeps its own 2022 anchor. This assumes a common
        freight margin within the parent.
        """
        if hasattr(self, "_japan_memory_split"):
            return self._japan_memory_split
        path = books.find_national_csv("japan")
        d = pd.read_csv(path, dtype={"code_hs6": str, "code_native": str}, low_memory=False)
        d = d[d.flow.eq("M") & d.code_hs6.eq("847330") & d.code_native.str.len().gt(6)].copy()
        d["pn"] = d.partner.map(books.NORM)
        assert not d.duplicated(["pn", "period", "code_native"]).any()
        d["kind"] = np.where(d.code_native.eq(MEMORY_CODES["Japan"]), "memory", "remainder")
        g = d.groupby(["pn", "period", "kind"])[["value_usd", "weight_kg", "qty"]].sum().reset_index()
        totals = g.groupby(["pn", "period"]).value_usd.sum().rename("native_total")
        parent = parent.copy()
        parent["pn"] = parent.partner.map(books.NORM)
        assert not parent.duplicated(["pn", "period"]).any()
        r = g.merge(totals, on=["pn", "period"]).merge(
            parent[["pn", "period", "partner", "value_usd", "weight_kg"]].rename(
                columns={"value_usd": "parent_fob", "weight_kg": "parent_kg"}),
            on=["pn", "period"], how="inner")
        assert set(map(tuple, r[["pn", "period"]].drop_duplicates().values)) == set(
            map(tuple, parent[["pn", "period"]].values))
        r["value_usd"] = np.where(r.native_total.gt(0), r.parent_fob*r.value_usd/r.native_total, 0.)
        check = r.groupby(["pn", "period"])[["value_usd", "weight_kg"]].sum()
        target = parent.set_index(["pn", "period"]).reindex(check.index)
        np.testing.assert_allclose(check.value_usd, target.value_usd, rtol=1e-10, atol=.01)
        np.testing.assert_allclose(check.weight_kg, target.weight_kg.fillna(0), rtol=1e-8, atol=.01)
        r["partner"] = r.partner + "|" + r.kind
        r["code_hs6"] = "847330"
        self._japan_memory_split = r
        return r

    @staticmethod
    def _cell_field(r, code):
        """The field that classifies the line: pieces where they cover CELL_FIELD_MIN of the value
        (machine codes), else net weight where it does; boards by weight only."""
        v = r["value_usd"].sum()
        if v <= 0:
            return None
        kg = r.loc[r["weight_kg"].fillna(0) > 0, "value_usd"].sum() / v
        n = r.loc[r["qty"].fillna(0) > 0, _qty_covered(r, "qty") or "value_usd"].sum() / v
        if code == "847330":
            return "weight_kg" if kg >= CELL_FIELD_MIN else None
        if n >= CELL_FIELD_MIN:
            return "qty"
        return "weight_kg" if kg >= CELL_FIELD_MIN else None

    def _cell_table(self, res, dest, code, field, cn_excluded, monthly=False):
        """Cells (origin x quarter, or x month with ``monthly``; member x origin for the EU) on one
        (dest, code): value, physical quantity, the cell's own 2022-pace baseline at the
        destination's index, and the content read off its $/unit or $/kg."""
        r = self._cell_rows(res, dest, code, cn_excluded)
        r = r[r[field].fillna(0) > 0]
        if r.empty:
            return None
        keys = ["rep_c", "partner"] if dest == "European Union" else ["partner"]
        qcov = _qty_covered(r, field)
        vq = {"vq": (qcov, "sum")} if qcov else {}
        om = (r.groupby(keys + ["period"], dropna=False)
              .agg(v=("value_usd", "sum"), p=(field, "sum"), **vq).reset_index())
        om["cell"] = om[keys].astype(str).agg("|".join, axis=1)
        b22 = om[om["period"].str[:4] == "2022"].groupby("cell")["v"].mean()
        gf = (1.0 + codes.NONAI_GROWTH["systems"]) ** om["period"].map(codes.years_since_mid2022)
        if self.bindex is not None:
            gf = self.bindex.lookup(np.repeat(dest, len(om)), np.repeat(code, len(om)),
                                    om["period"], gf)
        om["bv"] = om["cell"].map(b22).fillna(0.0) * np.asarray(gf, dtype=float)
        om["q"] = om["period"].map(codes.quarter)
        oq = (om.groupby(["cell", "q"], as_index=False)
              .agg(v=("v", "sum"), p=("p", "sum"), bv=("bv", "sum"),
                   **({"vq": ("vq", "sum")} if qcov else {})))
        if monthly:
            oq = om.copy()
        # $/unit on the value the pieces cover; the cell's excess stays on its whole value
        oq["uv"] = (oq.pop("vq") if qcov else oq["v"]) / oq["p"]
        side = self._side(dest)
        srv = np.array([self.mix(side).get(q, np.nan) for q in oq["q"]])
        if field == "qty":
            asp = np.array([self.asp(side).get(q, np.nan) for q in oq["q"]])
            oq["asp_ref"] = asp
            oq["ratio"] = oq["uv"] / asp
            oq["c"] = [_tier_content(x, sc) for x, sc in zip(oq["ratio"], srv)]
        else:
            lo, hi = CELL_DENS["boards" if code == "847330" else "server"]
            c_hi = TIER_C["board"] if code == "847330" else srv
            w = np.clip((np.log(np.maximum(oq["uv"].to_numpy(dtype=float), 1e-9)) - np.log(lo))
                        / (np.log(hi) - np.log(lo)), 0.0, 1.0)
            oq["asp_ref"] = np.nan
            oq["ratio"] = np.nan
            oq["c"] = TIER_C["low"] + (c_hi - TIER_C["low"]) * w
        oq["excess"] = (oq["v"] - oq["bv"]).clip(lower=0.0)
        if dest == "Japan" and code == "847330":
            oq.loc[oq.cell.str.endswith("|memory"), "c"] = 0.
        return oq

    def _cell_classifier(self, res, cn_excluded):
        """For every counted book and code with a usable field: the excess-weighted cell content
        per quarter, median-smoothed, with the pooled AI-window prior for thin quarters.
        Replaces the slice ladder's entries for the pairs it measures."""
        frames, installed = [], {}
        for d_ in CELL_LEGS:
            for code in MEASURED_CODES + ("847330",):
                if (d_, code) in NATIVE_TIERS:
                    continue
                r = self._cell_rows(res, d_, code, cn_excluded)
                field = self._cell_field(r, code)
                if field is None:
                    continue
                oq = self._cell_table(res, d_, code, field, cn_excluded)
                if oq is None:
                    continue
                cl = oq[oq["c"].notna() & (oq["excess"] > 0)]
                if cl.empty:
                    continue
                g = (cl.assign(wc=cl["excess"] * cl["c"]).groupby("q")
                     .agg(w=("excess", "sum"), wc=("wc", "sum")).sort_index())
                g["c_raw"] = g["wc"] / g["w"]
                g["c_smooth"] = _median3(g["c_raw"], g.index)
                ok = g["w"] >= MIN_EXCESS_Q
                pw = g[g.index >= AI_LO]
                if pw["w"].sum() <= 0:
                    continue
                prior = float((pw["c_smooth"] * pw["w"]).sum() / pw["w"].sum())
                for k in [k for k in self.cells if k[0] == d_ and k[1] == code]:
                    del self.cells[k]
                for q, row in g[ok].iterrows():
                    self.cells[(d_, code, q)] = float(row["c_smooth"])
                self.prior[(d_, code)] = prior
                installed[(d_, code)] = (field, int(ok.sum()), prior)
                oq = oq.assign(dest=d_, code_hs6=code, field=field)
                oq = oq.merge(g[["c_raw", "c_smooth", "w"]].rename(columns={"w": "excess_tot"}),
                              left_on="q", right_index=True, how="left")
                frames.append(oq)
        self._cellcls_pairs = installed
        self._cellcls_frame = pd.concat(frames, ignore_index=True) if frames else None

    def _monthly_classifier(self, res, cn_excluded):
        """Unsmoothed monthly readings for the classified pairs and the native lines (same fields
        and reference prices); months with too little evidence keep the quarterly reading."""
        frames = []
        for (dest, code), (field, _, _) in self._cellcls_pairs.items():
            r = self._cell_rows(res, dest, code, cn_excluded)
            total = r.groupby("period").value_usd.sum()
            covered = (r[r[field].fillna(0).gt(0)].groupby("period")[
                _qty_covered(r, field) or "value_usd"].sum())
            f = self._cell_table(res, dest, code, field, cn_excluded, monthly=True)
            f = f.assign(dest=dest, code_hs6=code, field=field)
            f["coverage"] = f.period.map(covered / total)
            frames.append(f)
        self._monthly_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.install_monthly(self._monthly_frame)
        n = self._native_monthly_frame.rename(columns={"c_line": "c"}).copy()
        n["coverage"] = 1.0
        self.install_monthly(n)

    def install_monthly(self, frame):
        """Install or replace the monthly readings of the (dest, code) pairs in ``frame``."""
        if frame.empty:
            return
        pairs = set(map(tuple, frame[["dest", "code_hs6"]].drop_duplicates().values))
        self.monthly_cells = {k: v for k, v in self.monthly_cells.items() if k[:2] not in pairs}
        cl = frame[frame.c.notna() & frame.excess.gt(0)]
        g = cl.assign(wc=cl.excess * cl.c).groupby(["dest", "code_hs6", "period"]).agg(
            w=("excess", "sum"), wc=("wc", "sum"), coverage=("coverage", "first"))
        g = g[g.w.ge(MIN_EXCESS_Q / 3) & g.coverage.ge(CELL_FIELD_MIN)]
        self.monthly_cells.update((k, float(row.wc / row.w)) for k, row in g.iterrows())

    # --- lookup ----------------------------------------------------------------------------------
    def lookup(self, dests, code_values, quarters, fallback, rack=None, periods=None):
        """Content per row: monthly reading (with ``periods``) -> quarterly cell -> line prior ->
        eligible mix (machine codes; ``rack`` overrides the rack constant) -> ``fallback``.
        Counts of each level are kept in ``self.fallback_counts``."""
        key = pd.DataFrame({"dest": np.asarray(dests), "code": np.asarray(code_values),
                            "q": np.asarray(quarters)})
        out = np.asarray(fallback, dtype=float).copy()
        served = np.full(len(key), "fallback", dtype=object)
        is_server = key["code"].isin(MEASURED_CODES).to_numpy()
        # level 3 first (overwritten by better levels below)
        for side in ("cn", "row"):
            mrows = is_server & (key["dest"].map(self._side) == side).to_numpy()
            if mrows.any():
                vals = self.mix(side, rack).reindex(key.loc[mrows, "q"]).to_numpy()
                okm = np.isfinite(vals)
                idx = np.flatnonzero(mrows)[okm]
                out[idx] = vals[okm]
                served[idx] = "mix"
        pr = pd.MultiIndex.from_frame(key[["dest", "code"]])
        pvals = pd.Series(self.prior).reindex(pr).to_numpy() if self.prior else\
            np.full(len(key), np.nan)
        okp = np.isfinite(pvals)
        out[okp] = pvals[okp]
        served[okp] = "prior"
        if self.cells:
            cvals = pd.Series(self.cells).reindex(
                pd.MultiIndex.from_frame(key[["dest", "code", "q"]])).to_numpy()
            okc = np.isfinite(cvals)
            out[okc] = cvals[okc]
            served[okc] = "cell"
        if periods is not None and getattr(self, "monthly_cells", None):
            mk = pd.MultiIndex.from_arrays([dests, code_values, periods])
            vals = pd.Series(self.monthly_cells).reindex(mk).to_numpy()
            ok = np.isfinite(vals)
            out[ok] = vals[ok]
            served[ok] = "monthly"
        self.fallback_counts.update(served.tolist())
        self.mix_dests |= set(map(tuple, key.loc[served == "mix",
                                                 ["dest", "code"]].drop_duplicates().to_numpy()))
        return out

    # --- reporting -------------------------------------------------------------------------------
    def report(self):
        """The evidence behind the quarterly contents, one row per slice, native line or cell,
        with the cell contents attached (the ``gpu_content_cells`` output)."""
        ev = self.evidence.copy()
        cf = self._cell_frame[["dest", "code_hs6", "q", "c_raw", "c_smooth", "cl_cov",
                               "excess_tot"]]
        ev = ev.merge(cf, on=["dest", "code_hs6", "q"], how="left")
        ev["source"] = "qty"
        nv = self._native_frame.rename(columns={"value_usd": "value"}).copy()
        nv["grp"] = "native:" + nv["code_native"]
        nv["uv_basis"] = nv["tier"]
        nv["c_grp"] = nv["c_line"]
        nv["source"] = "native"
        parts = [ev, nv.reindex(columns=ev.columns)]
        if self._cellcls_frame is not None:          # AI-window cells and any cell with excess
            cf = self._cellcls_frame
            cf = cf[(cf["q"] >= AI_LO) | (cf["excess"] > 0)]
            cc = pd.DataFrame({
                "dest": cf["dest"], "code_hs6": cf["code_hs6"], "grp": "cell:" + cf["cell"],
                "q": cf["q"], "value": cf["v"], "qty": cf["p"], "vqcov": cf["v"],
                "excess": cf["excess"], "uv": cf["uv"],
                "uv_basis": np.where(cf["field"] == "qty", "qty", "kg"),
                "asp_ref": cf["asp_ref"], "ratio": cf["ratio"], "c_grp": cf["c"],
                "c_raw": cf["c_raw"], "c_smooth": cf["c_smooth"], "cl_cov": 1.0,
                "excess_tot": cf["excess_tot"], "source": "cell"})
            parts.append(cc.reindex(columns=ev.columns))
        out = pd.concat(parts, ignore_index=True)
        out["measured_cell"] = [
            (d, c, q) in self.cells for d, c, q in zip(out["dest"], out["code_hs6"], out["q"])]
        return out.sort_values(["dest", "code_hs6", "q", "grp"])
