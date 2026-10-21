"""Measured non-AI baseline index per destination x code.

A parametric baseline ("2022 trade grown at 5% a year") assumes ordinary IT trade kept growing
smoothly. It did not: the post-COVID IT downturn and the end of GPU crypto-mining pushed non-AI
IT trade well below its 2022 level through 2023, and early AI arrivals would have filled that
hole instead of registering as excess. This module replaces the growth assumption with a
measurement: per destination x HS-6 code, a monthly index of observed non-AI trade (2022 mean =
1) that multiplies the baseline anchor (see ``ledger`` and ``history``).

* **Systems codes (847149/50/80):** the destination's imports of client-IT products with no AI
  content -- laptops 847130, desktops 847141, monitors 852851/52, printers 844331/32 and I/O units
  847160 -- summed into one composite. The non-AI share of the systems codes (ordinary servers
  and workstations) is assumed to move with the client-IT cycle; no HS code isolates non-AI
  servers.
* **Boards and chips (847330/854231):** the destination's imports of the same code from origins
  outside the AI supply chain (all but Taiwan, Hong Kong, Singapore, Malaysia, Vietnam and
  Thailand). Boards and logic from those origins carry no Taiwan-fabricated accelerators, so
  their path is the code's non-AI counterfactual. This is an origin-based adjustment, not an
  identified non-AI series: it can also absorb AI-related non-accelerator demand and repricing.

Inputs are UN Comtrade monthly import pulls (``data/external/baseline_index/`` for the basket and
laptops/desktops, ``baseline_index_alt/`` for monitors and printers, ``placebo_candidates/`` for
I/O units; one file per reporter, built by ``ai_chip_imports.sources.comtrade``). Each source file
contributes only its own codes, and the composite stops at the earliest source frontier so a
lagging source cannot fake a demand drop.

Construction: 3-month centered moving average, divided by its 2022 mean, clipped to [0.25, 2.5]
(a guard against thin denominators; [0.25, 6.0] where the 2022 mean is at least $200M a month,
so genuine broad moves such as the 2026 memory repricing are not capped). Beyond the Comtrade
frontier:

* origin-based levels are extended with the same construction on the destination's own national
  customs file, linked by the median ratio over the last six overlapping months;
* a cell frozen for more than six months (a reporter that stopped filing to Comtrade) continues
  on the median growth of the peer destinations whose index for the code is still live;
* otherwise the last value is carried forward.

Destinations without their own index use a world composite of the measured indices, weighted by
2022 trade (``set_composite``).
"""
import glob
import os

import numpy as np
import pandas as pd

from . import paths

IDX_DIR = str(paths.EXTERNAL_DIR / "baseline_index")
ALT_DIR = str(paths.EXTERNAL_DIR / "baseline_index_alt")
P2_DIR = str(paths.EXTERNAL_DIR / "placebo_candidates")

PLACEBO_CODES = ["847130", "847141"]            # laptops + desktops (used if the composite is missing)
SYSTEMS_IDX = ["847149", "847150", "847180"]    # baseline follows the client-IT index
ORIGIN_IDX = ["847330", "854231"]               # baseline follows the non-AI-origin index
# Origins excluded from the origin-based index: the Taiwan-fabricated supply chain and its hub
# and assembly stops (normalized names, Comtrade spellings included).
AI_ORIGIN_NORMS = {"taiwan", "other asia, nes", "hong kong", "china, hong kong sar",
                   "hong kong, china", "singapore", "malaysia", "vietnam", "viet nam", "thailand"}
IDX_CLIP = (0.25, 2.5)
MA_MONTHS = 3
# Per-cell exceptions to the excluded origins, keyed (reporter id, code). China's imports of
# 847330 from Vietnam are memory modules (flat ~$1.6k/kg through 2023-25, then the DRAM
# repricing), not AI assembly, so Vietnam stays in that cell's non-AI index.
PLACEBO_ORIGIN_OVERRIDES = {
    ("china", "847330"): frozenset(AI_ORIGIN_NORMS - {"vietnam", "viet nam"}),
}
FAT_DENOM_USD = 200e6       # 2022 monthly mean at or above which the wider clip applies
IDX_CLIP_FAT = (0.25, 6.0)
SEAM_MONTHS = 6             # overlap window for linking the national-file extension
STALE_MONTHS = 6            # a freeze longer than this is no longer an ordinary publication lag
MIN_LIVE_PEERS = 3          # fewer live peers than this leaves the cell forward-filled
# Index-file reporter id -> national book used to extend its origin-based levels. India's book
# is not used: whole months of it can collapse onto a single partner.
NATIONAL_LEG = {"us": "us", "china": "china", "hong kong": "hk", "singapore": "singapore",
                "malaysia": "malaysia", "korea": "korea", "uk": "uk", "japan": "japan",
                "indonesia": "indonesia"}
# (directory, file tag, the codes this source contributes to the client-IT composite)
WIDE_SOURCES = [
    (IDX_DIR, "baseidx", ("847130", "847141")),                      # laptops + desktops
    (ALT_DIR, "altidx", ("852851", "852852", "844331", "844332")),   # monitors + printers
    (P2_DIR, "placebo2", ("847160",)),                               # input/output units
]

# Index-file reporter id -> destination name in the estimator
REPORTER_DEST = {
    "us": "United States", "china": "China", "eu": "European Union", "hong kong": "Hong Kong",
    "singapore": "Singapore", "malaysia": "Malaysia", "korea": "South Korea",
    "uk": "United Kingdom", "japan": "Japan", "indonesia": "Indonesia", "india": "India",
}
# Destinations without a book that have their own index pulls
THIN_REPORTER_DEST = {
    "australia": "Australia", "norway": "Norway", "canada": "Canada", "switzerland": "Switzerland",
    "iceland": "Iceland", "israel": "Israel", "united arab emirates": "United Arab Emirates",
    "saudi arabia": "Saudi Arabia", "south africa": "South Africa", "new zealand": "New Zealand",
    "turkey": "Turkey", "thailand": "Thailand", "brazil": "Brazil",
}
REPORTER_DEST.update(THIN_REPORTER_DEST)
# The EU aggregate also books intra-EU partners; the estimator counts extra-EU imports only.
EU_MEMBER_NORMS = {
    "austria", "belgium", "bulgaria", "croatia", "cyprus", "czech republic", "czechia", "denmark",
    "estonia", "finland", "france", "germany", "greece", "hungary", "ireland", "italy", "latvia",
    "lithuania", "luxembourg", "malta", "netherlands", "poland", "portugal", "romania", "slovakia",
    "slovenia", "spain", "sweden",
}


def _norm(s):
    return " ".join(str(s).strip().lower().split())


def _load_files(idx_dir=IDX_DIR):
    """Newest basket index file per reporter, as one long frame (reporter id as pulled)."""
    frames = []
    for rep in REPORTER_DEST:
        pat = os.path.join(idx_dir, f"{rep.replace(' ', '_')}_comtrade_baseidx_*_pulled*.csv")
        hits = sorted(glob.glob(pat))
        if not hits:
            continue
        df = pd.read_csv(hits[-1], dtype={"code_hs6": str, "code_native": str})
        df["rep_id"] = rep
        frames.append(df[["rep_id", "partner", "code_hs6", "period", "value_usd"]])
    if not frames:
        raise SystemExit(f"No *_comtrade_baseidx_*.csv files in {idx_dir} (see module docstring).")
    d = pd.concat(frames, ignore_index=True)
    d["value_usd"] = pd.to_numeric(d["value_usd"], errors="coerce").fillna(0.0)
    d["pnorm"] = d["partner"].map(_norm)
    d = d[~((d["rep_id"] == "eu") & d["pnorm"].isin(EU_MEMBER_NORMS))]
    return d


def _smooth_norm(series):
    """Monthly value series -> index: 3-month centered mean / its 2022 mean, clipped.
    None when 2022 has fewer than six months or no positive mean."""
    s = series.sort_index()
    if s.empty:
        return None
    sm = s.rolling(MA_MONTHS, center=True, min_periods=1).mean()
    ref = sm[sm.index.str[:4] == "2022"]
    if len(ref) < 6 or ref.mean() <= 0:
        return None
    clip = IDX_CLIP_FAT if ref.mean() >= FAT_DENOM_USD else IDX_CLIP
    return (sm / ref.mean()).clip(*clip)


def _wide_level(rep):
    """One reporter's client-IT composite level: each source file restricted to its own codes,
    summed per month, truncated at the earliest source frontier. None if no source file exists."""
    parts, frontiers = [], []
    for dirpath, tag, codes in WIDE_SOURCES:
        pat = os.path.join(dirpath, f"{rep.replace(' ', '_')}_comtrade_{tag}_*_pulled*.csv")
        hits = sorted(glob.glob(pat))
        if not hits:
            continue
        df = pd.read_csv(hits[-1], dtype={"code_hs6": str})
        df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce").fillna(0.0)
        if rep == "eu":
            df = df[~df["partner"].map(_norm).isin(EU_MEMBER_NORMS)]
        df = df[df["code_hs6"].isin(codes)]
        if df.empty:
            continue
        parts.append(df.groupby("period")["value_usd"].sum())
        frontiers.append(df["period"].max())
    if not parts:
        return None
    lvl = pd.concat(parts).groupby(level=0).sum().sort_index()
    return lvl[lvl.index <= min(frontiers)]


_NAT_LEVELS = {}
FRONTIER = None     # the trade frame's last month (set by estimator.build_universe)


def _national_origin_levels(overrides=None):
    """{(reporter id, code): monthly non-AI-origin level} from the national books, used to
    extend the Comtrade levels. Cached per frontier. Each book is cut at the trade frame's
    frontier ``FRONTIER`` (when set) so the extension ends where the trade data end; the per-cell
    exclusions (``overrides``, default PLACEBO_ORIGIN_OVERRIDES) match the Comtrade part of the
    spliced series."""
    from . import books
    overrides = PLACEBO_ORIGIN_OVERRIDES if overrides is None else overrides
    frontier = FRONTIER
    key = (frontier, frozenset(overrides.items()))
    if key in _NAT_LEVELS:
        return _NAT_LEVELS[key]
    out = {}
    for rep, leg in NATIONAL_LEG.items():
        path = books.find_national_csv(leg)
        if path is None:
            continue
        d = books._resolve(books._read_compact(path))
        if frontier is None:
            d = books.drop_provisional(d, books.DROP_TRAILING_MONTHS)
        else:
            d = d[d["period"] <= frontier]
        d = d[(d["flow"] == "M") & d["code_hs6"].isin(ORIGIN_IDX)].copy()
        if d.empty:
            continue
        d["pnorm"] = d["partner"].map(books.NORM).map(_norm)
        for code in ORIGIN_IDX:
            excl = overrides.get((rep, code), AI_ORIGIN_NORMS)
            lvl = (d[(d["code_hs6"] == code) & ~d["pnorm"].isin(excl)]
                   .groupby("period")["value_usd"].sum())
            if not lvl.empty:
                out[(rep, code)] = lvl
    _NAT_LEVELS[key] = out
    return out


def _splice(base, ext, rep, code, verbose=True):
    """Extend the Comtrade level ``base`` past its frontier with the national level ``ext``,
    linked by the median base/ext ratio over the last SEAM_MONTHS overlapping months. Both
    measure the same imports (national offices feed Comtrade), so the ratio is close to one; it
    absorbs small valuation and coverage differences. ``base`` is returned unchanged when the
    overlap is too thin or the national data end no later than Comtrade's."""
    frontier = base.index.max()
    tail = ext[ext.index > frontier]
    if tail.empty:
        return base
    overlap = sorted(set(base.index) & set(ext.index))[-SEAM_MONTHS:]
    ov = [p for p in overlap if ext.get(p, 0) > 0]
    if len(ov) < 3:
        if verbose:
            print(f"[baseline-index] national extension SKIPPED for {rep} {code}: "
                  f"only {len(ov)} usable overlap months")
        return base
    ratio = float(np.median([base[p] / ext[p] for p in ov]))
    if verbose:
        print(f"[baseline-index] national extension: {rep} {code} +{len(tail)}mo beyond {frontier} "
              f"(seam ratio {ratio:.2f})")
    return pd.concat([base, tail * ratio]).sort_index()


REFERENCES = ("adopted", "client_it_boards", "flat_systems")


class BaselineIndex:
    """Measured non-AI index per destination x code (2022 mean = 1).

    ``factor(dest, code)`` -> monthly factor Series, or None. ``lookup`` vectorizes it with a
    fallback (the world composite once ``set_composite`` has run, else the caller's parametric
    factors).

    ``reference`` selects the construction: ``adopted`` (systems follow the client-IT index,
    boards the non-AI-origin index), or one of two structural alternatives used by the
    uncertainty run -- ``client_it_boards`` (boards follow the destination's client-IT index
    too) and ``flat_systems`` (the systems codes held at their reference level).
    ``origin_overrides`` replaces PLACEBO_ORIGIN_OVERRIDES (``{}`` drops every exception).
    """

    def __init__(self, idx_dir=IDX_DIR, verbose=True, reference="adopted", origin_overrides=None):
        if reference not in REFERENCES:
            raise ValueError(f"unknown baseline reference {reference!r}")
        self.reference = reference
        overrides = PLACEBO_ORIGIN_OVERRIDES if origin_overrides is None else origin_overrides
        self.raw = _load_files(idx_dir)
        self.levels = {}    # (dest, code) -> raw monthly $ level
        self.idx = {}       # (dest, code) -> factor series
        self.fallbacks = set()
        self.composite = None            # {code: factor series}, the fallback for the rest
        self.composite_fallbacks = set()
        nat = _national_origin_levels(overrides)
        for rep, dest in REPORTER_DEST.items():
            d = self.raw[self.raw["rep_id"] == rep]
            if d.empty:
                continue
            sys_lvl = (d[d["code_hs6"].isin(PLACEBO_CODES)]
                       .groupby("period")["value_usd"].sum())
            wl = _wide_level(rep)
            if wl is None or wl.empty:
                print(f"[baseline-index] client-IT composite MISSING for {rep} -> "
                      f"laptops+desktops only")
            else:
                if verbose:
                    print(f"[baseline-index] client-IT composite: {rep} "
                          f"${float(wl[wl.index.str[:4] == '2022'].mean())/1e6:,.0f}M/mo "
                          f"2022, through {wl.index.max()}")
                sys_lvl = wl
            for code in SYSTEMS_IDX:
                self._add(dest, code, sys_lvl)
            for code in ORIGIN_IDX:
                excl = overrides.get((rep, code), AI_ORIGIN_NORMS)
                lvl = (d[(d["code_hs6"] == code) & ~d["pnorm"].isin(excl)]
                       .groupby("period")["value_usd"].sum())
                if (rep, code) in nat:
                    lvl = _splice(lvl, nat[(rep, code)], rep, code, verbose=verbose)
                self._add(dest, code, lvl)
        self._splice_stale(verbose=verbose)
        if reference == "client_it_boards":
            for dest, code in list(self.idx):
                if code == "847330" and (dest, "847150") in self.idx:
                    self.idx[(dest, code)] = self.idx[(dest, "847150")].copy()
        elif reference == "flat_systems":
            for key, f in list(self.idx.items()):
                if key[1] in SYSTEMS_IDX:
                    self.idx[key] = pd.Series(1., index=f.index)
        if verbose:
            built = sorted({d for d, _ in self.idx})
            print(f"[baseline-index] {len(self.idx)} (dest, code) indices for {built}")

    def _add(self, dest, code, level):
        from . import history
        if history.ENABLED and dest == "United Kingdom" and code in history.CODES:
            # The UK's 2021 months come from a reconstruction consistent with its EU dispatch
            # records (see prepare.uk_index); origin- and dispatch-based ratios are never mixed.
            patch = pd.read_csv(history.INPUT / "uk_index_2021.csv", index_col="period")
            level = level.astype(float).copy()
            level.loc[patch.index] = patch["parts" if code == "847330" else "client"]
            level = level.sort_index()
        f = _smooth_norm(level)
        if f is None:
            return
        self.levels[(dest, code)] = level
        self.idx[(dest, code)] = f

    def _splice_stale(self, verbose=True):
        """Continue cells frozen more than STALE_MONTHS before the frontier on peer growth.

        Per code, the peers are the indices still live at the frontier; their monthly median is
        the sector path. A stale cell continues as factor(t_freeze) x median(t) / median(t_freeze),
        re-clipped with the cell's own ceiling. Codes with fewer than MIN_LIVE_PEERS live peers are
        left forward-filled."""
        def _mdiff(a, b):
            return (int(b[:4]) - int(a[:4])) * 12 + (int(b[5:7]) - int(a[5:7]))

        frontier = max(f.index.max() for f in self.idx.values())
        by_code = {}
        for (dest, code), f in self.idx.items():
            by_code.setdefault(code, []).append((dest, f))
        for code, items in by_code.items():
            live = [f for _, f in items if _mdiff(f.index.max(), frontier) <= STALE_MONTHS]
            if len(live) < MIN_LIVE_PEERS:
                continue
            allp = sorted(set().union(*[set(f.index) for f in live]))
            med = pd.DataFrame({i: f.reindex(allp).ffill()
                                for i, f in enumerate(live)}).median(axis=1)
            for dest, f in items:
                last = f.index.max()
                if _mdiff(last, frontier) <= STALE_MONTHS or last not in med.index:
                    continue
                if not (med.loc[last] > 0):
                    continue
                tail = med[med.index > last] / med.loc[last]
                if tail.empty:
                    continue
                lvl = self.levels.get((dest, code))
                ref = (lvl[lvl.index.str[:4] == "2022"].mean()
                       if lvl is not None and len(lvl) else 0.0)
                clip = IDX_CLIP_FAT if ref >= FAT_DENOM_USD else IDX_CLIP
                ext = (float(f.loc[last]) * tail).clip(*clip)
                self.idx[(dest, code)] = pd.concat([f, ext]).sort_index()
                if verbose:
                    rel = float((ext / float(f.loc[last])).mean())
                    print(f"[baseline-index] stale splice: {dest} {code} frozen at {last} "
                          f"({_mdiff(last, frontier)}mo) -> peer-median growth "
                          f"({len(live)} live peers): factor {float(f.loc[last]):.2f} becomes a "
                          f"path averaging {rel:.2f}x it (range "
                          f"{float(ext.min())/float(f.loc[last]):.2f}-"
                          f"{float(ext.max())/float(f.loc[last]):.2f}x) through {ext.index[-1]}")

    def factor(self, dest, code):
        """The (dest, code) factor series, or None."""
        return self.idx.get((dest, code))

    def lookup(self, dests, codes, periods, fallback):
        """Factor per row of equal-length array-likes. Beyond an index's last month its last
        value is carried forward. A (dest, code) without its own index uses the composite when
        set, else keeps ``fallback`` (recorded in ``self.fallbacks``)."""
        key = pd.DataFrame({"dest": np.asarray(dests), "code": np.asarray(codes),
                            "period": np.asarray(periods)})
        out = np.asarray(fallback, dtype=float).copy()
        for (dest, code), g in key.groupby(["dest", "code"], sort=False):
            f = self.idx.get((dest, code))
            if f is None:
                if self.composite is not None and code in self.composite:
                    f = self.composite[code]
                    self.composite_fallbacks.add((dest, code))
                else:
                    self.fallbacks.add((dest, code))
                    continue
            ext = f.reindex(sorted(set(f.index) | set(g["period"]))).ffill()
            vals = ext.reindex(g["period"]).to_numpy()
            ok = ~np.isnan(vals)
            out[g.index[ok]] = vals[ok]
        return out

    def set_composite(self, weights):
        """Install the world composite (``aggregate(weights)``) as the fallback for every
        (dest, code) without its own index."""
        self.composite = self.aggregate(weights)
        fr = {c: f.index.max() for c, f in self.composite.items()}
        print(f"[baseline-index] composite fallback set for {sorted(fr)} (frontiers {fr}); "
              f"thin destinations with their own index: "
              f"{sorted({d for d, _ in self.idx} & set(THIN_REPORTER_DEST.values()))}")

    def aggregate(self, weights):
        """{code: factor series}: the destination indices averaged with ``weights``
        {(dest, code): 2022 monthly trade}."""
        by_code = {}
        for (dest, code), f in self.idx.items():
            w = float(weights.get((dest, code), 0.0))
            if w <= 0:
                continue
            by_code.setdefault(code, []).append((w, f))
        out = {}
        for code, items in by_code.items():
            allp = sorted(set().union(*[set(f.index) for _, f in items]))
            num = sum(w * f.reindex(allp).ffill().bfill() for w, f in items)
            out[code] = num / sum(w for w, _ in items)
        return out
