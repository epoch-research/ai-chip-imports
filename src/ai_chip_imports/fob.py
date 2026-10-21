"""Every import book on the FOB basis.

Customs offices record imports at their value on arrival. Most books value imports CIF -- cost,
insurance and freight -- so the price includes shipping the goods there. The US, Mexico and
Canada record a FOB-type value: the price at the exporter's border, before international
freight. FOB is also the basis of every country's exports and of Epoch's sell-in dollars, so all
CIF import books are put on FOB before anything else runs.

1. Measure the margin where it is reported. The US Census publishes, for every import line, the
   customs value, the CIF value and the freight and insurance charges by air and by sea
   (``data/external/us_cif/``, see ``sources.us``). Charges / customs value is the margin, per HS-6
   code x origin x month. Overland corridors (Canada, Mexico) are left out of the fit.
2. Model the margin as a power law in value density: ``tau = A_q * density^(-b)``. Goods worth
   ~$100/kg pay ~2.5% of their value in freight and insurance, a GPU server at ~$400/kg ~1.3%,
   ~$4,000/kg ~0.3%. The exponent ``b`` is fitted once on the whole sample; the level ``A_q`` is
   refitted every quarter from that quarter's US cells alone (value-weighted least squares), so
   the curve follows the freight cycle.
3. Apply the curve to every import row of every CIF book. The row's own value density sets its
   margin where the book carries weight; otherwise the pooled density of the weight-bearing CIF
   books for the same code and quarter. Densities are held inside the range the US sample covers.
   The import value is divided by (1 + tau); exports are untouched.

This is the first component of the CEPII BACI reconciliation -- a margin fitted on the product's
transportability, removed from import values -- fitted on explicit charges rather than mirror
ratios, and log-log in density so it cannot extrapolate below zero for very expensive goods.

Known approximations: the US curve is fitted on gross shipping weight while other books report
net weight (margins understated by well under 0.1% of value); imports from neighbours that travel
by truck are over-deflated by at most ~1% of those flows.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import paths

SIDECAR_DIR = str(paths.EXTERNAL_DIR / "us_cif")

# Reporter ids whose import book is already FOB-type (US customs value; Mexico and Canada value
# imports FOB at the point of direct shipment). Every other book values imports CIF, including
# Australia's UN Comtrade submission, whose primary value is the CIF value.
FOB_IMPORT_REPORTERS = {"us", "mexico", "canada"}
LAND_BORDER_ORIGINS = {"CANADA", "MEXICO"}       # overland corridors, excluded from the fit
B_GRID = np.round(np.arange(0.0, 1.5001, 0.01), 2)
SUPPORT_Q = (0.01, 0.99)                         # value-weighted density quantiles bounding the curve
AI_LO_PERIOD = "2023-07"


def quarter(p):
    return f"{p[:4]}Q{(int(p[5:7]) - 1) // 3 + 1}"


def _wquantile(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(x[o][min(int(np.searchsorted(cw, q)), len(x) - 1)]) for q in qs]


def newest_sidecar():
    hits = sorted(glob.glob(os.path.join(SIDECAR_DIR, "us_cif_charges_M_*_pulled*.csv")))
    return hits[-1] if hits else None


class FobModel:
    """The US-measured margin curve and its application to CIF import books."""

    def __init__(self, path=None, verbose=True):
        self.path = path or newest_sidecar()
        if self.path is None:
            raise SystemExit(f"[fob-basis] no US charges file us_cif_charges_M_*.csv in {SIDECAR_DIR}")
        d = pd.read_csv(self.path, dtype={"cty_code": str, "code_hs6": str, "period": str})
        d = d[(d["gen_val"] > 0) & ~d["partner"].str.upper().isin(LAND_BORDER_ORIGINS)].copy()
        d["kg"] = d["air_kg"] + d["ves_kg"]
        c = d[d["kg"] > 0].copy()
        c["dens"] = c["gen_val"] / c["kg"]
        c["tau"] = c["gen_cha"] / c["gen_val"]
        c["q"] = c["period"].map(quarter)
        # a quarter gets its own level only once all three months are published; later months
        # ride the last complete quarter (the pipeline's own provisional-frontier rule)
        months = c.groupby("q")["period"].nunique()
        complete = set(months[months >= 3].index)
        self.partial_quarters = sorted(set(months.index) - complete)
        c = c[c["q"].isin(complete)].copy()
        self.cells = c
        groups = {q: g for q, g in c.groupby("q")}
        sse = [sum(self._sse(g, self._level(g, b), b) for g in groups.values()) for b in B_GRID]
        self.b = float(B_GRID[int(np.argmin(sse))])
        self.A = {q: self._level(g, self.b) for q, g in groups.items()}
        self.quarters = sorted(self.A)
        self.dens_lo, self.dens_hi = _wquantile(c["dens"].to_numpy(), c["gen_val"].to_numpy(), SUPPORT_Q)
        cd = c.groupby("code_hs6").agg(v=("gen_val", "sum"), w=("kg", "sum"))
        self.code_dens = (cd["v"] / cd["w"]).to_dict()          # last-resort fallback density
        self.median_dens = _wquantile(c["dens"].to_numpy(), c["gen_val"].to_numpy(), [0.5])[0]
        self.fb = None                                          # (code, quarter) -> pooled density
        self.fbc = None                                         # code -> pooled density
        self.removed = None                                     # per-leg diag from the last apply
        if verbose:
            print(f"[fob-basis] CIF import books -> FOB. US explicit charges "
                  f"{os.path.basename(self.path)}: {len(c):,} cells, "
                  f"${c['gen_val'].sum()/1e9:.0f}B customs value, margin "
                  f"{100*c['gen_cha'].sum()/c['gen_val'].sum():.2f}%; common slope b={self.b:.2f}, "
                  f"density support ${self.dens_lo:,.0f}-${self.dens_hi:,.0f}/kg; quarter levels "
                  f"{self.quarters[0]}..{self.quarters[-1]}"
                  + (f" (partial {'/'.join(self.partial_quarters)} rides {self.quarters[-1]})"
                     if self.partial_quarters else ""))
            print("  tau at $400/kg (%): " + ", ".join(
                f"{q} {100*self.tau_scalar(q, 400.0):.2f}" for q in self.quarters))

    @staticmethod
    def _level(g, b):
        x = g["dens"].to_numpy() ** (-b)
        w = g["gen_val"].to_numpy()
        return float((w * x * g["tau"].to_numpy()).sum() / (w * x * x).sum())

    @staticmethod
    def _sse(g, A, b):
        return float((g["gen_val"].to_numpy()
                      * (g["tau"].to_numpy() - A * g["dens"].to_numpy() ** (-b)) ** 2).sum())

    def level(self, q):
        if q in self.A:
            return self.A[q]
        return self.A[self.quarters[-1]] if q > self.quarters[-1] else self.A[self.quarters[0]]

    def tau_scalar(self, q, dens):
        return self.level(q) * float(np.clip(dens, self.dens_lo, self.dens_hi)) ** (-self.b)

    def tau(self, q, dens):
        """Vectorized margin for arrays of quarters and densities."""
        A = np.asarray(pd.Series(q).map(self.level).to_numpy(), dtype=float)
        d = np.clip(np.asarray(dens, dtype=float), self.dens_lo, self.dens_hi)
        return A * d ** (-self.b)

    # --- fallback densities from the CIF books that carry weight -------------------------------
    def fallback_from(self, res):
        m = ((res["flow"] == "M") & ~res["reporter"].isin(FOB_IMPORT_REPORTERS)
             & (res["weight_kg"] > 0) & (res["value_usd"] > 0))
        wb = res.loc[m, ["code_hs6", "period", "value_usd", "weight_kg"]]
        q = wb["period"].map(quarter)
        g = wb.groupby([wb["code_hs6"], q]).agg(v=("value_usd", "sum"), w=("weight_kg", "sum"))
        g.index.names = ["code_hs6", "q"]
        self.fb = (g["v"] / g["w"]).to_dict()
        gc = wb.groupby("code_hs6").agg(v=("value_usd", "sum"), w=("weight_kg", "sum"))
        self.fbc = (gc["v"] / gc["w"]).to_dict()

    def density(self, code, q, value, weight):
        """Own density where the row carries weight, else pooled (code, quarter), else code, else
        the US sample's code density, else its median. Arrays in, array out."""
        code = np.asarray(code, dtype=object)
        q = np.asarray(q, dtype=object)
        value = np.asarray(value, dtype=float)
        weight = np.asarray(weight, dtype=float)
        fb = self.fb or {}
        fbc = self.fbc or {}
        pooled = np.array([fb.get((c_, q_), fbc.get(c_, self.code_dens.get(c_, self.median_dens)))
                           for c_, q_ in zip(code, q)], dtype=float)
        own = np.divide(value, weight, out=np.full(len(value), np.nan), where=weight > 0)
        return np.where(np.isfinite(own) & (weight > 0), own, pooled)

    # --- application -----------------------------------------------------------------------------
    def apply(self, res, verbose=True):
        """Deflate every CIF import row of ``res`` (the ``books.load_all`` frame) to FOB."""
        if self.fb is None:
            self.fallback_from(res)
        m = ((res["flow"] == "M") & ~res["reporter"].isin(FOB_IMPORT_REPORTERS)
             & (res["value_usd"] > 0))
        sub = res.loc[m]
        q = sub["period"].map(quarter).to_numpy()
        dens = self.density(sub["code_hs6"].to_numpy(), q, sub["value_usd"].to_numpy(),
                            sub["weight_kg"].to_numpy())
        f = 1.0 / (1.0 + self.tau(q, dens))
        out = res.copy()
        out.loc[m, "value_usd"] = sub["value_usd"].to_numpy() * f
        if "value_qtycov" in out.columns:
            out.loc[m, "value_qtycov"] = sub["value_qtycov"].to_numpy() * f
        leg = sub["rep_c"] if "rep_c" in sub.columns else sub["reporter"]
        ai = (sub["period"] >= AI_LO_PERIOD).to_numpy()
        diag = pd.DataFrame({"leg": leg.fillna(sub["reporter"]).to_numpy(), "cif": sub["value_usd"].to_numpy(),
                             "removed": sub["value_usd"].to_numpy() * (1.0 - f), "ai": ai})
        rep = diag[diag["ai"]].groupby("leg")[["cif", "removed"]].sum().sort_values("cif", ascending=False)
        rep["pct"] = 100 * rep["removed"] / rep["cif"]
        self.removed = rep
        if verbose:
            own = float(sub.loc[sub["weight_kg"] > 0, "value_usd"].sum() / sub["value_usd"].sum())
            print(f"[fob-basis] applied to {int(m.sum()):,} CIF import rows; own-weight density on "
                  f"{100*own:.0f}% of their value, pooled code x quarter density on the rest. "
                  f"Removed since {AI_LO_PERIOD} (all codes in the compact frame), $B:")
            show = rep[rep["cif"] > 5e9].copy()
            show[["cif", "removed"]] = show[["cif", "removed"]] / 1e9
            print(show.round(3).to_string())
        return out

    def fobize_book(self, df, verbose=False):
        """Deflate the import rows of a book read outside ``books.load_all`` (Thailand's) the same
        way, using the pooled densities of the last ``apply`` when available."""
        if "flow" not in df.columns:
            return df
        m = (df["flow"] == "M") & (pd.to_numeric(df["value_usd"], errors="coerce") > 0)
        if not m.any():
            return df
        sub = df.loc[m]
        q = sub["period"].astype(str).map(quarter).to_numpy()
        w = (pd.to_numeric(sub["weight_kg"], errors="coerce").fillna(0.0).to_numpy()
             if "weight_kg" in sub.columns else np.zeros(len(sub)))
        dens = self.density(sub["code_hs6"].astype(str).to_numpy(), q,
                            pd.to_numeric(sub["value_usd"], errors="coerce").to_numpy(), w)
        f = 1.0 / (1.0 + self.tau(q, dens))
        out = df.copy()
        out.loc[m, "value_usd"] = pd.to_numeric(sub["value_usd"], errors="coerce").to_numpy() * f
        if verbose:
            print(f"[fob-basis] book fobized: {int(m.sum()):,} M rows, "
                  f"{100*(1-float(np.average(f, weights=sub['value_usd'].to_numpy()))):.2f}% removed")
        return out

    def manifest(self):
        return {"sidecar": os.path.basename(self.path), "b": round(self.b, 4),
                "complete_quarters": [self.quarters[0], self.quarters[-1]],
                "partial_quarters_riding_last": self.partial_quarters,
                "dens_support_usd_per_kg": [round(self.dens_lo, 1), round(self.dens_hi, 1)],
                "tau_400_pct_by_quarter": {q: round(100 * self.tau_scalar(q, 400.0), 3) for q in self.quarters},
                "removed_ai_window_usd_by_leg": ({k: round(float(v), 0) for k, v in self.removed["removed"].items()}
                                                 if self.removed is not None else None)}


_MODEL = None


def get_model(verbose=True):
    """The process-wide margin model, fitted on first use."""
    global _MODEL
    if _MODEL is None:
        _MODEL = FobModel(verbose=verbose)
    return _MODEL


def fobize(res, verbose=True):
    """Every CIF import row of the loaded frame, deflated to FOB."""
    return get_model(verbose).apply(res, verbose=verbose)


def fobize_book(df, verbose=False):
    """``FobModel.fobize_book`` on the shared model."""
    return get_model(verbose=False).fobize_book(df, verbose=verbose)

