"""Parameter and sales draws.

Each draw is a ``dict`` (a "spec") of parameter values, built from one row of a scrambled Sobol
sequence. Two replicates of 2,048 rows each are generated with fixed seeds; a run of ``n``
draws per case uses the first ``n / 2`` rows of each, so a larger run extends a smaller one.

**Model parameters** (symmetric triangular shocks T on [-1, 1], mode 0, held for the whole
history within a draw):

* non-AI baseline: the reference of book c is multiplied by 1 + 0.10 (0.7 T_shared + 0.3 T_c)
  from January 2023 (``baseline``, ``baseline_by_leg``);
* k of own-made server exports: 0.10 (0.7 K_shared + 0.3 K_c) is added to k read from a book's
  own fields, clipped to [0, 1] (``k_shift``, ``k_by_leg``);
* classifier thresholds: the unit-value ladder's reference price and the $/kg ramp are each
  scaled by an independent 1 + 0.25 T (``price_anchor``, ``density_anchor``);
* category shifts: 0.10 T added to the physically classified 847150 and 847330 content, clipped
  to [0.05, 0.93] (``category``).

**Sales** (``apply_sales``): every model's units and H100e follow a two-piece lognormal through
Epoch's 5th, 50th and 95th percentiles, with one latent normal per model, z_m = sqrt(0.5) Z_d +
sqrt(0.5) Z_m (Z_d shared by the designer's models, so a 0.5 correlation within a designer). The
same rank applies in every quarter. Prices move with quantities: for Nvidia a shared hardware-
revenue factor r (lognormal, from the 0.96-0.99 hardware share of Epoch's notebook as an 80%
interval) fixes spending, so price = r / quantity factor; for AMD, Google and Cambricon half of
the log-quantity movement goes to price and half to spending; for every other designer the
price is an independent lognormal with 90% interval 0.8-1.25.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import qmc, triang

from .. import epoch, netting

Z90 = float(ndtri(.95))
SEED = 20260919             # replicate r uses SEED + r
PREFIX_POWER = 12           # 2 replicates x 2**(PREFIX_POWER - 1) rows
REPLICATE_SIZE = 2 ** (PREFIX_POWER - 1)
WITHIN_DESIGNER_CORRELATION = .5
BASELINE_WIDTH = .10
K_WIDTH = .10
ANCHOR_WIDTH = .25
CATEGORY_WIDTH = .10
CATEGORY_CODES = ("847150", "847330")
SHARED_WEIGHT, OWN_WEIGHT = .7, .3     # amplitudes of the shared and the book's own shocks
NVIDIA_HARDWARE_SHARE = (.96, .99)     # 80% interval of the hardware fraction of revenue
HALF_COUPLED_DESIGNERS = ("AMD", "Google", "Cambricon")
OTHER_PRICE_95 = 1.25       # 90% interval of the independent price factor: 1/1.25 .. 1.25


def quantile_factor(mid, lo, hi, z):
    """Multipliers on ``mid`` at standard-normal ``z`` from a two-piece lognormal through the
    5th/50th/95th percentiles ``lo``/``mid``/``hi``. Missing or nonpositive bounds keep the
    median on that side."""
    m = np.asarray(mid, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    lo = np.where((lo > 0) & (m > 0), lo, m)
    hi = np.where((hi > 0) & (m > 0), hi, m)
    if not (np.all(lo <= m + 1e-6) and np.all(hi >= m - 1e-6)):
        raise ValueError("Epoch's sales percentiles are not ordered")
    left = np.log(np.divide(m, lo, out=np.ones_like(m), where=lo > 0)) / Z90
    right = np.log(np.divide(hi, m, out=np.ones_like(m), where=m > 0)) / Z90
    return np.exp(np.asarray(z) * np.where(np.asarray(z) < 0, left, right))


def draw_spec(u, models, designers, model_designer, legs):
    """One draw from one Sobol row ``u`` (see the module docstring for the layout)."""
    normal = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
    i = 0
    zd = dict(zip(designers, normal[i:i + len(designers)]))
    i += len(designers)
    zm = dict(zip(models, normal[i:i + len(models)]))
    i += len(models)
    zp = dict(zip(models, normal[i:i + len(models)]))
    i += len(models)
    zr = dict(zip(designers, normal[i:i + len(designers)]))
    i += len(designers)
    rho = WITHIN_DESIGNER_CORRELATION
    z = {m: np.sqrt(rho) * zd[model_designer[m]] + np.sqrt(1 - rho) * zm[m] for m in models}
    v = triang.ppf(u[i:], c=.5, loc=-1., scale=2.)
    j = 0
    b_shared = v[j]
    j += 1
    b_own = dict(zip(legs, v[j:j + len(legs)]))
    j += len(legs)
    k_shared = v[j]
    j += 1
    k_own = dict(zip(legs, v[j:j + len(legs)]))
    j += len(legs)
    price_anchor, density_anchor = v[j:j + 2]
    j += 2
    category = dict(zip(CATEGORY_CODES, CATEGORY_WIDTH * v[j:j + 2]))
    own = OWN_WEIGHT
    return dict(
        z=z, zp=zp, zr=zr,
        baseline=1 + BASELINE_WIDTH * SHARED_WEIGHT * b_shared,
        baseline_by_leg={leg: 1 + BASELINE_WIDTH * (SHARED_WEIGHT * b_shared + own * b_own[leg])
                         for leg in legs},
        k_shift=K_WIDTH * SHARED_WEIGHT * k_shared,
        k_by_leg={leg: K_WIDTH * (SHARED_WEIGHT * k_shared + own * k_own[leg]) for leg in legs},
        price_anchor=1 + ANCHOR_WIDTH * price_anchor,
        density_anchor=1 + ANCHOR_WIDTH * density_anchor,
        category=category)


def all_draws():
    """Every draw of both replicates, numbered 0..4095 (replicate 1 starts at 2048)."""
    models_q = epoch.models_by_arrival_quarter()
    models = sorted(models_q.model.unique())
    designers = sorted(models_q.designer.unique())
    model_designer = models_q.drop_duplicates("model").set_index("model").designer.to_dict()
    legs = netting._un_legs()
    dim = 2 * len(designers) + 2 * len(models) + 2 + 2 * len(legs) + 4
    specs = []
    for rep in range(2):
        for row in qmc.Sobol(dim, scramble=True, seed=SEED + rep).random_base2(PREFIX_POWER - 1):
            spec = draw_spec(row, models, designers, model_designer, legs)
            spec.update(replicate=rep, draw=len(specs))
            specs.append(spec)
    return specs


def draws(samples):
    """The first ``samples / 2`` draws of each replicate (``samples`` a power of two, 32-4096)."""
    if not (32 <= samples <= 4096 and samples & (samples - 1) == 0):
        raise ValueError("samples must be a power of two between 32 and 4096")
    return [s for s in all_draws() if s["draw"] % REPLICATE_SIZE < samples // 2]


def neutral():
    """A draw with every parameter at its central value (the case's deterministic result)."""
    s = all_draws()[0]
    for key in ("z", "zp", "zr", "k_by_leg", "category"):
        s[key] = {k: 0. for k in s[key]}
    s["baseline_by_leg"] = {k: 1. for k in s["baseline_by_leg"]}
    s.update(baseline=1., k_shift=0., price_anchor=1., density_anchor=1., draw=-1)
    return s


def apply_sales(frame, spec, kind):
    """Perturb Epoch's sales for one draw.

    ``kind='models'``: a per-model frame (``epoch.models_by_sales_quarter``); units, H100e, ASP
    and cost change. ``kind='timelines'``: Epoch's timelines (``epoch.load_timelines``); units,
    H100e and cost change. Returns the perturbed copy and the per-row factors.
    """
    e = frame.copy()
    if kind == "models":
        models, designers = e.model, e.designer
        mid, lo, hi = e.units, e.units_lo, e.units_hi
        hm, hl, hh = e.h100e, e.h100e_lo, e.h100e_hi
    elif kind == "timelines":
        models = e["Chip type (linked)"].astype(str).str.strip()
        designers = e["Chip manufacturer"]
        mid = e["Number of Units"]
        lo, hi = e["Number of Units (5th percentile)"], e["Number of Units (95th percentile)"]
        hm = e["Compute estimate in H100e (median)"]
        hl, hh = e["H100e (5th percentile)"], e["H100e (95th percentile)"]
    else:
        raise ValueError(kind)
    z = models.map(spec["z"]).to_numpy()
    if not np.isfinite(z).all():
        raise ValueError("a model has no sales draw")
    qfactor = quantile_factor(mid, lo, hi, z)
    hfactor = quantile_factor(hm, hl, hh, z)
    logq = np.log(qfactor)
    price = np.ones(len(e))
    for des in designers.unique():
        mask = designers.eq(des).to_numpy()
        if des == "Nvidia":
            lo_share, hi_share = NVIDIA_HARDWARE_SHARE
            sigma = np.log(hi_share / lo_share) / (2 * ndtri(.9))
            price[mask] = np.exp(sigma * spec["zr"][des]) / qfactor[mask]
        elif des in HALF_COUPLED_DESIGNERS:
            price[mask] = np.exp(-.5 * logq[mask])
        else:
            price[mask] = np.exp(np.log(OTHER_PRICE_95) / Z90 * models[mask].map(spec["zp"]).to_numpy())
    if kind == "models":
        e["units"] = mid * qfactor
        e["h100e"] = hm * hfactor
        e["asp"] *= price
        e["cost"] = e.units * e.asp
    else:
        e["Number of Units"] = mid * qfactor
        e["Compute estimate in H100e (median)"] = hm * hfactor
        e["Cost Estimate (USD)"] *= qfactor * price
    factors = pd.DataFrame({"model": models, "designer": designers, "q_factor": qfactor,
                            "price_factor": price, "spend_factor": qfactor * price})
    return e, factors
