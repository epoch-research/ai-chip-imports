"""The structural cases: alternative readings of what the trade contains, run as whole models.

The parameter draws (``sampling``) vary values around one interpretation of the data; the cases
change the interpretation itself and rerun everything. No probabilities are attached to them.
There are 48 specifications, each crossed with three Malaysia server-gap deductions (0%, 50%,
100%), for 144 cases:

* **Reference x content (12).** Three non-AI references -- the adopted one (client IT for
  systems, the non-AI-origin component index for boards), client IT for boards too
  (``boards_client_it``), and flat systems references (``systems_flat``) -- crossed with four
  content treatments: the adopted classifier (``central``), two explicit server mixtures
  (``server_mix_5k``/``server_mix_15k``, see ``mixture_content``) and ``parts_mixed``.
* **Selection (27).** The three references crossed with a server ladder capped at the eligible
  system mix (``capped_ladder``) or either server mixture, and with three rules for which
  products remain after the baseline is removed (``selected_chip_value``): proportional,
  halfway, ordinary-first.
* **Named (6).** ``low_content`` (server readings capped at the mix, the US NESOI boards line read
  by density, merchant-only reference prices), ``ref2022`` (the 2022 reference with its AI
  carve), ``parts_low``, ``parts_high``, ``units_mixed`` (``transform``) and ``china_high`` (no
  diversion premium, Vietnam excluded from China's boards reference).
* **Composition (2).** ``joint_mixed`` (parts and units both mixed) and ``units_assembly``
  (other units capped at the HGX-server content).
* **Price reference (1).** ``asp_merchant``: the classifier's reference price from merchant
  models only.

The central case, ``central__central``, is the published model; its draws give the conditional
interval.
"""
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

SERVER_CODES = ("847149", "847150")
LOW, BOARD, SYSTEM = .05, .93, .73          # content benchmarks: no accelerator, board, HGX server
MIXTURE_GPUS = 8
MIRROR_SHARES = {"": .5, "__my_none": 0., "__my_full": 1.}
REFERENCES = {"central": "adopted", "boards_client_it": "client_it_boards", "systems_flat": "flat_systems"}
CONTENTS = ("central", "server_mix_5k", "server_mix_15k", "parts_mixed")
SELECTION_CONTENTS = ("capped_ladder", "server_mix_5k", "server_mix_15k")
SELECTION_RULES = {"proportional": 0., "half_preferential": .5, "ordinary_first": 1.}
NAMED = ("low_content", "ref2022", "parts_low", "parts_high", "units_mixed", "china_high")
COMPOSITIONS = ("parts_low", "parts_mixed", "parts_high", "units_mixed", "joint_mixed", "units_assembly")
CENTRAL = "central__central"


@dataclass(frozen=True)
class Case:
    """One structural case. ``baseline`` names the reference (``REFERENCES``); ``content`` the
    content treatment; ``preference`` the selection rule's weight on ordinary-first removal
    (None: no selection rule, no cap); ``named``/``composition`` a named or composition case;
    ``asp_merchant`` the merchant-only price reference; ``mirror_share`` the Malaysia
    deduction."""
    name: str
    baseline: str = "central"
    content: str = "central"
    preference: Optional[float] = None
    named: Optional[str] = None
    composition: Optional[str] = None
    asp_merchant: bool = False
    mirror_share: float = .5

    @property
    def capped(self):
        """Selection cases cap measured server content at the scenario's ceiling."""
        return self.preference is not None

    @property
    def low_content(self):
        return self.named == "low_content"

    @property
    def merchant_prices(self):
        return self.asp_merchant or self.low_content

    @property
    def composition_mode(self):
        """The ``transform`` mode applied to ambiguous 847330/847180 readings."""
        mode = self.composition or self.named or self.content
        return mode if mode in COMPOSITIONS else "control"

    @property
    def reference_year(self):
        return 2022 if self.named == "ref2022" else 2021

    @property
    def premium(self):
        """Whether China's diversion premium applies (None: the allocator's default)."""
        return False if self.named == "china_high" else None

    @property
    def index_options(self):
        options = {"reference": REFERENCES[self.baseline]}
        if self.named == "china_high":
            options["origin_overrides"] = {}
        return options

    @property
    def rebuild(self):
        """Whether the case needs its own universe (a different reference or content model at
        build time), rather than the central one."""
        return self.baseline != "central" or self.named in ("china_high", "ref2022", "low_content")


def _specifications():
    specs = [Case(f"{b}__{c}", baseline=b, content=c) for b in REFERENCES for c in CONTENTS]
    specs += [Case(f"{b}__{c}__{rule}", baseline=b, content=c, preference=weight)
              for b in REFERENCES for c in SELECTION_CONTENTS for rule, weight in SELECTION_RULES.items()]
    specs += [Case(f"named__{n}", named=n) for n in NAMED]
    specs += [Case(m, composition=m) for m in ("joint_mixed", "units_assembly")]
    specs += [Case("asp_merchant", asp_merchant=True)]
    return specs


CASES = {}
for _suffix, _share in MIRROR_SHARES.items():
    for _spec in _specifications():
        _case = replace(_spec, name=_spec.name + _suffix, mirror_share=_share)
        CASES[_case.name] = _case


def transform(values, code, mode):
    """Rewrite ambiguous readings for a composition case. Readings at or below the 0.05 floor
    (and NaNs) are kept; above it, 847330 goes to the floor (``parts_low``), halfway to it
    (``parts_mixed``, ``joint_mixed``) or to the board benchmark (``parts_high``), and 847180
    halfway to the floor (``units_mixed``, ``joint_mixed``) or capped at the HGX-server content
    (``units_assembly``). These are alternative interpretations, not measured memory shares."""
    a = np.asarray(values, dtype=float).copy()
    mask = a > .05000001
    if code == "847330":
        if mode == "parts_low":
            a[mask] = LOW
        elif mode in ("parts_mixed", "joint_mixed"):
            a[mask] = LOW + .5 * (a[mask] - LOW)
        elif mode == "parts_high":
            a[mask] = BOARD
    if code == "847180":
        if mode in ("units_mixed", "joint_mixed"):
            a[mask] = LOW + .5 * (a[mask] - LOW)
        elif mode == "units_assembly":
            a[mask] = np.minimum(a[mask], SYSTEM)
    return a


def mixture_content(uv, asp, ordinary_price, gpus=MIXTURE_GPUS, system_content=SYSTEM):
    """Chip-dollar share of a server cell read as a mix of ordinary servers and accelerator systems.

    An accelerator system holds ``gpus`` chips at reference price ``asp`` and is priced
    p_A = gpus x asp / ``system_content``; an ordinary server costs ``ordinary_price`` and holds
    no accelerator. A cell with average value ``uv`` between the two has accelerator-system item
    share a = (uv - p_0) / (p_A - p_0) and chip-dollar share a x p_A x system_content / uv. Cells
    outside [p_0, p_A] (or with p_A <= p_0) return NaN and keep the classifier's reading. These
    are assumed configurations, not ones the customs averages identify.
    """
    uv, asp = np.broadcast_arrays(np.asarray(uv, float), np.asarray(asp, float))
    p_ai = gpus * asp / system_content
    valid = (uv > 0) & (p_ai > ordinary_price) & (uv <= p_ai)
    share = np.clip((uv - ordinary_price) / (p_ai - ordinary_price), 0, 1)
    return np.where(valid, share * p_ai * system_content / np.maximum(uv, 1e-9), np.nan)


def mixture_price(content):
    """The ordinary-server price of a server-mixture content treatment."""
    return 5000. if content == "server_mix_5k" else 15000.


def selected_chip_value(value, excess, content, ceiling, preference):
    """Chip dollars left in the excess under a selection rule.

    With retained trade N, excess E after the baseline and ledger, gross chip share c and the
    accelerator product's chip share s: proportional selection keeps cE; ordinary-first selection
    keeps as much of the gross chip value cN as fits in E at share s, min(cN, sE); ``preference``
    interpolates between them (0 proportional, 1 ordinary-first).
    """
    n, e, c, s = np.broadcast_arrays(value, excess, content, ceiling)
    if not 0 <= preference <= 1:
        raise ValueError("preference must be in [0, 1]")
    if not (np.isfinite([n, e, c, s]).all() and (n >= 0).all() and (e >= 0).all()
            and (e <= n + .01).all() and (c >= 0).all() and (c <= s + 1e-10).all() and (s <= 1).all()):
        raise ValueError("invalid composition or retained-value inputs")
    return (1 - preference) * c * e + preference * np.minimum(c * n, s * e)
