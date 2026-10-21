"""Intervals from a completed run.

Per case and destination: the 5th, 50th and 95th percentiles of cumulative H100e over the case's
accepted draws (a draw is rejected when the allocator reports a warning or an infeasible fit;
``unfiltered_*`` keep every draw). Two stability diagnostics accompany each endpoint: its change
from the first half of the draws to all of them (``step_*_pct``), and the gap between the two
independently scrambled replicates (``scramble_*_pct``).

``country_intervals.csv`` reports, per destination, the central estimate, the conditional 90%
interval (the central case's p5-p95) and the structural envelope (the lowest case p5 to the
highest case p95). The envelope carries no probability: it spans the tested alternatives only.
"""
import json

import numpy as np
import pandas as pd

from .cases import CASES, CENTRAL
from .sampling import REPLICATE_SIZE

DISPLAY = ["United States", "China", "European Union", "Malaysia", "Japan", "India", "Australia",
           "United Kingdom", "South Korea", "Singapore"]


def case_intervals(run):
    """(per-case interval table, acceptance table)."""
    samples = run.plan["samples"]
    rows, accept = [], []
    for name in CASES:
        files = sorted((run.out / "cases" / name).glob("draw_[0-9][0-9][0-9][0-9].json"))
        accepted, every, replicate = {}, {}, {}
        for f in files:
            meta = json.loads(f.read_text())
            v = pd.read_csv(f.with_suffix(".csv")).set_index("dest").h100e
            v.loc["World"] = v.sum()
            every[meta["draw"]] = v
            if not meta["warnings"]:
                accepted[meta["draw"]] = v
                replicate[meta["draw"]] = meta["replicate"]
        accept.append(dict(case=name, proposals=len(files), accepted=len(accepted)))
        if not accepted:
            continue
        mat = pd.DataFrame(accepted).fillna(0)
        q = mat.quantile([.05, .5, .95], axis=1).T
        unfiltered = pd.DataFrame(every).fillna(0).quantile([.05, .95], axis=1).T
        half = mat[[i for i in mat if i % REPLICATE_SIZE < samples // 4]].quantile([.05, .95], axis=1).T
        a, b = [mat[[i for i in mat if replicate[i] == r]].quantile([.05, .95], axis=1).T for r in (0, 1)]
        for dest, v in q.iterrows():
            rows.append(dict(
                case=name, dest=dest, p05=v[.05], median=v[.5], p95=v[.95],
                unfiltered_p05=unfiltered.loc[dest, .05], unfiltered_p95=unfiltered.loc[dest, .95],
                step_p05_pct=100 * (v[.05] / half.loc[dest, .05] - 1) if half.loc[dest, .05] > 0 else np.nan,
                step_p95_pct=100 * (v[.95] / half.loc[dest, .95] - 1) if half.loc[dest, .95] > 0 else np.nan,
                scramble_p05_pct=100 * abs(a.loc[dest, .05] - b.loc[dest, .05]) / v[.05] if v[.05] > 0 else np.nan,
                scramble_p95_pct=100 * abs(a.loc[dest, .95] - b.loc[dest, .95]) / v[.95] if v[.95] > 0 else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(accept)


def write(run):
    """Write acceptance.csv, case_intervals.csv and, once every case is complete,
    country_intervals.csv and REPORT.md. Returns the country table (None while incomplete)."""
    plan = run.verify()
    table, accept = case_intervals(run)
    accept.to_csv(run.out / "acceptance.csv", index=False)
    if table.empty:
        return None
    table.to_csv(run.out / "case_intervals.csv", index=False)
    if not (accept.proposals == plan["samples"]).all():
        (run.out / "report_status.json").write_text(json.dumps({"complete": False}))
        return None      # an unfinished union of cases is not the envelope
    if (accept.accepted == 0).any():
        raise RuntimeError("a case has no accepted draws")
    grid = pd.MultiIndex.from_product([list(CASES), table.dest.unique()], names=["case", "dest"])
    table = table.set_index(["case", "dest"]).reindex(grid)
    table[["p05", "median", "p95"]] = table[["p05", "median", "p95"]].fillna(0)
    envelope = table.groupby("dest").agg(envelope_low=("p05", "min"), envelope_high=("p95", "max"))
    result = table.loc[CENTRAL, ["p05", "median", "p95"]].join(envelope)
    central = pd.Series(json.loads((run.out / "controls.json").read_text())[CENTRAL]["h100e"])
    central.loc["World"] = central.sum()
    result = result.join(central.rename("central_h100e")).fillna(0)
    result.index.name = "dest"
    result.to_csv(run.out / "country_intervals.csv")
    primary = ["World"] + central.drop("World").nlargest(20).index.tolist()
    check = table.reset_index()
    check = check[check.dest.isin(primary)]
    step = float(check[["step_p05_pct", "step_p95_pct"]].abs().max().max())
    gap = float(check[["scramble_p05_pct", "scramble_p95_pct"]].max().max())
    (run.out / "report_status.json").write_text(json.dumps(
        dict(complete=True, samples=plan["samples"], max_endpoint_step_pct=step, max_scramble_gap_pct=gap), indent=1))
    lines = ["# AI chip imports: uncertainty", "",
             f"{len(CASES)} structural cases, {plan['samples']} draws each, two independent scrambles.", "",
             "The conditional interval is the central case's 5th-95th percentile. The structural envelope "
             "runs from the lowest case 5th percentile to the highest case 95th percentile; no "
             "probabilities are assigned to the cases. Million H100e, cumulative arrivals.", "",
             "| Destination | Central | Conditional p5-p95 | Structural envelope |", "|---|--:|--:|--:|"]
    for dest in ["World"] + DISPLAY:
        r = result.loc[dest]
        lines.append(f"| {dest} | {r.central_h100e / 1e6:.3f} | {r.p05 / 1e6:.3f}-{r.p95 / 1e6:.3f} | "
                     f"{r.envelope_low / 1e6:.3f}-{r.envelope_high / 1e6:.3f} |")
    lines += ["", f"Largest half-to-full endpoint movement: {step:.2f}%; largest gap between the "
              f"independent scrambles: {gap:.2f}% (World and the 20 largest destinations)."]
    (run.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result
