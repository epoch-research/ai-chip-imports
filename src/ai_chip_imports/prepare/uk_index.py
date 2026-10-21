"""Build the UK's 2021 baseline-index levels, ``uk_index_2021.csv``.

The UK's 2021 imports from the EU are recorded on a different basis from later years (see
``uk_reconstruction``), so its 2021 index levels cannot be read from its import records
directly. Both of its index groups are rebuilt on a dispatch-consistent basis:

* **client** (the client-IT composite of ``baseline``): HMRC's 2021 totals with the imports from
  the EU replaced by EU members' exports to the UK, scaled by the UK-import / EU-export ratio of
  July 2022-June 2023, then linked to the Comtrade-based index series over the same window.
* **parts** (847330, non-AI origins only): the non-EU part as reported, plus the reconstructed
  EU part times the share of eligible (non-AI-supply-chain) origins in the UK's imports
  dispatched from the EU. That share is measured from Comtrade's origin-by-dispatch records for
  July-December 2022 and assumed to hold in 2021 -- a constant-composition backcast, not an
  observation of 2021.

Inputs in ``data/external/baseline_2021/``:

* ``uk_index_native_raw.json`` -- HMRC OTS monthly totals by flow type and HS-6 for the index
  codes, 2021-01 to 2023-06: ``{"rows": [{"MonthId", "FlowTypeId", "Commodity": {"Hs6Code"},
  "V"}, ...]}`` (flow type 1 = imports from the EU, 3 = from outside the EU);
* ``eu_index_exports_raw.json`` -- a Eurostat COMEXT JSON-stat response: EU members' exports to
  the UK of the same codes, 2021-01 to 2023-06;
* ``uk_comtrade_origin_dispatch_2022.json`` -- the UK's 2022 UN Comtrade import records with the
  second-partner (country of dispatch) breakdown;
* ``fx_2021.csv`` -- 2021 monthly rates (GBP, EUR); later months come from ``data/fx``;

plus the UK's index pulls in ``baseline_index/``, ``baseline_index_alt/`` and
``placebo_candidates/``.

    python -m ai_chip_imports.prepare.uk_index [--out-dir DIR]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import baseline, history, paths
from ..sources.eu import decode
from .uk_reconstruction import require_months

CLIENT = [c for _, _, codes in baseline.WIDE_SOURCES for c in codes]


def _fx(input_dir):
    fx = pd.concat([pd.read_csv(paths.FX_CSV), pd.read_csv(Path(input_dir) / "fx_2021.csv")])
    return fx.drop_duplicates(["currency_local", "period"]).set_index(
        ["currency_local", "period"]).units_per_usd


def eligible_dispatch_share(input_dir=history.INPUT):
    """Share of the UK's 847330 imports dispatched from the EU, July-December 2022, whose
    origin is outside the AI supply chain."""
    origin = pd.DataFrame(json.loads(
        (Path(input_dir) / "uk_comtrade_origin_dispatch_2022.json").read_text())["data"])
    origin = origin[(origin.cmdCode == "847330") & (origin.flowCode == "M")
                    & (origin.motCode == 0) & (origin.customsCode == "C00")
                    & origin.period.between("202207", "202212")]
    joint = origin[origin.partnerCode.ne(0) & origin.partner2Code.ne(0)]
    total = origin[origin.partnerCode.ne(0) & origin.partner2Code.eq(0)]
    if set(total.period) != {f"2022{month:02}" for month in range(7, 13)}:
        raise ValueError("Origin/dispatch bridge requires all six months of 2022H2")
    np.testing.assert_allclose(joint.groupby("period").primaryValue.sum(),
                               total.groupby("period").primaryValue.sum(), rtol=1e-9, atol=.01)
    via_eu = joint[joint.partner2Desc.map(baseline._norm).isin(baseline.EU_MEMBER_NORMS)]
    eligible = ~via_eu.partnerDesc.map(baseline._norm).isin(baseline.AI_ORIGIN_NORMS)
    return via_eu.loc[eligible, "primaryValue"].sum() / via_eu.primaryValue.sum()


def build(input_dir=history.INPUT):
    """The 2021 monthly levels (columns ``client``, ``parts``) and an audit record per group."""
    input_dir = Path(input_dir)
    fx = _fx(input_dir)
    rows = json.loads((input_dir / "uk_index_native_raw.json").read_text())["rows"]
    native = pd.DataFrame([{"period": f"{r['MonthId'] // 100}-{r['MonthId'] % 100:02d}",
                            "code_hs6": r["Commodity"]["Hs6Code"], "eu": r["FlowTypeId"] == 1,
                            "value_local": r["V"]} for r in rows])
    native["value_usd"] = [v / fx["GBP", p] for v, p in zip(native.value_local, native.period)]
    exports = pd.DataFrame([{"period": p, "code_hs6": c, "value_usd": v / fx["EUR", p]}
                            for _, _, _, c, p, _, v in decode(json.loads(
                                (input_dir / "eu_index_exports_raw.json").read_text()))])
    origin_share = eligible_dispatch_share(input_dir)
    base = baseline._load_files(baseline.IDX_DIR)
    parts = base[(base.rep_id == "uk") & (base.code_hs6 == "847330")]
    client = []
    for directory, tag, codes in baseline.WIDE_SOURCES:
        path = sorted(glob.glob(str(Path(directory) / f"uk_comtrade_{tag}_*_pulled*.csv")))[-1]
        d = pd.read_csv(path, dtype={"code_hs6": str})
        client.append(d[d.code_hs6.isin(codes)])
    client = pd.concat(client)
    result, audit = [], []
    for name, codes, data in [("client", CLIENT, client), ("parts", ["847330"], parts)]:
        n = native[native.code_hs6.isin(codes)]
        em = exports[exports.code_hs6.isin(codes)].groupby("period").value_usd.sum()
        eu_imports = n[n.eu].groupby("period").value_usd.sum()
        for label, series in [("EU imports", eu_imports), ("EU exports", em)]:
            require_months(series, "2022-07", "2023-06", f"{name} {label} calibration")
        ratio = eu_imports.loc["2022-07":"2023-06"].sum() / em.loc["2022-07":"2023-06"].sum()
        target = em.loc["2021-01":"2021-12"] * ratio
        if name == "client":
            # link the all-origin native totals to the published-dollar Comtrade series
            ct = data.groupby("period").value_usd.sum()
            nt = n.groupby("period").value_usd.sum()
            require_months(ct, "2022-07", "2023-06", "Comtrade client calibration")
            require_months(nt, "2022-07", "2023-06", "native client calibration")
            seam = ct.loc["2022-07":"2023-06"].sum() / nt.loc["2022-07":"2023-06"].sum()
            level = (nt.reindex(target.index) - eu_imports.reindex(target.index) + target) * seam
        else:
            eligible_non_eu = parts[~parts.pnorm.isin(baseline.AI_ORIGIN_NORMS | baseline.EU_MEMBER_NORMS)]
            original = eligible_non_eu.groupby("period").value_usd.sum()
            level = original.reindex(target.index, fill_value=0) + target * origin_share
            seam = 1.0
        assert len(level) == 12 and level.notna().all() and level.gt(0).all()
        result.append(level.rename(name))
        audit.append({"group": name, "native_EU_import_to_export_ratio": ratio,
                      "all_origin_seam": seam, "level_2021_usd": level.sum()})
    levels = pd.concat(result, axis=1).rename_axis("period").reset_index()
    return levels, audit, origin_share


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ai_chip_imports.prepare.uk_index",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=history.INPUT,
                    help="where to write (default: the input directory)")
    args = ap.parse_args(argv)
    levels, audit, origin_share = build()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    levels.to_csv(args.out_dir / "uk_index_2021.csv", index=False)
    (args.out_dir / "uk_index_manifest.json").write_text(json.dumps({
        "calibration": "2022-07 to 2023-06, dispatch-consistent HMRC/Eurostat",
        "parts_origin_bridge": "2022-07 to 2022-12 Comtrade origin-by-dispatch, monthly totals reconciled",
        "EU_dispatch_eligible_origin_share": origin_share,
        "assumption": "2021 EU dispatch origin composition equals 2022H2 composition",
        "groups": audit}, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
