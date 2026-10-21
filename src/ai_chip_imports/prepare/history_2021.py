"""Build the measured books' 2021 history, ``measured_history.csv.gz``.

Reads three prepared 2021 books from ``data/external/baseline_2021/``, each in the loaded-frame
layout (``reporter, flow, partner, rep_c, par_c, code_hs6, period, value_usd, qty, weight_kg,
value_qtycov``) and on the FOB basis:

* ``comtrade_prepared.csv.gz`` -- UN Comtrade monthly submissions of the measured books;
* ``eu_native_prepared.csv.gz`` -- Eurostat's native records for the 27 members;
* ``uk_native_prepared.csv.gz`` -- HMRC's native records for the UK.

The EU and UK books replace their members' Comtrade rows, and the UK's 2021 imports from the EU
are reconstructed from EU members' exports to the UK (``uk_reconstruction``), calibrated on the
current books over July 2022-June 2023. The result is restricted to the counted codes and
validated: 2021 only, finite non-negative values, every measured book present, and twelve months
of imports for every book x code.

    python -m ai_chip_imports.prepare.history_2021 [--out-dir DIR]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import books, estimator, history, netting
from . import uk_reconstruction as uk

SOURCES = ("comtrade_prepared.csv.gz", "eu_native_prepared.csv.gz", "uk_native_prepared.csv.gz")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(res=None, input_dir=history.INPUT):
    """The measured books' 2021 rows, the UK calibration ratios and the UK monthly targets."""
    com, eu, hmrc = [pd.read_csv(Path(input_dir) / name, dtype={"code_hs6": str}) for name in SOURCES]
    current = estimator.load_res() if res is None else res
    imports = current[current.rep_c.eq("United Kingdom") & current.flow.eq("M")
                      & current.par_c.isin(books.EU_MEMBERS)]
    exports = current[current.rep_c.isin(books.EU_MEMBERS) & current.flow.eq("X")
                      & current.par_c.eq("United Kingdom")]
    exports21 = (eu[eu.flow.eq("X") & eu.par_c.eq("United Kingdom")]
                 .groupby(["code_hs6", "period"]).value_usd.sum())
    corrected, ratios, targets = uk.reconstruct(hmrc, imports, exports, exports21, books.EU_MEMBERS)
    com = com[~com.rep_c.isin(set(books.EU_MEMBERS) | {"United Kingdom"})]
    result = pd.concat([com, eu, corrected], ignore_index=True)
    result = result[result.code_hs6.isin(history.CODES)]
    assert result.period.str.startswith("2021-").all()
    assert np.isfinite(result.value_usd).all() and result.value_usd.ge(0).all()
    assert set(netting._un_legs()) - {"European Union"} <= set(result.rep_c)
    coverage = result[result.flow.eq("M")].groupby(["rep_c", "code_hs6"]).period.nunique()
    assert coverage.min() == 12, coverage[coverage < 12]
    return result, ratios, targets


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ai_chip_imports.prepare.history_2021",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=history.INPUT,
                    help="where to write (default: the input directory)")
    args = ap.parse_args(argv)
    result, ratios, targets = build()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "measured_history.csv.gz", index=False)
    targets.to_csv(out / "uk_reconstruction_targets.csv", index=False)
    manifest = {
        "reference_year": 2021,
        "covered_books": netting._un_legs(),
        "uk_calibration": [uk.CALIBRATION_START, uk.CALIBRATION_END],
        "uk_calibration_ratios": ratios.to_dict(),
        "sources": {name: _digest(history.INPUT / name) for name in SOURCES},
        "history_sha256": _digest(out / "measured_history.csv.gz"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(result):,} historical rows for {len(netting._un_legs())} measured books")


if __name__ == "__main__":
    main()
