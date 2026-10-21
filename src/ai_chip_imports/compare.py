"""Compare the outputs of two pipeline runs, cell for cell.

The pipeline is deterministic: the same inputs and code reproduce every output byte for byte.
This tool checks that property between a reference run and a candidate run, e.g. after a
refactor, or to confirm that a reproduction from independently obtained data matches a
published run.

Each output is ``<name>_<YYYYMMDD>.csv``; the newest stamp in each directory is compared
(the reference may also hold ``<name>.csv.gz``). Rows are compared as a multiset (row order
can vary between runs) and cells as verbatim CSV text (no float round trip), so any change
in any value is reported.

    python -m ai_chip_imports.compare REFERENCE_DIR [CANDIDATE_DIR] [--files NAME ...]
                                    [--shared-columns] [--subset]

``--shared-columns`` compares only the columns both runs have (reporting the rest);
``--subset`` accepts a candidate whose rows are a subset of the reference's.
Exit status 0 means every compared file matched.
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

MAX_EXAMPLES = 5
_DATED = re.compile(r"^(?P<name>.+)_(?P<stamp>\d{8})\.csv$")


def newest_outputs(directory: Path) -> dict[str, Path]:
    """``{name: path}`` for the newest-stamped ``<name>_<YYYYMMDD>.csv`` (or ``<name>.csv.gz``)."""
    found: dict[str, tuple[str, Path]] = {}
    for path in Path(directory).iterdir():
        m = _DATED.match(path.name)
        if m:
            name, stamp = m.group("name"), m.group("stamp")
        elif path.name.endswith(".csv.gz"):
            name, stamp = path.name[: -len(".csv.gz")], ""
        else:
            continue
        if name not in found or stamp > found[name][0]:
            found[name] = (stamp, path)
    return {name: path for name, (_, path) in found.items()}


def read_canonical(path: Path) -> pd.DataFrame:
    """A CSV as verbatim text, columns sorted by name and rows sorted by every column."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        df = pd.read_csv(fh, dtype=str, keep_default_na=False)
    return _sorted(df[sorted(df.columns)])


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df) or not len(df.columns):
        return df.reset_index(drop=True)
    return df.sort_values(by=list(df.columns), kind="mergesort").reset_index(drop=True)


def _numeric_gap(a: pd.Series, b: pd.Series) -> str:
    x, y = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    return f" (max |diff| {np.abs(x[ok] - y[ok]).max():.6g})" if ok.any() else ""


def compare_frames(ref: pd.DataFrame, new: pd.DataFrame, shared_columns: bool = False,
                   subset: bool = False) -> tuple[bool, list[str]]:
    """Compare two canonical frames. Returns ``(matched, report lines)``."""
    lines: list[str] = []
    added = sorted(set(new.columns) - set(ref.columns))
    removed = sorted(set(ref.columns) - set(new.columns))
    if added:
        lines.append(f"  columns only in candidate: {added}")
    if removed:
        lines.append(f"  columns only in reference: {removed}")
    ok = shared_columns or not (added or removed)
    cols = [c for c in ref.columns if c in set(new.columns)]
    ref, new = _sorted(ref[cols]), _sorted(new[cols])

    if subset:
        merged = new.merge(ref.drop_duplicates(), on=cols, how="left", indicator=True)
        missing = merged[merged["_merge"] == "left_only"]
        if len(missing) or len(new) > len(ref):
            lines.append(f"  {len(missing)} candidate rows not in the reference")
            return False, lines
        lines.append(f"  {len(new)} of {len(ref)} reference rows reproduced exactly")
        return ok, lines

    if len(ref) != len(new):
        lines.append(f"  row count: reference {len(ref)} vs candidate {len(new)}")
        return False, lines
    for c in cols:
        neq = ref[c].to_numpy() != new[c].to_numpy()
        if neq.any():
            ok = False
            idx = np.flatnonzero(neq)[:MAX_EXAMPLES]
            ex = "; ".join(f"{ref[c].iat[i]!r} -> {new[c].iat[i]!r}" for i in idx)
            lines.append(f"  column {c!r}: {int(neq.sum())} cells differ"
                         f"{_numeric_gap(ref[c], new[c])}, e.g. {ex}")
    if ok:
        lines.append(f"  identical: {len(ref)} rows x {len(cols)} columns")
    return ok, lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ai_chip_imports.compare",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path, nargs="?", default=paths.OUTPUT_DIR)
    ap.add_argument("--files", nargs="*", help="output names to compare (default: all in candidate)")
    ap.add_argument("--shared-columns", action="store_true",
                    help="compare only the columns both runs have")
    ap.add_argument("--subset", action="store_true",
                    help="accept a candidate whose rows are a subset of the reference's")
    args = ap.parse_args(argv)

    ref_files, new_files = newest_outputs(args.reference), newest_outputs(args.candidate)
    names = args.files or sorted(new_files)
    all_ok = True
    for name in names:
        if name not in ref_files or name not in new_files:
            side = "reference" if name not in ref_files else "candidate"
            print(f"MISSING {name} (not in {side})")
            all_ok = False
            continue
        ok, lines = compare_frames(read_canonical(ref_files[name]), read_canonical(new_files[name]),
                                   shared_columns=args.shared_columns, subset=args.subset)
        all_ok &= ok
        print(("MATCH  " if ok else "DIFFER ") + name)
        print("\n".join(lines))
    print("\nall outputs match" if all_ok else "\ndifferences found")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
