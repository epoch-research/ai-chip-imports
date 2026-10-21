"""Running the cases: preparation, one process per case, resumable draws, provenance checks.

Layout of a run directory::

    plan.json             what the run is: draws per case, cases, code and input fingerprints
    universe.pkl          the prepared central universe (a trusted local cache of this run)
    controls.json         deterministic results the neutral draws must reproduce
    probe/<case>/         the probe's neutral, cached and uncached draws
    cases/<case>/         neutral.csv, draw_NNNN.{csv,json,log}, draw_NNNN_factors.csv
    logs/<case>.log       each case process's output

A draw is complete when its ``.json`` exists (written last, atomically); an interrupted run
resumes where it stopped. Code or inputs that change under a run are refused: start a new run
directory instead. More draws per case (``--samples``) extend the same sequences.
"""
import contextlib
import hashlib
import json
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__, paths
from . import draw, sampling
from .cases import CASES, CENTRAL

PROBE_CASES = ("central__central", "central__server_mix_5k__ordinary_first", "boards_client_it__parts_mixed",
               "named__low_content", "central__central__my_none", "central__central__my_full", "asp_merchant")
FLAGS = ("WARNING", "infeasible")     # allocator messages that reject a draw
CONTROL_TOLERANCE = {CENTRAL: (1e-9, .01)}    # (rtol, atol) on units/H100e; others (1e-7, 1)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint():
    """The package's code and assumptions, and the inputs (size, mtime) a run reads."""
    code = {str(p.relative_to(paths.PACKAGE_DIR)): _sha(p)
            for p in sorted(paths.PACKAGE_DIR.rglob("*")) if p.suffix in (".py", ".csv")}
    inputs = {}
    for root in (paths.CUSTOMS_DIR, paths.EXTERNAL_DIR, paths.FX_CSV.parent):
        for p in sorted(root.rglob("*.csv*")):
            if "raw" not in p.relative_to(root).parts:
                st = p.stat()
                inputs[str(p.relative_to(paths.DATA_DIR))] = [st.st_size, st.st_mtime_ns]
    return {"version": __version__, "code": code, "inputs": inputs}


class Run:
    """A run directory (default ``<output>/uncertainty``)."""

    def __init__(self, out=None):
        self.out = Path(out) if out else paths.OUTPUT_DIR / "uncertainty"

    @property
    def plan(self):
        return json.loads((self.out / "plan.json").read_text())

    def verify(self):
        """Refuse to continue when the code or the inputs changed since preparation."""
        plan = self.plan
        now = fingerprint()
        if now["code"] != plan["fingerprint"]["code"] or now["inputs"] != plan["fingerprint"]["inputs"]:
            raise RuntimeError("code or inputs changed since this run was prepared: use a new run directory")
        return plan

    # --- preparation -------------------------------------------------------------------------
    def prepare(self):
        """Build and cache the central universe and the deterministic controls."""
        if (self.out / "plan.json").exists():
            raise RuntimeError(f"{self.out} is already prepared")
        self.out.mkdir(parents=True, exist_ok=True)
        before = fingerprint()
        with (self.out / "prepare.log").open("w") as log, contextlib.redirect_stdout(log):
            u = draw.prepare_universe()
            with (self.out / "universe.pkl").open("wb") as f:
                pickle.dump(u, f, protocol=pickle.HIGHEST_PROTOCOL)
            ctl = draw.controls(u)
        (self.out / "controls.json").write_text(json.dumps(
            {k: {c: v[c].to_dict() for c in ("units", "h100e")} for k, v in ctl.items()}, indent=1))
        if fingerprint() != before:
            raise RuntimeError("inputs changed during preparation")
        plan = dict(samples=0, cases=list(CASES), fingerprint=before, universe_sha256=_sha(self.out / "universe.pkl"),
                    prepared=time.strftime("%Y-%m-%d"))
        (self.out / "plan.json").write_text(json.dumps(plan, indent=1))

    def load_universe(self):
        plan = self.verify()
        if _sha(self.out / "universe.pkl") != plan["universe_sha256"]:
            raise RuntimeError("universe.pkl does not match the plan")
        with (self.out / "universe.pkl").open("rb") as f:
            return pickle.load(f)

    def set_samples(self, samples):
        plan = self.verify()
        if samples < plan["samples"]:
            raise ValueError(f"the run already has {plan['samples']} draws per case")
        if samples > plan["samples"]:
            plan.setdefault("extensions", []).append([plan["samples"], samples])
            plan["samples"] = samples
            (self.out / "plan.json").write_text(json.dumps(plan, indent=1))

    # --- one case ------------------------------------------------------------------------------
    def worker(self, name, samples, probe=False, only=None):
        """Run case ``name`` in this process: its neutral draw (checked against the controls or
        for allocator warnings), then its draws (``only``: these draw numbers) or the probe."""
        central = self.load_universe()
        folder = self.out / ("probe" if probe else "cases") / name
        folder.mkdir(parents=True, exist_ok=True)
        with draw.frozen_thailand(central["cmodel"]._extra_books["Thailand"].copy()):
            case = draw.Case(name, central)
            stock, meta = self._execute(case, sampling.neutral(), folder / "neutral")
            self._check_neutral(name, stock, meta)
            if probe:
                spec = sampling.draws(samples)[0]
                cached, info = self._execute(case, spec, folder / "cached")
                case.cached = False
                raw, raw_info = self._execute(case, spec, folder / "uncached")
                pd.testing.assert_frame_equal(cached, raw, check_exact=False, rtol=1e-9, atol=1e-5)
                np.testing.assert_allclose(info["chip_B"], raw_info["chip_B"], rtol=1e-10, atol=1e-8)
                (folder / "probe_passed.json").write_text(json.dumps({"seconds": info["seconds"]}))
                return
            for spec in sampling.draws(samples):
                if only is not None and spec["draw"] not in only:
                    continue
                stem = folder / f"draw_{spec['draw']:04d}"
                if stem.with_suffix(".json").exists():
                    continue
                _, meta = self._execute(case, spec, stem)
                print(name, spec["draw"], round(meta["seconds"], 1), flush=True)
        self.verify()

    @staticmethod
    def _execute(case, spec, stem):
        start = time.monotonic()
        with stem.with_suffix(".log").open("w") as log, contextlib.redirect_stdout(log):
            stock, meta, factors = case.draw(spec)
        stock.to_csv(stem.with_suffix(".csv"))
        factors.to_csv(stem.with_name(stem.name + "_factors.csv"), index=False)
        meta.update(draw=spec["draw"], replicate=spec["replicate"], seconds=time.monotonic() - start,
                    warnings=[x.strip() for x in stem.with_suffix(".log").read_text().splitlines()
                              if any(f in x for f in FLAGS)])
        tmp = stem.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(stem.with_suffix(".json"))
        return stock, meta

    def _check_neutral(self, name, stock, meta):
        controls = json.loads((self.out / "controls.json").read_text())
        if name not in controls:
            if meta["warnings"]:
                raise AssertionError(f"{name}: neutral draw has allocator warnings: {meta['warnings']}")
            return
        rtol, atol = CONTROL_TOLERANCE.get(name, (1e-7, 1.))
        for col in ("units", "h100e"):
            ref = pd.Series(controls[name][col])
            idx = stock.index.union(ref.index)
            np.testing.assert_allclose(stock[col].reindex(idx).fillna(0), ref.reindex(idx).fillna(0),
                                       rtol=rtol, atol=atol, err_msg=f"{name} neutral {col}")

    # --- orchestration ---------------------------------------------------------------------------
    def _command(self, name, samples, probe=False):
        cmd = [sys.executable, "-u", "-m", "ai_chip_imports.uncertainty", "--out", str(self.out),
               "worker", name, "--samples", str(samples)] + (["--probe"] if probe else [])
        env = draw.single_threaded_env()
        env["AI_CHIP_IMPORTS_REFERENCE_YEAR"] = str(CASES[name].reference_year)
        return cmd, env

    def _launch(self, names, samples, workers, probe=False):
        (self.out / "logs").mkdir(exist_ok=True)

        def one(name):
            cmd, env = self._command(name, samples, probe)
            with (self.out / "logs" / f"{'probe_' if probe else ''}{name}.log").open("a") as log:
                code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"{name}: {'ok' if code == 0 else f'FAILED (exit {code}), see logs/'}", flush=True)
            return code
        with ThreadPoolExecutor(max_workers=workers) as pool:
            codes = list(pool.map(one, names))
        failed = [n for n, c in zip(names, codes) if c]
        if failed:
            raise RuntimeError(f"cases failed: {failed}")

    def probe(self, workers):
        """Check every probe case's neutral draw and its cached against uncached classification."""
        self.verify()
        self._launch(list(PROBE_CASES), 64, workers, probe=True)

    def run(self, samples, workers, names=None):
        """Run ``samples`` draws of every case (or ``names``), ``workers`` cases at a time."""
        for name in PROBE_CASES:
            if not (self.out / "probe" / name / "probe_passed.json").exists():
                raise RuntimeError("run the probe first")
        self.set_samples(samples)
        self._launch(list(names or CASES), samples, workers)
        return self.status()

    def status(self):
        plan = self.verify()
        counts = {n: len(list((self.out / "cases" / n).glob("draw_[0-9][0-9][0-9][0-9].json"))) for n in CASES}
        s = dict(completed=sum(counts.values()), target=len(CASES) * plan["samples"], by_case=counts)
        (self.out / "progress.json").write_text(json.dumps(s, indent=1))
        return s

