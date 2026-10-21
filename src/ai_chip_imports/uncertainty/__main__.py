"""Command line: ``python -m ai_chip_imports.uncertainty [--out DIR] COMMAND``.

    prepare                       build the central universe and the deterministic controls
    probe [--workers N]           check the probe cases (neutral draws, cached vs uncached)
    run [--samples 64] [--workers N] [--cases A,B]
                                  draw every case (resumable; a larger --samples extends a run)
    status                        draws completed per case
    report                        case_intervals.csv, country_intervals.csv, REPORT.md

``worker CASE`` runs one case in the current process (the runner starts these).
"""
import argparse
import json

from . import report
from .cases import CASES
from .runner import Run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ai_chip_imports.uncertainty", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="run directory (default: <output>/uncertainty)")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("probe")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("run")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--cases", help="comma-separated case names (default: all)")
    sub.add_parser("status")
    sub.add_parser("report")
    p = sub.add_parser("worker")
    p.add_argument("case", choices=list(CASES))
    p.add_argument("--samples", type=int, required=True)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--draws", help="comma-separated draw numbers (default: all)")
    a = ap.parse_args(argv)
    run = Run(a.out)
    if a.command == "prepare":
        run.prepare()
    elif a.command == "probe":
        run.probe(a.workers)
    elif a.command == "run":
        status = run.run(a.samples, a.workers, a.cases.split(",") if a.cases else None)
        print(f"{status['completed']} of {status['target']} draws complete")
    elif a.command == "status":
        status = run.status()
        print(json.dumps({k: v for k, v in status.items() if k != "by_case"}))
    elif a.command == "report":
        result = report.write(run)
        print((run.out / "REPORT.md").read_text() if result is not None else "run incomplete: see case_intervals.csv")
    elif a.command == "worker":
        run.worker(a.case, a.samples, probe=a.probe,
                   only={int(x) for x in a.draws.split(",")} if a.draws else None)


if __name__ == "__main__":
    main()
