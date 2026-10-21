"""Command line: ``python -m ai_chip_imports [estimate|run]``.

``estimate`` runs the estimator and writes its outputs; ``run`` (the default) also runs the
allocator. Inputs and outputs are located by ``ai_chip_imports.paths``.
"""
import argparse

from . import allocator, estimator


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ai_chip_imports",
                                 description="Estimate AI accelerator arrivals by destination "
                                             "from customs data.")
    ap.add_argument("command", nargs="?", default="run", choices=("run", "estimate"),
                    help="run = estimator + allocator (default); estimate = estimator only")
    args = ap.parse_args(argv)
    U = estimator.build_universe()
    est = estimator.run(U)
    if args.command == "run":
        allocator.run(U, est)


if __name__ == "__main__":
    main()
