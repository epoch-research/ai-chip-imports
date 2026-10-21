"""Test setup: the pipeline's data and output roots point at empty temporary directories.

This runs before any test module imports ``ai_chip_imports``, so no test can read real inputs or
write outputs, and the tunable parameters take their default values.
"""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_ROOT = tempfile.mkdtemp(prefix="ai_chip_imports_tests_")
atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)
os.environ["AI_CHIP_IMPORTS_DATA"] = os.path.join(_ROOT, "data")
os.environ["AI_CHIP_IMPORTS_OUTPUT"] = os.path.join(_ROOT, "output")
for _name in ("AI_CHIP_IMPORTS_NETTING_QUARTERS", "AI_CHIP_IMPORTS_LAG_MONTHS",
              "AI_CHIP_IMPORTS_MALAYSIA_MIRROR_SHARE", "AI_CHIP_IMPORTS_REFERENCE_YEAR"):
    os.environ.pop(_name, None)
