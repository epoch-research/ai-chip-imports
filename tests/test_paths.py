"""Filesystem layout: every input and output path derives from two overridable roots."""
import importlib
import re

import pytest

from ai_chip_imports import paths


@pytest.fixture
def reload_paths(monkeypatch):
    """Reload ``paths`` under a patched environment; the module is restored afterwards."""
    def load(**env):
        for name, value in env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, str(value))
        return importlib.reload(paths)
    yield load
    monkeypatch.undo()
    importlib.reload(paths)


def test_data_root_follows_its_environment_variable(reload_paths, tmp_path):
    """AI_CHIP_IMPORTS_DATA moves every input path with it."""
    root = tmp_path / "inputs"
    p = reload_paths(AI_CHIP_IMPORTS_DATA=root)
    assert p.DATA_DIR == root
    assert p.CUSTOMS_DIR == root / "customs"
    assert p.EXTERNAL_DIR == root / "external"
    assert p.FX_CSV == root / "fx" / "imf_fx_monthly.csv"
    assert p.raw_dir("hk") == root / "customs" / "hk" / "raw"
    assert p.clean_dir("hk") == root / "customs" / "hk" / "clean"


def test_output_root_follows_its_environment_variable(reload_paths, tmp_path):
    """AI_CHIP_IMPORTS_OUTPUT moves every output file, each stamped with the run date."""
    root = tmp_path / "outputs"
    p = reload_paths(AI_CHIP_IMPORTS_OUTPUT=root)
    assert p.OUTPUT_DIR == root
    assert p.output_path("gpu_country_flows") == root / f"gpu_country_flows_{p.STAMP}.csv"
    assert p.output_path("manifest", "json").name == f"manifest_{p.STAMP}.json"
    assert re.fullmatch(r"\d{8}", p.STAMP)


def test_roots_default_to_the_repository(reload_paths):
    """Without overrides, data and outputs live beside the package's source tree."""
    p = reload_paths(AI_CHIP_IMPORTS_DATA=None, AI_CHIP_IMPORTS_OUTPUT=None)
    assert p.REPO_DIR == p.PACKAGE_DIR.parents[1]
    assert p.DATA_DIR == p.REPO_DIR / "data"
    assert p.OUTPUT_DIR == p.REPO_DIR / "output"


def test_assumptions_ship_with_the_package(reload_paths, tmp_path):
    """Method assumptions stay inside the package whatever the data root."""
    p = reload_paths(AI_CHIP_IMPORTS_DATA=tmp_path)
    assert p.ASSUMPTIONS_DIR == p.PACKAGE_DIR / "assumptions"
    assert (p.ASSUMPTIONS_DIR / "captive_geography.csv").is_file()
