"""Section 1 / G01: all version surfaces match.

Verifies that every active version surface agrees on the canonical
version declared in ``pyproject.toml`` and that no active source file
defaults to an older version string.

The canonical version is read from ``pyproject.toml`` (single source of
truth). Tests should NOT hard-code a version string — that creates the
exact drift this test is designed to catch.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Older versions that must NOT appear as defaults in active source.
FORBIDDEN_DEFAULTS = (
    "0.3.10.3.1-alpha",
    "0.3.10.3-alpha",
    "0.3.10.2-alpha",
    "0.3.10.1-alpha",
    "0.3.10-alpha",
    "0.3.10.5-alpha",
)


def _pyproject_version() -> str:
    """Read the canonical version from pyproject.toml — single source of truth."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    pp = REPO_ROOT / "pyproject.toml"
    with open(pp, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


@pytest.fixture(scope="module")
def target_version() -> str:
    return _pyproject_version()


def test_init_version(target_version):
    import daph_learning
    assert daph_learning.__version__ == target_version, (
        f"__version__ is {daph_learning.__version__!r}, "
        f"expected {target_version!r}")


def test_pyproject_version_is_canonical():
    """pyproject.toml is the canonical source — just verify it's readable."""
    v = _pyproject_version()
    assert v, "pyproject.toml must declare a version"
    assert "-alpha" in v or "-beta" in v or "-rc" in v or v[-1].isdigit(), (
        f"version {v!r} should look like a release or pre-release")


def test_experiment_config_version(target_version):
    from daph_learning.policy.config import ExperimentConfig
    cfg = ExperimentConfig()
    assert cfg.autolearn_version == target_version, (
        f"ExperimentConfig.autolearn_version is {cfg.autolearn_version!r}")


def test_provenance_record_version(target_version):
    from daph_learning.policy.provenance import ProvenanceRecord
    prov = ProvenanceRecord()
    assert prov.release_version == target_version, (
        f"ProvenanceRecord.release_version is {prov.release_version!r}")


def test_final_access_ledger_version(target_version):
    from daph_learning.policy.stage import FinalAccessLedger
    ledger = FinalAccessLedger()
    assert ledger.release_version == target_version, (
        f"FinalAccessLedger.release_version is {ledger.release_version!r}")


def test_crossover_generator_version():
    from daph_learning.data.crossover_benchmark import GENERATOR_VERSION
    assert "0.3.10.3.2" in GENERATOR_VERSION, (
        f"GENERATOR_VERSION is {GENERATOR_VERSION!r}")


def test_cli_version_output(target_version):
    cli = subprocess.run(
        [sys.executable, "-m", "daph_learning.cli.autolearn", "--version"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30)
    out = (cli.stdout + cli.stderr).strip()
    assert target_version in out, (
        f"CLI --version output {out!r} missing {target_version!r}")


def test_readme_header_version(target_version):
    readme = (REPO_ROOT / "README.md").read_text()
    first_line = readme.split("\n")[0]
    assert target_version in first_line, (
        f"README header {first_line!r} missing {target_version!r}")


def test_claims_header_version(target_version):
    claims = (REPO_ROOT / "CLAIMS.md").read_text()
    first_line = claims.split("\n")[0]
    assert target_version in first_line, (
        f"CLAIMS header {first_line!r} missing {target_version!r}")


def test_no_forbidden_version_defaults_in_source():
    """No active source file under src/ should default to an older
    version string. Docstrings describing historical versions are
    acceptable, but default values (e.g. ``= "0.3.10.3.1-alpha"``)
    are not."""
    src_dir = REPO_ROOT / "src"
    violations = []
    # Pattern: version string assigned as a default value.
    # Matches: = "0.3.10.x-alpha" or : str = "0.3.10.x-alpha"
    pattern = re.compile(
        r'=\s*["\'](' + "|".join(re.escape(v) for v in FORBIDDEN_DEFAULTS) + r')["\']'
    )
    for py_file in src_dir.rglob("*.py"):
        text = py_file.read_text()
        for match in pattern.finditer(text):
            line_num = text[:match.start()].count("\n") + 1
            violations.append(f"{py_file}:{line_num}: {match.group()}")
    assert not violations, (
        f"Forbidden version defaults found in source:\n" +
        "\n".join(violations))


def test_no_forbidden_version_defaults_in_scripts():
    """No active script should default to an older version string."""
    scripts_dir = REPO_ROOT / "scripts"
    violations = []
    pattern = re.compile(
        r'=\s*["\'](' + "|".join(re.escape(v) for v in FORBIDDEN_DEFAULTS) + r')["\']'
    )
    for py_file in scripts_dir.rglob("*.py"):
        text = py_file.read_text()
        for match in pattern.finditer(text):
            line_num = text[:match.start()].count("\n") + 1
            violations.append(f"{py_file}:{line_num}: {match.group()}")
    assert not violations, (
        f"Forbidden version defaults found in scripts:\n" +
        "\n".join(violations))
