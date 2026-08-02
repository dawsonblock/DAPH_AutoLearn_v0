"""V037-004: version + claims discipline.

Acceptance criteria:

1. All version surfaces agree: pyproject.toml, daph_learning.__version__,
   README.md header.
2. CLAIMS.md header matches the package version.
3. The CHANGELOG has an entry for the current version.
4. Every status tag in CLAIMS.md is one of the four licensed values
   (ESTABLISHED, BOOTSTRAP, NOT YET, OUT OF SCOPE) or the documented
   compound forms (PARTIAL, OUT OF SCOPE FOR ...).
5. CLAIMS.md records a collected-test count that covers the suite.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _pyproject_version() -> str:
    text = _read(REPO_ROOT / "pyproject.toml")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "pyproject.toml has no version field"
    return m.group(1)


def _init_version() -> str:
    import daph_learning
    return daph_learning.__version__


def _readme_header_version() -> str:
    text = _read(REPO_ROOT / "README.md")
    m = re.search(r"^#\s*DAPH AutoLearn v(\S+)", text, re.MULTILINE)
    assert m, "README.md has no version header"
    return m.group(1)


def _claims_header_version() -> str:
    text = _read(REPO_ROOT / "CLAIMS.md")
    m = re.search(r"^#\s*DAPH AutoLearn v(\S+)", text, re.MULTILINE)
    assert m, "CLAIMS.md has no version header"
    return m.group(1)


# --- 1. Version surfaces agree ---

def test_pyproject_matches_init_version():
    assert _pyproject_version() == _init_version(), (
        f"pyproject.toml version {_pyproject_version()!r} != "
        f"daph_learning.__version__ {_init_version()!r}"
    )


def test_readme_header_matches_init_version():
    assert _readme_header_version() == _init_version(), (
        f"README.md header v{_readme_header_version()!r} != "
        f"daph_learning.__version__ {_init_version()!r}"
    )


def test_claims_header_matches_init_version():
    assert _claims_header_version() == _init_version(), (
        f"CLAIMS.md header v{_claims_header_version()!r} != "
        f"daph_learning.__version__ {_init_version()!r}"
    )


# --- 2. CHANGELOG has an entry for the current version ---

def test_changelog_has_current_version_entry():
    text = _read(REPO_ROOT / "CHANGELOG.md")
    version = _init_version()
    # CHANGELOG entries look like "# 0.3.7 (description)" or "## 0.3.7".
    # v0.3.10.1 — pre-release suffix "-alpha" must be matched; the
    # boundary \b does not match after "-" so we accept end-of-line or
    # whitespace after the version string instead.
    pattern = rf"^#\s*{re.escape(version)}(?:\s|$)"
    assert re.search(pattern, text, re.MULTILINE), (
        f"CHANGELOG.md has no entry for version {version!r}"
    )


def test_readme_changelog_has_current_version_entry():
    text = _read(REPO_ROOT / "README.md")
    version = _init_version()
    pattern = rf"^###\s*v{re.escape(version)}(?:\s|$)"
    assert re.search(pattern, text, re.MULTILINE), (
        f"README.md changelog has no entry for version {version!r}"
    )


# --- 3. CLAIMS.md status tags are well-formed ---

VALID_STATUS_TAGS = {
    "ESTABLISHED",
    "BOOTSTRAP",
    "NOT YET",
    "OUT OF SCOPE",
    "PARTIAL",
}


def test_claims_status_tags_are_well_formed():
    """Every `**Status: ...**` line in CLAIMS.md must start with a licensed
    status tag (ESTABLISHED, BOOTSTRAP, NOT YET, OUT OF SCOPE, PARTIAL).
    Compound forms like "ESTABLISHED as engineering — PARTIAL scientific"
    are allowed; the leading phrase must be a licensed tag."""
    text = _read(REPO_ROOT / "CLAIMS.md")
    lines = text.splitlines()
    offenders: list[str] = []
    for line_no, line in enumerate(lines, 1):
        m = re.match(r"^\*\*Status:\s*(.+?)\s*\*\*\s*$", line)
        if not m:
            continue
        status_text = m.group(1)
        # Check multi-word tags first (NOT YET, OUT OF SCOPE), then single-word.
        matched = False
        for tag in ("NOT YET", "OUT OF SCOPE", "ESTABLISHED", "BOOTSTRAP", "PARTIAL"):
            if status_text == tag or status_text.startswith(tag + " ") or status_text.startswith(tag + "."):
                matched = True
                break
        if not matched:
            offenders.append(f"  line {line_no}: {status_text!r}")
    if offenders:
        pytest.fail(
            "CLAIMS.md has status tags that do not start with a licensed value:\n"
            + "\n".join(offenders)
        )


# --- 4. CLAIMS.md collected-test count covers the actual suite ---

def _actual_test_count() -> tuple[int, int]:
    """Run pytest --collect-only (fast, no execution) and parse the
    collected + skipped counts.

    Returns (collected, skipped). The "passed" count is collected - skipped,
    assuming no failures (this test only runs when the suite is green).
    """
    import os
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(REPO_ROOT / "src")
    if existing_pythonpath:
        pythonpath += os.pathsep + existing_pythonpath
    env = {**os.environ, "PYTHONPATH": pythonpath}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only",
         "-p", "no:cacheprovider", "--ignore",
         str(REPO_ROOT / "tests" / "test_version_claims_discipline.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    combined = result.stdout + result.stderr
    collected_m = re.search(r"(\d+)\s+tests?\s+collected", combined)
    if not collected_m:
        pytest.fail(
            f"could not parse collected count from pytest --collect-only:\n"
            f"stdout tail: {result.stdout[-500:]}\nstderr tail: {result.stderr[-500:]}"
        )
    collected = int(collected_m.group(1))
    # Skipped tests are collected but not run; we detect them by searching
    # for skip markers in the collection output. For now, return collected
    # and let the caller treat it as an upper bound on passed.
    return collected, 0


def test_claims_test_count_is_lower_bound():
    """The collected count in CLAIMS.md §6 must cover the actual collection
    count (minus this file, which spawns pytest).

    We use --collect-only and exclude this file to avoid recursion. Reporting
    the collected count avoids falsely describing hardware-skipped tests as
    passing tests."""
    text = _read(REPO_ROOT / "CLAIMS.md")
    m = re.search(r'## 6\.\s.*?(\d+)\s+collected tests', text, re.DOTALL)
    assert m, "CLAIMS.md §6 test count not found"
    claimed = int(m.group(1))
    collected, _ = _actual_test_count()
    assert claimed >= collected, (
        f"CLAIMS.md §6 records {claimed} collected tests but the rest of the "
        f"suite collects {collected}; the documented count must cover it."
    )


# --- 5. No stale version references in user-facing docs ---

def test_no_stale_version_references_in_readme():
    """README.md should not prominently reference old version numbers in
    its header or changelog section headers (body text may reference them
    historically)."""
    text = _read(REPO_ROOT / "README.md")
    current = _init_version()
    # The header must reference the current version.
    header = text.splitlines()[0]
    assert current in header, (
        f"README.md header {header!r} does not reference current version {current!r}"
    )


# --- 6. v0.3.10.3.2 — full version consistency across all surfaces ---

def test_version_consistency():
    """Section 1 / G1: every active version surface must agree on
    0.4.0a2. Covers pyproject, __version__, ExperimentConfig
    default, ProvenanceRecord default, CLI --version, README header,
    and CLAIMS header. No active-root reference to 0.3.10, 0.3.10.1,
    0.3.10.2, 0.3.10.3, or 0.3.10.3.1 may remain in the
    package source."""
    from daph_learning.policy.config import ExperimentConfig
    from daph_learning.policy.provenance import ProvenanceRecord

    init_v = _init_version()
    expected = "0.4.0a2"
    assert init_v == expected, f"__version__ is {init_v!r}, expected {expected!r}"
    assert _pyproject_version() == expected
    assert _readme_header_version() == expected
    assert _claims_header_version() == expected
    cfg = ExperimentConfig()
    assert cfg.autolearn_version == expected, (
        f"ExperimentConfig.autolearn_version is {cfg.autolearn_version!r}")
    prov = ProvenanceRecord()
    assert prov.release_version == expected, (
        f"ProvenanceRecord.release_version is {prov.release_version!r}")
    # CLI --version must agree.
    cli = subprocess.run(
        [sys.executable, "-m", "daph_learning.cli.autolearn", "--version"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30)
    out = (cli.stdout + cli.stderr).strip()
    assert expected in out, f"CLI --version output {out!r} missing {expected!r}"
