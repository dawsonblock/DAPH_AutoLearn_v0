"""v0.3.10.3.1-alpha — artifact directory discipline + source hash
enforcement (Sections 32-35).

Section 32: artifact directory discipline
  * ``artifacts/<policy_id>/`` contains: ``policy.json``,
    ``calibration.json``, ``feature_transform.json``, ``ood_model.json``
  * ``results/<experiment_id>/`` contains: ``config.json``,
    ``metadata.json``, ``results.jsonl``, ``summary.json``
  * No ad-hoc files in the repo root.

Section 33: source tree hash
  * Every artifact records ``source_tree_hash`` (SHA-256 of all .py
    files under src/).
  * Re-running with a different source tree hash produces a warning
    (or error in strict mode), not silent reuse.

Section 34: test collection hash
  * ``pytest --collect-only -q`` output is hashed and recorded.

Section 35: re-run test report
  * A function that re-runs the test suite and records pass/fail
    counts, recording the result in the artifact directory.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --- Section 32: artifact directory discipline ---

POLICY_ARTIFACT_FILES = (
    "policy.json",
    "calibration.json",
    "feature_transform.json",
    "ood_model.json",
)

RESULT_FILES = (
    "config.json",
    "metadata.json",
    "results.jsonl",
    "summary.json",
)


def validate_policy_artifact_dir(path: str | Path) -> list[str]:
    """Section 32: validate that a policy artifact directory contains
    the expected files. Returns a list of missing files (empty = OK)."""
    p = Path(path)
    missing = [f for f in POLICY_ARTIFACT_FILES if not (p / f).exists()]
    return missing


def validate_result_dir(path: str | Path) -> list[str]:
    """Section 32: validate that a result directory contains the
    expected files. Returns a list of missing files."""
    p = Path(path)
    return [f for f in RESULT_FILES if not (p / f).exists()]


# --- Section 33: source tree hash ---

def compute_source_tree_hash(src_dir: str | Path) -> str:
    """Section 33: SHA-256 of all .py files under src/.

    Files are sorted by relative path and concatenated, then hashed.
    This binds every artifact to the exact source tree that generated
    it.
    """
    src = Path(src_dir)
    h = hashlib.sha256()
    py_files = sorted(src.rglob("*.py"), key=lambda p: str(p.relative_to(src)))
    for f in py_files:
        rel = str(f.relative_to(src))
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def assert_source_tree_matches(
    artifact_hash: str, src_dir: str | Path, *, strict: bool = False,
) -> None:
    """Section 33: warn (or error in strict mode) if the current source
    tree hash differs from the artifact's recorded hash."""
    current = compute_source_tree_hash(src_dir)
    if current != artifact_hash:
        msg = (f"source tree hash mismatch: artifact={artifact_hash[:16]}... "
               f"current={current[:16]}... — artifacts may be stale")
        if strict:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}", file=sys.stderr)


# --- Section 34: test collection hash ---

def compute_test_collection_hash(repo_root: str | Path) -> str:
    """Section 34: hash of ``pytest --collect-only -q`` output.

    This binds artifacts to the exact test collection that was active
    when they were generated.
    """
    root = Path(repo_root)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=root, capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


# --- Section 35: re-run test report ---

@dataclass
class ReRunTestReport:
    """Section 35: result of re-running the test suite."""
    n_passed: int
    n_failed: int
    n_errors: int
    n_skipped: int
    exit_code: int
    collection_hash: str
    source_tree_hash: str
    raw_output: str = ""

    @property
    def all_passed(self) -> bool:
        return self.n_failed == 0 and self.n_errors == 0 and self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_errors": self.n_errors,
            "n_skipped": self.n_skipped,
            "exit_code": self.exit_code,
            "all_passed": self.all_passed,
            "collection_hash": self.collection_hash,
            "source_tree_hash": self.source_tree_hash,
        }


def rerun_tests(
    repo_root: str | Path,
    src_dir: str | Path,
    *,
    test_paths: list[str] | None = None,
    timeout: int = 600,
) -> ReRunTestReport:
    """Section 35: re-run the test suite and record the result.

    Parameters
    ----------
    repo_root : path
    src_dir : path
        Used to compute the source tree hash bound to the report.
    test_paths : list of str, optional
        Specific test paths to re-run. Defaults to the full suite.
    timeout : int
        Subprocess timeout in seconds.
    """
    root = Path(repo_root)
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"]
    if test_paths:
        cmd.extend(test_paths)
    result = subprocess.run(
        cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
    output = result.stdout + result.stderr
    n_passed = n_failed = n_errors = n_skipped = 0
    for line in output.splitlines():
        if "passed" in line and "failed" not in line:
            # e.g. "69 passed in 1.2s"
            try:
                n_passed = int(line.split("passed")[0].strip().split()[-1])
            except (ValueError, IndexError):
                pass
        if "failed" in line:
            try:
                n_failed = int(line.split("failed")[0].strip().split()[-1])
            except (ValueError, IndexError):
                pass
        if "error" in line:
            try:
                n_errors = int(line.split("error")[0].strip().split()[-1])
            except (ValueError, IndexError):
                pass
        if "skipped" in line:
            try:
                n_skipped = int(line.split("skipped")[0].strip().split()[-1])
            except (ValueError, IndexError):
                pass
    # Fallback: if the summary line was not captured (some pytest versions
    # put it on a line that gets buffered differently), count progress dots.
    if n_passed == 0 and n_failed == 0 and n_errors == 0:
        # Count dots (passed), F's (failed), E's (errors), s (skipped).
        for line in output.splitlines():
            if "[100%]" in line or set(line.strip()) <= set(".FEsx"):
                n_passed += line.count(".")
                n_failed += line.count("F")
                n_errors += line.count("E")
                n_skipped += line.count("s")
    return ReRunTestReport(
        n_passed=n_passed, n_failed=n_failed, n_errors=n_errors,
        n_skipped=n_skipped, exit_code=result.returncode,
        collection_hash=compute_test_collection_hash(root),
        source_tree_hash=compute_source_tree_hash(src_dir),
        raw_output=output[-2000:],
    )


__all__ = [
    "POLICY_ARTIFACT_FILES",
    "RESULT_FILES",
    "ReRunTestReport",
    "assert_source_tree_matches",
    "compute_source_tree_hash",
    "compute_test_collection_hash",
    "rerun_tests",
    "validate_policy_artifact_dir",
    "validate_result_dir",
]
