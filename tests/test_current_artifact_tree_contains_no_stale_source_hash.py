"""v0.3.10.3.2-alpha — Section 4 / G32: current artifact tree contains
no stale source hash.

Verifies that every artifact in ``artifacts/current/`` has a
``source_tree_sha256`` that matches the current source tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CURRENT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "current"


def test_current_artifacts_dir_exists():
    assert CURRENT_ARTIFACTS_DIR.is_dir(), (
        f"artifacts/current/ directory not found at {CURRENT_ARTIFACTS_DIR}")


def test_current_artifact_tree_contains_no_stale_source_hash():
    """Section 4: every artifact in artifacts/current/ must have a
    source_tree_sha256 matching the current source tree."""
    from daph_learning.provenance import compute_source_tree_sha256
    current_hash = compute_source_tree_sha256(REPO_ROOT)

    artifact_files = list(CURRENT_ARTIFACTS_DIR.rglob("*.json"))
    if not artifact_files:
        pytest.skip("no current artifacts yet — will be populated by gate run")

    stale = []
    for af in artifact_files:
        with open(af) as f:
            data = json.load(f)
        artifact_hash = data.get("source_tree_sha256", "")
        if not artifact_hash:
            stale.append(f"{af}: missing source_tree_sha256 field")
            continue
        cmp_len = min(len(artifact_hash), len(current_hash))
        if artifact_hash[:cmp_len] != current_hash[:cmp_len]:
            stale.append(
                f"{af}: hash={artifact_hash[:16]}... "
                f"current={current_hash[:16]}...")

    assert not stale, (
        f"Stale artifacts in artifacts/current/:\n" + "\n".join(stale))


def test_archived_artifacts_are_not_in_current():
    """Section 4: previous artifacts must be in archive/, not current/."""
    archive_dir = REPO_ROOT / "artifacts" / "archive"
    assert archive_dir.is_dir(), "artifacts/archive/ directory not found"
    # The old root-level artifacts should not exist.
    old_files = [
        REPO_ROOT / "artifacts" / "real_qwen_experiment_result.json",
        REPO_ROOT / "artifacts" / "smoke_real_model_result.json",
    ]
    for f in old_files:
        assert not f.exists(), (
            f"stale artifact still at root level: {f}")
