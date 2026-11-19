"""v0.3.10.4-alpha — Section 4 / G32: current artifact tree contains
no stale source hash.

Verifies that the current artifact pointer (and any current evidence)
records a source hash matching the current source tree. Under the
v0.3.10.4-alpha layout, ``artifacts/current/`` holds only a pointer;
the full bundle-validation CI gate lives in ``test_artifact_integrity.py``.
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
    """Section 4: every artifact in artifacts/current/ must record a
    source hash matching the current source tree (or be a
    NOT_YET_REQUALIFIED pointer with no target bundle)."""
    from daph_learning.provenance import compute_canonical_source_hash
    current_hash = compute_canonical_source_hash(REPO_ROOT)

    artifact_files = list(CURRENT_ARTIFACTS_DIR.rglob("*.json"))
    if not artifact_files:
        pytest.skip("no current artifacts yet — will be populated by gate run")

    stale = []
    for af in artifact_files:
        with open(af) as f:
            data = json.load(f)
        # A NOT_YET_REQUALIFIED pointer/status with no target asserts no
        # evidence and is always acceptable.
        status = str(data.get("status", "")).upper()
        gate_status = str(data.get("gate_a_status", "")).upper()
        evidence = str(data.get("evidence_level", "")).upper()
        is_unqualified = (
            status == "NOT_YET_REQUALIFIED"
            or gate_status == "NOT_YET_REQUALIFIED"
            or evidence == "NOT_YET_REQUALIFIED"
        )
        if is_unqualified and data.get("target") is None:
            continue
        artifact_hash = (
            data.get("source_hash")
            or data.get("source_tree_sha256")
            or data.get("source_tree_hash")
            or ""
        )
        if not artifact_hash or artifact_hash.startswith("PLACEHOLDER"):
            stale.append(f"{af}: missing/placeholder source hash field")
            continue
        cmp_len = min(len(artifact_hash), len(current_hash))
        if artifact_hash[:cmp_len] != current_hash[:cmp_len]:
            stale.append(
                f"{af}: hash={artifact_hash[:16]}... "
                f"current={current_hash[:16]}...")

    assert not stale, (
        f"Stale artifacts in artifacts/current/:\n" + "\n".join(stale))
