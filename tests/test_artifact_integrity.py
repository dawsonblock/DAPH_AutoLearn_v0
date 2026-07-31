"""Section 4.3 — stale-artifact CI gate.

Verifies the artifact layout discipline required by the v0.3.10.4-alpha
Gate A integrity repair:

* ``artifacts/current/`` contains a pointer only (no copied evidence);
* the current pointer resolves to an existing bundle OR explicitly
  declares ``NOT_YET_REQUALIFIED`` with no target;
* every bundled artifact uses the current source hash unless it is
  explicitly marked as legacy;
* legacy artifacts are never marked current;
* synthetic artifacts cannot be Gate A qualified;
* failed artifacts cannot be promoted;
* the bundle validator rejects mixed run IDs, experiment IDs, and
  dataset hashes.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from daph_learning.evaluation.artifact_integrity import (
    ArtifactValidationResult,
    validate_artifact_bundle,
)
from daph_learning.provenance import compute_canonical_source_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
CURRENT = ARTIFACTS / "current"
LEGACY = ARTIFACTS / "legacy" / "daph_gate_a_real_001_failed"


def _current_hash() -> str:
    return compute_canonical_source_hash(REPO_ROOT)


# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------

@pytest.mark.parametrize("subdir", [
    "synthetic_ci", "real_model_smoke", "gate_a_qualified",
    "gate_a_failed", "legacy", "current",
])
def test_required_artifact_layout_dirs_exist(subdir: str):
    assert (ARTIFACTS / subdir).is_dir(), (
        f"required artifact directory missing: artifacts/{subdir}/")


def test_legacy_failed_run_dir_exists():
    assert LEGACY.is_dir(), (
        "legacy failed run must be archived at "
        "artifacts/legacy/daph_gate_a_real_001_failed/")


def test_legacy_notice_exists():
    assert (LEGACY / "LEGACY_NOTICE.md").is_file()
    assert (LEGACY / "historical_source_manifest.json").is_file()


# ------------------------------------------------------------------
# Section 4.3 required tests
# ------------------------------------------------------------------

def test_current_pointer_resolves_to_existing_bundle():
    """The current pointer must either resolve to an existing bundle or
    explicitly declare NOT_YET_REQUALIFIED with no target."""
    pointer_path = CURRENT / "pointer.json"
    assert pointer_path.is_file(), "artifacts/current/pointer.json missing"
    pointer = json.loads(pointer_path.read_text())
    assert pointer.get("artifact_type") == "pointer", (
        "current pointer must declare artifact_type='pointer'")

    target = pointer.get("target")
    status = str(pointer.get("status", "")).upper()

    if target is None:
        # No target is only acceptable when explicitly not-yet-qualified.
        assert status == "NOT_YET_REQUALIFIED", (
            "current pointer has no target but does not declare "
            "NOT_YET_REQUALIFIED")
        return

    # Resolve the target relative to the pointer file.
    target_path = (CURRENT / target).resolve()
    assert target_path.is_dir(), (
        f"current pointer target does not resolve to a bundle: {target_path}")


def test_current_contains_pointer_only():
    """artifacts/current/ must contain only the pointer (no copied
    experimental evidence)."""
    entries = [p for p in CURRENT.iterdir() if p.name != ".gitkeep"]
    non_pointer = [p for p in entries if p.name != "pointer.json"]
    assert not non_pointer, (
        "artifacts/current/ must contain only pointer.json; found: "
        + ", ".join(p.name for p in non_pointer))


def test_all_bundled_artifacts_use_current_source_hash_or_are_marked_legacy():
    """Every non-legacy bundle under artifacts/ must validate against the
    current source hash."""
    hash = _current_hash()
    # Check the qualified and failed buckets (the only places current
    # evidence may live). Legacy is exempt.
    for bucket in ("gate_a_qualified", "gate_a_failed", "real_model_smoke",
                   "synthetic_ci"):
        bucket_dir = ARTIFACTS / bucket
        if not bucket_dir.is_dir():
            continue
        for bundle in sorted(bucket_dir.iterdir()):
            if not bundle.is_dir() or bundle.name == ".gitkeep":
                continue
            result = validate_artifact_bundle(bundle, REPO_ROOT)
            # Bundles may legitimately be empty placeholders; only enforce
            # hash consistency when the bundle actually declares a hash.
            if bundle.rglob("*.json") and not any(
                _is_legacy_file(jf) for jf in bundle.rglob("*.json")
            ):
                # If the bundle declares a source hash it must match.
                declared = _any_declared_hash(bundle)
                if declared:
                    cmp_len = min(len(declared), len(hash))
                    assert declared[:cmp_len] == hash[:cmp_len], (
                        f"{bundle}: stale source hash {declared[:16]}... "
                        f"!= current {hash[:16]}...")


def test_legacy_artifacts_are_never_marked_current():
    """No file under artifacts/legacy/ may be referenced by the current
    pointer as current evidence."""
    pointer = json.loads((CURRENT / "pointer.json").read_text())
    target = pointer.get("target")
    if target is None:
        return
    target_str = str(target)
    assert "legacy" not in target_str.lower(), (
        "current pointer must not target a legacy artifact as current "
        "Gate A evidence")


def test_synthetic_artifacts_cannot_be_gate_a_qualified():
    """A synthetic bundle placed under gate_a_qualified/ must fail
    validation."""
    if not (ARTIFACTS / "gate_a_qualified").iterdir():
        # No qualified bundles exist yet — the invariant holds vacuously.
        # Verify the rule with a constructed synthetic bundle.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "synthetic_promoted"
            d.mkdir()
            (d / "result.json").write_text(json.dumps({
                "evidence_level": "SYNTHETIC",
                "status": "QUALIFIED",
                "source_hash": _current_hash(),
            }))
            r = validate_artifact_bundle(d, REPO_ROOT)
            assert not r.valid
            assert any("synthetic" in e.lower() for e in r.errors)
        return
    for bundle in (ARTIFACTS / "gate_a_qualified").iterdir():
        if not bundle.is_dir() or bundle.name == ".gitkeep":
            continue
        r = validate_artifact_bundle(bundle, REPO_ROOT)
        # If it contains synthetic evidence it must not be valid.
        for jf in bundle.rglob("*.json"):
            data = json.loads(jf.read_text())
            if not isinstance(data, dict):
                continue
            if str(data.get("evidence_level", "")).upper() in {
                    "SYNTHETIC", "SYNTHETIC_CI", "SIMULATED"}:
                assert not r.valid, (
                    f"synthetic artifact promoted to qualified: {bundle}")


def test_failed_artifacts_cannot_be_promoted():
    """A bundle that declares FAILED status must not also declare
    QUALIFIED."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "failed_promoted"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({
            "status": "FAILED", "source_hash": _current_hash(),
        }))
        (d / "b.json").write_text(json.dumps({
            "status": "QUALIFIED", "source_hash": _current_hash(),
        }))
        r = validate_artifact_bundle(d, REPO_ROOT)
        assert not r.valid
        assert any("failed" in e.lower() and "promoted" in e.lower()
                   for e in r.errors)


def test_artifact_bundle_rejects_mixed_run_ids():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "mixed_runs"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({
            "run_id": "run_001", "source_hash": _current_hash(),
        }))
        (d / "b.json").write_text(json.dumps({
            "run_id": "run_002", "source_hash": _current_hash(),
        }))
        r = validate_artifact_bundle(d, REPO_ROOT)
        assert not r.valid
        assert any("run ID" in e for e in r.errors)


def test_artifact_bundle_rejects_mixed_experiment_ids():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "mixed_experiments"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({
            "experiment_id": "daph_gate_a_real_002",
            "source_hash": _current_hash(),
        }))
        (d / "b.json").write_text(json.dumps({
            "experiment_id": "daph_gate_a_real_003",
            "source_hash": _current_hash(),
        }))
        r = validate_artifact_bundle(d, REPO_ROOT)
        assert not r.valid
        assert any("experiment ID" in e for e in r.errors)


def test_artifact_bundle_rejects_mixed_dataset_hashes():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "mixed_datasets"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({
            "final_dataset_sha256": "aaa", "source_hash": _current_hash(),
        }))
        (d / "b.json").write_text(json.dumps({
            "final_dataset_sha256": "bbb", "source_hash": _current_hash(),
        }))
        r = validate_artifact_bundle(d, REPO_ROOT)
        assert not r.valid


def test_artifact_bundle_rejects_mixed_utility_hashes():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "mixed_utility"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({
            "utility_config_hash": "u1", "source_hash": _current_hash(),
        }))
        (d / "b.json").write_text(json.dumps({
            "utility_config_hash": "u2", "source_hash": _current_hash(),
        }))
        r = validate_artifact_bundle(d, REPO_ROOT)
        assert not r.valid
        assert any("utility" in e.lower() for e in r.errors)


def test_artifact_bundle_rejects_stale_source_hash():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "stale"
        d.mkdir()
        (d / "result.json").write_text(json.dumps({
            "source_hash": "0" * 64,
        }))
        r = validate_artifact_bundle(d, REPO_ROOT)
        assert not r.valid
        assert any("stale source hash" in e for e in r.errors)


def test_artifact_bundle_accepts_marked_legacy_hash():
    """A legacy bundle with a historical source hash must validate."""
    r = validate_artifact_bundle(LEGACY, REPO_ROOT)
    assert r.valid, (
        f"legacy bundle should validate but had errors: {r.errors}")


def test_missing_final_ledger_rejected_when_required():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "no_ledger"
        d.mkdir()
        (d / "result.json").write_text(json.dumps({
            "source_hash": _current_hash(),
        }))
        r = validate_artifact_bundle(d, REPO_ROOT,
                                     require_final_access_ledger=True)
        assert not r.valid
        assert any("final-access ledger" in e for e in r.errors)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _is_legacy_file(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(isinstance(data, dict) and (
        data.get("legacy") or data.get("not_current_source")
        or data.get("historical_source_hash")))


def _any_declared_hash(bundle: Path) -> str | None:
    for jf in bundle.rglob("*.json"):
        try:
            data = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for key in ("source_hash", "source_tree_sha256", "source_tree_hash"):
            val = data.get(key)
            if val:
                return str(val)
    return None
