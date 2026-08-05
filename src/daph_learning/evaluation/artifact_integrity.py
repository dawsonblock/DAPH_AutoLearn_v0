"""v0.3.10.3.2-alpha — artifact directory discipline + source hash
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
import json
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
    """Section 2: canonical SHA-256 of the source tree.

    .. deprecated::
        This function now delegates to the canonical
        :func:`daph_learning.provenance.compute_source_tree_sha256`.
        All new code should call the canonical function directly.

    For backward compatibility, if ``src_dir`` is a subdirectory of a
    repository root (e.g. ``repo/src``), the hash is computed over the
    repository root. If ``src_dir`` is the repository root itself, the
    hash is computed over it directly.

    Returns the **full 64-character** SHA-256 hex digest.
    """
    from daph_learning.provenance import compute_source_tree_sha256
    src = Path(src_dir)
    # If src_dir is "repo/src", walk up to the repo root.
    # If src_dir is the repo root itself, use it directly.
    root = src
    if src.name == "src":
        root = src.parent
    return compute_source_tree_sha256(root)


def assert_source_tree_matches(
    artifact_hash: str, src_dir: str | Path, *, strict: bool = False,
) -> None:
    """Section 3: warn (or error in strict mode) if the current source
    tree hash differs from the artifact's recorded hash.

    Both the artifact hash and current hash are compared as full
    64-character SHA-256 digests. If the artifact hash is a truncated
    16-character hash (legacy), only the first 16 characters are
    compared for backward compatibility.
    """
    current = compute_source_tree_hash(src_dir)
    # Support both full 64-char and legacy 16-char artifact hashes.
    cmp_len = min(len(artifact_hash), len(current))
    if current[:cmp_len] != artifact_hash[:cmp_len]:
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
        raw_output=output[-4000:],
    )


# --- Section 3: artifact integrity fields ---

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "release_version",
    "source_tree_sha256",
    "config_sha256",
    "created_at",
)

# Additional fields required for model experiment artifacts.
MODEL_ARTIFACT_FIELDS: tuple[str, ...] = (
    "model_id",
    "model_revision",
    "tokenizer_revision",
)


def assert_artifact_has_integrity_fields(
    artifact: dict, *, is_model_artifact: bool = False,
) -> None:
    """Section 3: verify that an artifact contains all required
    integrity fields.

    Required for all artifacts:
    - ``release_version``
    - ``source_tree_sha256``
    - ``config_sha256``
    - ``created_at``

    For model experiment artifacts also:
    - ``model_id``
    - ``model_revision``
    - ``tokenizer_revision``
    """
    required = REQUIRED_ARTIFACT_FIELDS
    if is_model_artifact:
        required = required + MODEL_ARTIFACT_FIELDS
    missing = [f for f in required if f not in artifact or artifact[f] is None]
    if missing:
        raise ValueError(
            f"artifact missing required integrity fields: {missing}")


def assert_artifact_matches_current_source(
    artifact: dict, current_hash: str, *, strict: bool = True,
) -> None:
    """Section 3: verify that an artifact's ``source_tree_sha256``
    matches the current source tree hash.

    If ``strict=True`` (default), raises ``RuntimeError`` on mismatch.
    Otherwise prints a warning.
    """
    artifact_hash = artifact.get("source_tree_sha256", "")
    if not artifact_hash:
        raise ValueError(
            "artifact has no source_tree_sha256 field — cannot verify "
            "integrity (Section 3)")
    # Support both full 64-char and legacy 16-char hashes.
    cmp_len = min(len(artifact_hash), len(current_hash))
    if artifact_hash[:cmp_len] != current_hash[:cmp_len]:
        msg = (
            f"artifact source_tree_sha256 mismatch: "
            f"artifact={artifact_hash[:16]}... current={current_hash[:16]}... "
            f"— stale artifact (Section 3)")
        if strict:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}", file=sys.stderr)


__all__ = [
    "POLICY_ARTIFACT_FILES",
    "RESULT_FILES",
    "ReRunTestReport",
    "REQUIRED_ARTIFACT_FIELDS",
    "ArtifactValidationResult",
    "assert_artifact_has_integrity_fields",
    "assert_artifact_matches_current_source",
    "assert_source_tree_matches",
    "compute_source_tree_hash",
    "compute_test_collection_hash",
    "rerun_tests",
    "validate_artifact_bundle",
    "validate_policy_artifact_dir",
    "validate_result_dir",
]


# ------------------------------------------------------------------
# Section 4.2: recursive artifact bundle validator
# ------------------------------------------------------------------

# Evidence levels that may be recorded in an artifact. Synthetic evidence
# must never be presented as real Gate A qualification evidence.
_REAL_EVIDENCE_LEVELS = frozenset({
    "REAL_MODEL_FINAL", "REAL_MODEL_SMOKE", "REAL",
})
_SYNTHETIC_EVIDENCE_LEVELS = frozenset({
    "SYNTHETIC", "SYNTHETIC_CI", "SIMULATED",
})
_QUALIFIED_STATUSES = frozenset({"PASS", "PASSED", "QUALIFIED", "QUALIFIED_PASS"})
_FAILED_STATUSES = frozenset({"FAIL", "FAILED"})


@dataclass(frozen=True)
class ArtifactValidationResult:
    """Section 4.2: result of validating an artifact bundle."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    # Lists are valid JSON artifacts (e.g. experiences) but have no
    # source-hash fields to check; return a sentinel dict so callers
    # don't treat them as unreadable.
    if isinstance(data, list):
        return {"__list_artifact__": True}
    return data if isinstance(data, dict) else None


def _is_legacy_marker(data: dict) -> bool:
    return bool(data.get("legacy") or data.get("not_current_source")
                or data.get("historical_source_hash"))


def validate_artifact_bundle(
    artifact_dir: Path,
    repo_root: Path,
    *,
    require_real_model: bool = False,
    require_final_access_ledger: bool = False,
) -> ArtifactValidationResult:
    """Section 4.2: recursively validate all evidence files in a bundle.

    Validates that:

    * the source hash matches the current repository (unless the bundle is
      explicitly marked as legacy);
    * experiment ID is consistent across all artifacts that declare one;
    * run ID is consistent across all artifacts that declare one;
    * evidence level is consistent;
    * dataset hashes match across artifacts that declare them;
    * model revision matches;
    * tokenizer revision matches;
    * utility configuration hash matches;
    * policy hash matches;
    * split names are valid;
    * a final-access ledger exists when required;
    * no synthetic result is labeled real;
    * no failed run is labeled qualified;
    * no metrics are copied from another run;
    * no manifest contains an obsolete source hash (when not legacy).

    Any inconsistency causes validation to fail (``valid=False``).
    """
    from daph_learning.provenance import compute_canonical_source_hash

    errors: list[str] = []
    warnings: list[str] = []
    bundle = Path(artifact_dir)
    if not bundle.is_dir():
        return ArtifactValidationResult(
            valid=False,
            errors=(f"artifact bundle directory not found: {bundle}",),
            warnings=(),
        )

    current_hash = compute_canonical_source_hash(Path(repo_root))

    # Collect every JSON evidence file in the bundle.
    json_files = sorted(bundle.rglob("*.json"))
    if not json_files:
        errors.append(f"bundle contains no JSON evidence files: {bundle}")

    # Pass 1: detect whether the bundle is explicitly marked as legacy.
    # A legacy bundle retains its historical source hash and is exempt
    # from the current-source-hash requirement across ALL its files.
    bundle_is_legacy = False
    for jf in json_files:
        data = _load_json(jf)
        if data is not None and _is_legacy_marker(data):
            bundle_is_legacy = True
            break

    # Aggregate declared values; any disagreement is an error.
    experiment_ids: set[str] = set()
    run_ids: set[str] = set()
    evidence_levels: set[str] = set()
    dataset_hashes: set[str] = set()
    dataset_hashes_per_split: dict[str, set[str]] = {}
    model_revisions: set[str] = set()
    tokenizer_revisions: set[str] = set()
    utility_hashes: set[str] = set()
    policy_hashes: set[str] = set()
    split_names: set[str] = set()
    statuses: set[str] = set()
    saw_real_evidence = False
    saw_synthetic_evidence = False
    saw_qualified_status = False
    saw_failed_status = False
    source_hashes: dict[str, str] = {}  # path -> declared source hash

    VALID_SPLITS = frozenset({"train", "dev", "development", "calibration", "final"})

    for jf in json_files:
        data = _load_json(jf)
        if data is None:
            errors.append(f"unreadable/invalid JSON: {jf.relative_to(bundle)}")
            continue

        if _is_legacy_marker(data):
            # Already accounted for in pass 1; skip further checks.
            continue

        # Source hash consistency.
        declared_hash = (
            data.get("source_hash")
            or data.get("source_tree_sha256")
            or data.get("source_tree_hash")
        )
        if declared_hash:
            source_hashes[str(jf.relative_to(bundle))] = str(declared_hash)
            # Legacy bundles retain their historical source hash and are
            # exempt from the current-source-hash requirement.
            if not bundle_is_legacy:
                cmp_len = min(len(declared_hash), len(current_hash))
                if declared_hash[:cmp_len] != current_hash[:cmp_len]:
                    errors.append(
                        f"{jf.relative_to(bundle)}: stale source hash "
                        f"{declared_hash[:16]}... != current {current_hash[:16]}..."
                    )

        # Experiment ID.
        eid = data.get("experiment_id")
        if eid:
            experiment_ids.add(str(eid))

        # Run ID.
        rid = data.get("run_id")
        if rid:
            run_ids.add(str(rid))

        # Evidence level.
        ev = data.get("evidence_level")
        if ev:
            evidence_levels.add(str(ev))
            if str(ev) in _REAL_EVIDENCE_LEVELS:
                saw_real_evidence = True
            if str(ev) in _SYNTHETIC_EVIDENCE_LEVELS:
                saw_synthetic_evidence = True

        # Dataset hashes — track per-split to avoid false positives
        # (different splits SHOULD have different hashes).
        for key in ("dataset_hash", "final_dataset_sha256", "train_dataset_sha256",
                    "dev_dataset_sha256", "calibration_dataset_sha256"):
            val = data.get(key)
            if val:
                dataset_hashes_per_split.setdefault(key, set()).add(str(val))

        # Model / tokenizer revisions.
        if data.get("model_revision"):
            model_revisions.add(str(data["model_revision"]))
        if data.get("tokenizer_revision"):
            tokenizer_revisions.add(str(data["tokenizer_revision"]))

        # Utility config hash.
        for key in ("utility_config_hash", "utility_config_sha256", "config_sha256"):
            val = data.get(key)
            if val and key == "utility_config_hash":
                utility_hashes.add(str(val))

        # Policy hash.
        for key in ("policy_hash", "policy_sha256"):
            val = data.get(key)
            if val:
                policy_hashes.add(str(val))

        # Split names.
        sp = data.get("split")
        if sp:
            split_names.add(str(sp))

        # Status / verdict.
        st = data.get("status") or data.get("gate_verdict") or data.get("verdict")
        if st:
            statuses.add(str(st).upper())
            if str(st).upper() in _QUALIFIED_STATUSES:
                saw_qualified_status = True
            if str(st).upper() in _FAILED_STATUSES:
                saw_failed_status = True

    # Cross-artifact consistency checks.
    if len(experiment_ids) > 1:
        errors.append(f"mixed experiment IDs in bundle: {sorted(experiment_ids)}")
    if len(run_ids) > 1:
        errors.append(f"mixed run IDs in bundle: {sorted(run_ids)}")
    if len(model_revisions) > 1:
        errors.append(f"mixed model revisions in bundle: {sorted(model_revisions)}")
    if len(tokenizer_revisions) > 1:
        errors.append(
            f"mixed tokenizer revisions in bundle: {sorted(tokenizer_revisions)}")
    if len(utility_hashes) > 1:
        errors.append(f"mixed utility config hashes in bundle: {sorted(utility_hashes)}")
    if len(policy_hashes) > 1:
        errors.append(f"mixed policy hashes in bundle: {sorted(policy_hashes)}")
    if len(dataset_hashes) > 1:
        errors.append(f"mixed dataset hashes in bundle: {sorted(dataset_hashes)}")
    # Per-split: each split key should have exactly one hash across
    # all artifacts in the bundle.
    for split_key, hashes in dataset_hashes_per_split.items():
        if len(hashes) > 1:
            errors.append(
                f"mixed dataset hashes for {split_key}: {sorted(hashes)}")

    bad_splits = split_names - VALID_SPLITS
    if bad_splits:
        errors.append(f"invalid split names in bundle: {sorted(bad_splits)}")

    # Synthetic evidence must never be labeled real or qualified.
    # These fraud checks apply to CURRENT bundles only; legacy bundles
    # are preserved as-is for audit history and mixing is recorded as a
    # warning rather than a hard failure.
    if saw_synthetic_evidence and saw_qualified_status:
        msg = ("synthetic evidence is labeled qualified — synthetic artifacts "
               "cannot pass Gate A")
        (errors if not bundle_is_legacy else warnings).append(msg)
    if saw_synthetic_evidence and require_real_model and not bundle_is_legacy:
        errors.append(
            "bundle contains synthetic evidence but real-model evidence was "
            "required")

    # A failed run must never be labeled qualified.
    if saw_failed_status and saw_qualified_status:
        msg = ("bundle contains both FAILED and QUALIFIED statuses — a failed "
               "run cannot be promoted")
        (errors if not bundle_is_legacy else warnings).append(msg)

    # Real/synthetic evidence-level mixing is a hard error for current
    # bundles and a warning for legacy bundles.
    if saw_real_evidence and saw_synthetic_evidence:
        msg = ("bundle mixes real and synthetic evidence levels: "
               f"{sorted(evidence_levels)}")
        (errors if not bundle_is_legacy else warnings).append(msg)

    # Final-access ledger requirement.
    if require_final_access_ledger:
        ledger = bundle / "final_access_ledger.json"
        if not ledger.is_file():
            errors.append(
                f"required final-access ledger missing: {ledger.relative_to(bundle)}")
        else:
            ledger_data = _load_json(ledger) or {}
            if ledger_data.get("final_access_count") not in (1, "1"):
                errors.append(
                    "final_access_ledger.json does not record exactly one "
                    "final access")
            ledger_hash = (
                ledger_data.get("source_hash")
                or ledger_data.get("source_tree_sha256")
            )
            if ledger_hash:
                cmp_len = min(len(ledger_hash), len(current_hash))
                if ledger_hash[:cmp_len] != current_hash[:cmp_len] and not bundle_is_legacy:
                    errors.append(
                        "final_access_ledger.json source hash does not match "
                        "current source")

    # Mixed source hashes (metrics copied from another run) — only an error
    # for non-legacy bundles.
    if not bundle_is_legacy and source_hashes:
        distinct = set(source_hashes.values())
        if len(distinct) > 1:
            errors.append(
                f"bundle contains metrics from multiple source hashes "
                f"(possible cross-run copy): {sorted(h[:16] for h in distinct)}")

    if bundle_is_legacy:
        warnings.append(
            "bundle is marked legacy and is exempt from the current "
            "source-hash requirement; it must not be presented as current "
            "Gate A evidence")

    return ArtifactValidationResult(
        valid=(len(errors) == 0),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
