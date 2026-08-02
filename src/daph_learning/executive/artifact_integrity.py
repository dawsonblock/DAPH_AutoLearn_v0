"""DAPH v0.4 — Artifact integrity validation.

A staged experiment MUST fail if any required machine-readable artifact:

* is absent,
* is empty,
* cannot be parsed,
* contains shell/SSH error output,
* contains placeholder values,
* contains zeroed hashes,
* references missing upstream artifacts,
* or fails its declared schema.

Markdown reports are never the source of truth.
Machine-readable artifacts are authoritative.

This module provides the validator that enforces these rules.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Known corruption signatures
# ──────────────────────────────────────────────────────────────────────

# Shell/SSH error strings that indicate a tool failure was captured
# as artifact content instead of real data.
_CORRUPTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Your SSH client doesn't support PTY", re.IGNORECASE),
    re.compile(r"PTY allocation request failed", re.IGNORECASE),
    re.compile(r"^Error:\s.*PTY", re.IGNORECASE | re.MULTILINE),
    re.compile(r"bash:.*command not found", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"Permission denied \(publickey\)", re.IGNORECASE),
    re.compile(r"Connection (refused|closed|reset)", re.IGNORECASE),
    re.compile(r"No such file or directory", re.IGNORECASE),
]

# Placeholder values that indicate an artifact was not actually populated.
_PLACEHOLDER_VALUES = {
    "",
    "unknown",
    "unknown",
    "placeholder",
    "tbd",
    "todo",
    "n/a",
    "null",
    "none",
    "0000000000000000000000000000000000000000000000000000000000000000",
}

# Zeroed SHA-256 hash
_ZERO_HASH = "0" * 64


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Validation result types
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ArtifactCheck:
    """Result of checking a single artifact."""

    path: str
    check_type: str  # "exists", "non_empty", "json_parse", "no_corruption", "schema", "hash"
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "check_type": self.check_type,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class IntegrityReport:
    """Full integrity validation report for an experiment artifact tree."""

    experiment_id: str
    artifact_root: str
    checks: list[ArtifactCheck] = field(default_factory=list)
    passed: bool = True
    failures: list[str] = field(default_factory=list)

    def add(self, check: ArtifactCheck) -> None:
        self.checks.append(check)
        if not check.passed:
            self.passed = False
            self.failures.append(
                f"{check.check_type}: {check.path} — {check.detail}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "artifact_root": self.artifact_root,
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failures": len(self.failures),
            "failures": list(self.failures),
            "checks": [c.to_dict() for c in self.checks],
        }


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Hashing utilities
# ──────────────────────────────────────────────────────────────────────

def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    """Compute a deterministic SHA-256 of a JSON-serializable object."""
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def is_zero_hash(h: str) -> bool:
    """Check if a hash is all zeros or otherwise invalid."""
    if not h:
        return True
    cleaned = h.strip().lower()
    if cleaned == _ZERO_HASH:
        return True
    if len(cleaned) != 64:
        return True
    if not all(c in "0123456789abcdef" for c in cleaned):
        return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Content corruption detection
# ──────────────────────────────────────────────────────────────────────

def detect_corruption(text: str) -> str | None:
    """Check if text content contains known corruption signatures.

    Returns the matched pattern description if corruption is detected,
    or ``None`` if the content appears clean.
    """
    for pattern in _CORRUPTION_PATTERNS:
        m = pattern.search(text)
        if m:
            return f"corruption pattern matched: {pattern.pattern!r} at pos {m.start()}"
    return None


def is_placeholder(value: Any) -> bool:
    """Check if a string value is a placeholder."""
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _PLACEHOLDER_VALUES


# ──────────────────────────────────────────────────────────────────────
# Section 5 — JSON artifact validation
# ──────────────────────────────────────────────────────────────────────

def validate_json_artifact(
    path: str | Path,
    *,
    required_fields: Sequence[str] | None = None,
    allow_empty: bool = False,
) -> ArtifactCheck:
    """Validate a JSON artifact file.

    Checks:
    1. File exists and is non-empty.
    2. Content parses as valid JSON.
    3. Content does not contain corruption signatures.
    4. Required fields are present and non-placeholder.
    """
    p = Path(path)
    sp = str(path)

    # 1. Exists
    if not p.exists():
        return ArtifactCheck(sp, "exists", False, "file does not exist")

    # 2. Non-empty
    raw = p.read_bytes()
    if len(raw) == 0:
        return ArtifactCheck(sp, "non_empty", False, "file is empty (0 bytes)")

    # 3. Corruption check on raw text
    text = raw.decode("utf-8", errors="replace")
    corruption = detect_corruption(text)
    if corruption:
        return ArtifactCheck(sp, "no_corruption", False, corruption)

    # 4. JSON parse
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return ArtifactCheck(sp, "json_parse", False, f"invalid JSON: {e}")

    # 5. Non-empty object (unless allowed)
    if not allow_empty:
        if isinstance(data, (dict, list)) and len(data) == 0:
            return ArtifactCheck(sp, "non_empty", False, "JSON is empty container")

    # 6. Required fields
    if required_fields and isinstance(data, dict):
        for field_name in required_fields:
            if field_name not in data:
                return ArtifactCheck(
                    sp, "schema", False,
                    f"missing required field: {field_name!r}",
                )
            val = data[field_name]
            if is_placeholder(val):
                return ArtifactCheck(
                    sp, "schema", False,
                    f"field {field_name!r} is placeholder: {val!r}",
                )

    return ArtifactCheck(sp, "json_parse", True, "valid JSON artifact")


def validate_hash_reference(
    path: str | Path,
    expected_sha256: str,
) -> ArtifactCheck:
    """Validate that a file's SHA-256 matches an expected value."""
    p = Path(path)
    sp = str(path)

    if is_zero_hash(expected_sha256):
        return ArtifactCheck(sp, "hash", False, "expected hash is zero/invalid")

    if not p.exists():
        return ArtifactCheck(sp, "hash", False, "file does not exist for hash check")

    actual = sha256_file(p)
    if actual != expected_sha256:
        return ArtifactCheck(
            sp, "hash", False,
            f"hash mismatch: expected {expected_sha256[:16]}..., got {actual[:16]}...",
        )
    return ArtifactCheck(sp, "hash", True, "hash matches")


# ──────────────────────────────────────────────────────────────────────
# Section 6 — Required artifact tree validation
# ──────────────────────────────────────────────────────────────────────

# The canonical B4 artifact tree (relative to artifact root).
B4_REQUIRED_ARTIFACTS: dict[str, list[str]] = {
    "root": ["manifest.json"],
    "config": ["experiment_config.json", "config_hash.txt"],
    "provenance": ["source.json", "environment.json", "model.json", "dependencies.txt"],
    "dataset": ["train.json", "dev.json", "final.json", "groups.json", "dataset_manifest.json"],
    "counterfactuals": ["train.json", "dev.json", "final.json", "summary.json"],
    "representations": ["train.npz", "dev.npz", "final.npz", "representation_manifest.json"],
    "selection": ["representation_sweep.json", "selected_representation.json"],
    "pca": ["pca_artifact.npz", "pca_manifest.json"],
    "policies": ["hidden_policy.json", "fixed_baselines.json", "policy_manifest.json"],
    "sham": ["sham_runs.json", "sham_per_task.json", "sham_manifest.json"],
    "qualification": ["qualification.json", "per_task_results.json", "per_group_results.json"],
    "checks": ["artifact_integrity.json", "leakage_checks.json", "reproducibility_check.json"],
}

# B5 adds OOD and abstention artifacts.
B5_REQUIRED_ARTIFACTS: dict[str, list[str]] = {
    **B4_REQUIRED_ARTIFACTS,
    "dataset": ["train.json", "dev.json", "final.json", "final_ood.json",
                "groups.json", "dataset_manifest.json", "crossover_report.json"],
    "qualification": ["qualification.json", "per_task_results.json",
                      "per_group_results.json", "ood_results.json",
                      "bootstrap_results.npz"],
    "policies": ["hidden_policy.json", "ridge_policy.json", "mlp_policy.json",
                 "fixed_baselines.json", "surface_baselines.json",
                 "policy_manifest.json"],
}


def validate_required_tree(
    artifact_root: str | Path,
    required: Mapping[str, list[str]],
    *,
    experiment_id: str = "",
) -> IntegrityReport:
    """Validate that all required artifacts exist in the artifact tree.

    Parameters
    ----------
    artifact_root : path
        Root directory of the experiment artifacts.
    required : mapping
        Mapping of subdirectory → list of required filenames.
    experiment_id : str
        Experiment identifier for the report.
    """
    root = Path(artifact_root)
    report = IntegrityReport(
        experiment_id=experiment_id,
        artifact_root=str(root),
    )

    for subdir, files in required.items():
        for fname in files:
            if subdir == "root":
                p = root / fname
            else:
                p = root / subdir / fname
            check = validate_json_artifact(p) if fname.endswith(".json") else \
                ArtifactCheck(str(p), "exists", p.exists(),
                              "ok" if p.exists() else "missing")
            if not p.exists():
                check = ArtifactCheck(str(p), "exists", False, "required artifact missing")
            elif fname.endswith(".json"):
                check = validate_json_artifact(p)
            elif fname.endswith(".npz"):
                # Check it's a valid numpy archive
                try:
                    import numpy as np
                    with np.load(p, allow_pickle=False) as _:
                        pass
                    check = ArtifactCheck(str(p), "npz_parse", True, "valid NPZ")
                except Exception as e:
                    check = ArtifactCheck(str(p), "npz_parse", False, f"invalid NPZ: {e}")
            else:
                raw = p.read_bytes()
                corruption = detect_corruption(raw.decode("utf-8", errors="replace"))
                if corruption:
                    check = ArtifactCheck(str(p), "no_corruption", False, corruption)
                elif len(raw) == 0:
                    check = ArtifactCheck(str(p), "non_empty", False, "file is empty")
                else:
                    check = ArtifactCheck(str(p), "exists", True, "ok")
            report.add(check)

    return report


# ──────────────────────────────────────────────────────────────────────
# Section 7 — Manifest cross-validation
# ──────────────────────────────────────────────────────────────────────

def validate_manifest_hashes(
    manifest_path: str | Path,
    artifact_root: str | Path,
) -> IntegrityReport:
    """Validate that all artifacts listed in a manifest exist and match hashes."""
    mp = Path(manifest_path)
    root = Path(artifact_root)
    report = IntegrityReport(
        experiment_id="manifest_validation",
        artifact_root=str(root),
    )

    # The manifest itself must be valid
    check = validate_json_artifact(mp, required_fields=["schema_version"])
    report.add(check)
    if not check.passed:
        return report

    with open(mp) as f:
        manifest = json.load(f)

    # Walk the manifest sections looking for artifact entries with paths+hashes
    def _walk(obj: Any, prefix: str = ""):
        if isinstance(obj, dict):
            if "path" in obj and "sha256" in obj:
                art_path = root / obj["path"]
                report.add(validate_hash_reference(art_path, obj["sha256"]))
            for k, v in obj.items():
                _walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{prefix}[{i}]")

    _walk(manifest)
    return report


__all__ = [
    "ArtifactCheck",
    "IntegrityReport",
    "sha256_file",
    "sha256_bytes",
    "sha256_json",
    "is_zero_hash",
    "detect_corruption",
    "is_placeholder",
    "validate_json_artifact",
    "validate_hash_reference",
    "validate_required_tree",
    "validate_manifest_hashes",
    "B4_REQUIRED_ARTIFACTS",
    "B5_REQUIRED_ARTIFACTS",
]
