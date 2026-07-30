"""v0.3.10.1-alpha — policy artifact provenance (Section 45).

Every policy artifact must contain full provenance so recorded results
are reproducible and traceable. No hard-coded lineage.

Provenance fields (Section 45):
- release version
- git/source tree hash if available
- model ID
- model revision
- tokenizer revision
- dataset hashes
- split hashes
- policy type
- target mode
- target temperature
- weight mode
- gap threshold
- confidence formulation
- feature transform
- PCA artifact hash
- OOD model hash
- selected layer
- steering alpha
- random seed
- train timestamp
- environment metadata
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProvenanceRecord:
    """Full provenance for a policy artifact (Section 45)."""

    release_version: str = "0.3.10.3.2-alpha"
    git_hash: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    train_dataset_sha256: str | None = None
    dev_dataset_sha256: str | None = None
    calibration_dataset_sha256: str | None = None
    final_dataset_sha256: str | None = None
    policy_type: str = "logistic"
    target_mode: str = "soft"
    target_temperature: float = 1.0
    weight_mode: str = "clipped_gap"
    gap_threshold: float = 0.0
    confidence_threshold: float = 0.70
    feature_transform: str = "raw"
    pca_components: int = 32
    selected_layer: int = 20
    steering_alpha: float = 1.0
    random_seed: int = 0
    train_timestamp: float = field(default_factory=time.time)
    environment: dict[str, str] = field(default_factory=dict)
    config_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Filter None values for clean JSON.
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_config(cls, config, **overrides) -> "ProvenanceRecord":
        """Build a ProvenanceRecord from an ExperimentConfig + overrides.

        Overrides take precedence over config values (e.g. computed
        dataset hashes override the config's None defaults).
        """
        env = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
        # Start with config values; let overrides replace them.
        kwargs = dict(
            release_version=config.autolearn_version,
            model_revision=config.model_revision,
            tokenizer_revision=config.tokenizer_revision,
            train_dataset_sha256=config.train_dataset_sha256,
            dev_dataset_sha256=config.dev_dataset_sha256,
            calibration_dataset_sha256=config.calibration_dataset_sha256,
            final_dataset_sha256=config.final_dataset_sha256,
            policy_type=config.policy_type,
            target_mode=config.target_mode,
            target_temperature=config.target_temperature,
            weight_mode=config.weight_mode,
            gap_threshold=config.gap_threshold,
            confidence_threshold=config.confidence_threshold,
            feature_transform=config.feature_transform,
            pca_components=config.pca_components,
            selected_layer=config.selected_layer,
            steering_alpha=config.steering_alpha,
            random_seed=config.random_seed,
            environment=env,
            config_sha256=config.config_sha256,
        )
        kwargs.update(overrides)
        return cls(**kwargs)


def git_tree_hash(repo_root: str | Path = ".") -> str | None:
    """Get the current git tree hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_hash(tasks: list[dict]) -> str:
    """Compute a deterministic hash of a dataset (list of task dicts)."""
    payload = json.dumps(tasks, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_provenance(config, *, model_id: str | None = None,
                     repo_root: str | Path = ".",
                     train_tasks: list[dict] | None = None,
                     dev_tasks: list[dict] | None = None,
                     calibration_tasks: list[dict] | None = None,
                     final_tasks: list[dict] | None = None) -> ProvenanceRecord:
    """Build a complete ProvenanceRecord with git hash and dataset hashes."""
    overrides = {}
    if model_id is not None:
        overrides["model_id"] = model_id
    gh = git_tree_hash(repo_root)
    if gh is not None:
        overrides["git_hash"] = gh
    # Compute dataset hashes if tasks are provided.
    if train_tasks is not None:
        overrides["train_dataset_sha256"] = dataset_hash(train_tasks)
    if dev_tasks is not None:
        overrides["dev_dataset_sha256"] = dataset_hash(dev_tasks)
    if calibration_tasks is not None:
        overrides["calibration_dataset_sha256"] = dataset_hash(calibration_tasks)
    if final_tasks is not None:
        overrides["final_dataset_sha256"] = dataset_hash(final_tasks)
    return ProvenanceRecord.from_config(config, **overrides)


def source_tree_sha256(repo_root: str | Path = ".") -> str:
    """Deterministic SHA-256 hash over relevant source files (Section 2).

    .. deprecated::
        This function now delegates to the canonical
        :func:`daph_learning.provenance.compute_source_tree_sha256`.
        All new code should call the canonical function directly.

    Returns the **full 64-character** SHA-256 hex digest (Section 2:
    artifacts must store the full hash, not a truncation).
    """
    from daph_learning.provenance import compute_source_tree_sha256
    repo = Path(repo_root)
    if not repo.is_dir():
        return "unknown"
    return compute_source_tree_sha256(repo)


__all__ = [
    "ProvenanceRecord",
    "build_provenance",
    "dataset_hash",
    "file_sha256",
    "git_tree_hash",
    "source_tree_sha256",
]
