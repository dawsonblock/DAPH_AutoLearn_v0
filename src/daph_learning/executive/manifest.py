"""DAPH v0.4 — Versioned experiment artifact manifest.

Every important artifact must include:
* relative path
* SHA-256
* byte size
* schema version where applicable

The manifest is the single source of truth for what artifacts exist,
what their hashes are, and how they relate to the experiment config.

Schema version 1.0.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from daph_learning.executive.artifact_integrity import (
    sha256_file,
    sha256_json,
    is_zero_hash,
    detect_corruption,
)

MANIFEST_SCHEMA_VERSION = "1.0"


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Artifact entry
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ArtifactEntry:
    """A single artifact entry in the manifest."""

    path: str
    sha256: str
    bytes: int
    schema_version: str | None = None
    records: int | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }
        if self.schema_version is not None:
            d["schema_version"] = self.schema_version
        if self.records is not None:
            d["records"] = self.records
        if self.description:
            d["description"] = self.description
        return d


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Manifest builder
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ManifestBuilder:
    """Builds a versioned experiment manifest.

    Usage::

        builder = ManifestBuilder(
            experiment_id="executive_b4_reproduction_v1",
            experiment_family="executive_b4_hidden_state",
            artifact_root=Path("artifacts/executive_b4_reproduction_v1"),
        )
        builder.add_file("dataset/train.json", schema_version="1.0")
        builder.add_file("counterfactuals/train.json")
        builder.set_model(provider="huggingface", model_id="Qwen/Qwen3-8B", ...)
        builder.set_config_hash(config_hash)
        builder.write()
    """

    experiment_id: str
    experiment_family: str
    artifact_root: Path
    created_at: str = ""
    source_commit: str = ""
    source_tree_hash: str = ""
    config_hash: str = ""

    model: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    counterfactuals: dict[str, Any] = field(default_factory=dict)
    representations: dict[str, Any] = field(default_factory=dict)
    pca: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    policies: dict[str, Any] = field(default_factory=dict)
    shams: dict[str, Any] = field(default_factory=dict)
    qualification: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    _entries: dict[str, ArtifactEntry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if not self.source_commit:
            self.source_commit = _git_commit()
        if not self.source_tree_hash:
            self.source_tree_hash = _git_tree_hash()

    def add_file(
        self,
        rel_path: str,
        *,
        schema_version: str | None = None,
        records: int | None = None,
        description: str = "",
        section: str | None = None,
    ) -> ArtifactEntry:
        """Add a file to the manifest, computing its hash."""
        full_path = self.artifact_root / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"artifact not found: {full_path}")

        h = sha256_file(full_path)
        size = full_path.stat().st_size

        # Detect corruption in text files
        if rel_path.endswith((".json", ".txt", ".md")):
            text = full_path.read_text(errors="replace")
            corruption = detect_corruption(text)
            if corruption:
                raise ValueError(f"corruption detected in {rel_path}: {corruption}")

        entry = ArtifactEntry(
            path=rel_path,
            sha256=h,
            bytes=size,
            schema_version=schema_version,
            records=records,
            description=description,
        )
        self._entries[rel_path] = entry

        if section:
            sect = getattr(self, section, None)
            if isinstance(sect, dict):
                # Use the filename (without extension) as the key
                key = Path(rel_path).stem
                sect[key] = entry.to_dict()

        return entry

    def add_npz(
        self,
        rel_path: str,
        *,
        description: str = "",
        section: str | None = None,
    ) -> ArtifactEntry:
        """Add an NPZ file, recording array shapes."""
        full_path = self.artifact_root / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"artifact not found: {full_path}")

        import numpy as np
        h = sha256_file(full_path)
        size = full_path.stat().st_size

        shapes: dict[str, list[int]] = {}
        try:
            with np.load(full_path, allow_pickle=False) as data:
                for key in data.files:
                    shapes[key] = list(data[key].shape)
        except (OSError, ValueError, KeyError):
            pass

        entry = ArtifactEntry(
            path=rel_path,
            sha256=h,
            bytes=size,
            description=description or f"NPZ with arrays: {list(shapes.keys())}",
        )
        self._entries[rel_path] = entry

        if section:
            sect = getattr(self, section, None)
            if isinstance(sect, dict):
                key = Path(rel_path).stem
                sect[key] = {**entry.to_dict(), "arrays": shapes}

        return entry

    def set_model(
        self,
        *,
        provider: str,
        model_id: str,
        revision: str = "",
        tokenizer_revision: str = "",
        dtype: str = "",
        chat_template_hash: str = "",
    ) -> None:
        self.model = {
            "provider": provider,
            "model_id": model_id,
            "revision": revision,
            "tokenizer_revision": tokenizer_revision,
            "dtype": dtype,
            "chat_template_hash": chat_template_hash,
        }

    def set_environment(self, env: dict[str, Any]) -> None:
        self.environment = env

    def set_config_hash(self, config_hash: str) -> None:
        self.config_hash = config_hash

    def build(self) -> dict[str, Any]:
        """Build the manifest dictionary."""
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "experiment_family": self.experiment_family,
            "created_at": self.created_at,
            "source_commit": self.source_commit,
            "source_tree_hash": self.source_tree_hash,
            "config_hash": self.config_hash,
            "model": dict(self.model),
            "dataset": dict(self.dataset),
            "counterfactuals": dict(self.counterfactuals),
            "representations": dict(self.representations),
            "pca": dict(self.pca),
            "selection": dict(self.selection),
            "policies": dict(self.policies),
            "shams": dict(self.shams),
            "qualification": dict(self.qualification),
            "environment": dict(self.environment),
        }

    def write(self, path: str | Path | None = None) -> Path:
        """Write the manifest to disk."""
        manifest = self.build()
        out_path = Path(path) if path else self.artifact_root / "manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)
        return out_path

    def config_hash_value(self) -> str:
        """Return the config hash, or empty string if not set."""
        return self.config_hash


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Config hashing
# ──────────────────────────────────────────────────────────────────────

def compute_config_hash(config: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of an experiment config.

    The config is serialized with sorted keys to ensure determinism.
    """
    return sha256_json(dict(config))


def write_config_hash(config: Mapping[str, Any], path: str | Path) -> str:
    """Write the config hash to a file and return it."""
    h = compute_config_hash(config)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(h + "\n")
    return h


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Git provenance helpers
# ──────────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    """Get the current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _git_tree_hash() -> str:
    """Get the git tree hash of HEAD, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def capture_environment() -> dict[str, Any]:
    """Capture environment information for provenance."""
    import platform
    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    # Key library versions
    try:
        import numpy as np
        env["numpy_version"] = np.__version__
    except ImportError:
        pass
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import transformers
        env["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    return env


# ──────────────────────────────────────────────────────────────────────
# Section 5 — Manifest loading and validation
# ──────────────────────────────────────────────────────────────────────

def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest from JSON."""
    with open(path) as f:
        return json.load(f)


def manifest_config_hash_matches(
    manifest_path: str | Path,
    expected_hash: str,
) -> bool:
    """Check if the manifest's config_hash matches the expected value."""
    manifest = load_manifest(manifest_path)
    actual = manifest.get("config_hash", "")
    if is_zero_hash(actual) or is_zero_hash(expected_hash):
        return False
    return actual == expected_hash


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ArtifactEntry",
    "ManifestBuilder",
    "compute_config_hash",
    "write_config_hash",
    "capture_environment",
    "load_manifest",
    "manifest_config_hash_matches",
]
