"""DAPH v0.4 — Frozen experiment lifecycle management.

Implements explicit experiment lifecycle states::

    DEVELOPMENT
        ↓
    FROZEN
        ↓
    FINAL_RUNNING
        ↓
    QUALIFIED / FAILED

Once the experiment is FROZEN, hash:
* action definitions
* prompts
* dataset generator
* task families
* model revision
* representation choices allowed
* policy class
* utility weights
* bootstrap settings
* qualification thresholds

Final evaluation must reject execution if these differ from the frozen config.

Never tune after observing FINAL.
If modifications are required, create a new experiment ID.

Experiment statuses (Section 43):
    DEVELOPMENT
    FROZEN
    RUNNING
    FAILED_EXECUTION
    FAILED_INTEGRITY
    FAILED_QUALIFICATION
    QUALIFIED
    INVALIDATED
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from daph_learning.executive.artifact_integrity import sha256_json, is_zero_hash
from daph_learning.executive.manifest import compute_config_hash


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Experiment status
# ──────────────────────────────────────────────────────────────────────

class ExperimentStatus(str, Enum):
    """Explicit experiment lifecycle states."""

    DEVELOPMENT = "DEVELOPMENT"
    FROZEN = "FROZEN"
    RUNNING = "RUNNING"
    FAILED_EXECUTION = "FAILED_EXECUTION"
    FAILED_INTEGRITY = "FAILED_INTEGRITY"
    FAILED_QUALIFICATION = "FAILED_QUALIFICATION"
    QUALIFIED = "QUALIFIED"
    INVALIDATED = "INVALIDATED"

    @classmethod
    def valid_transitions(cls, current: "ExperimentStatus") -> set["ExperimentStatus"]:
        """Return the set of statuses that can be transitioned to from current."""
        transitions: dict[ExperimentStatus, set[ExperimentStatus]] = {
            cls.DEVELOPMENT: {cls.FROZEN, cls.INVALIDATED},
            cls.FROZEN: {cls.RUNNING, cls.INVALIDATED},
            cls.RUNNING: {
                cls.QUALIFIED,
                cls.FAILED_EXECUTION,
                cls.FAILED_INTEGRITY,
                cls.FAILED_QUALIFICATION,
                cls.INVALIDATED,
            },
            cls.FAILED_EXECUTION: {cls.INVALIDATED},
            cls.FAILED_INTEGRITY: {cls.INVALIDATED},
            cls.FAILED_QUALIFICATION: {cls.INVALIDATED},
            cls.QUALIFIED: {cls.INVALIDATED},
            cls.INVALIDATED: set(),
        }
        return transitions.get(current, set())

    def can_transition_to(self, target: "ExperimentStatus") -> bool:
        return target in self.valid_transitions(self)


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Frozen config
# ──────────────────────────────────────────────────────────────────────

# Fields that must be frozen before final evaluation.
FROZEN_CONFIG_FIELDS: list[str] = [
    "experiment_id",
    "action_space",
    "prompts",
    "dataset_generator",
    "task_families",
    "model",
    "representation",
    "policy_class",
    "utility_weights",
    "bootstrap_settings",
    "qualification_thresholds",
]


@dataclass
class FrozenConfig:
    """A frozen experiment configuration with its hash.

    Once frozen, the config hash is recorded. Any subsequent change
    to the frozen fields invalidates the experiment.
    """

    config: dict[str, Any]
    config_hash: str = ""
    frozen_at: str = ""
    frozen_fields: list[str] = field(default_factory=lambda: list(FROZEN_CONFIG_FIELDS))

    def __post_init__(self) -> None:
        if not self.config_hash:
            self.config_hash = _partial_config_hash(self.config, self.frozen_fields)
        if not self.frozen_at:
            self.frozen_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def matches(self, other_config: Mapping[str, Any]) -> bool:
        """Check if another config matches this frozen config on frozen fields."""
        other_hash = _partial_config_hash(other_config, self.frozen_fields)
        return other_hash == self.config_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": dict(self.config),
            "config_hash": self.config_hash,
            "frozen_at": self.frozen_at,
            "frozen_fields": list(self.frozen_fields),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrozenConfig":
        return cls(
            config=dict(data.get("config", {})),
            config_hash=data.get("config_hash", ""),
            frozen_at=data.get("frozen_at", ""),
            frozen_fields=list(data.get("frozen_fields", FROZEN_CONFIG_FIELDS)),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FrozenConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _partial_config_hash(
    config: Mapping[str, Any],
    fields: Sequence[str],
) -> str:
    """Hash only the specified fields of a config."""
    partial = {}
    for f in fields:
        if f in config:
            partial[f] = config[f]
    return compute_config_hash(partial)


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Experiment state file
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentState:
    """Persistent experiment state tracking.

    Stored as ``status.json`` in the artifact root.
    """

    experiment_id: str
    status: ExperimentStatus = ExperimentStatus.DEVELOPMENT
    frozen_config: FrozenConfig | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if not self.updated_at:
            self.updated_at = self.created_at

    def _record_transition(self, old: ExperimentStatus, new: ExperimentStatus) -> None:
        self.history.append({
            "from": old.value,
            "to": new.value,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })

    def transition_to(self, target: ExperimentStatus) -> None:
        """Transition to a new status, validating the transition is allowed."""
        if not self.status.can_transition_to(target):
            raise ValueError(
                f"invalid status transition: {self.status.value} → {target.value}"
            )
        old = self.status
        self.status = target
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._record_transition(old, target)

    def freeze(self, config: Mapping[str, Any]) -> None:
        """Freeze the experiment with the given config."""
        if self.status != ExperimentStatus.DEVELOPMENT:
            raise ValueError(
                f"can only freeze from DEVELOPMENT, current={self.status.value}"
            )
        self.frozen_config = FrozenConfig(config=dict(config))
        self.transition_to(ExperimentStatus.FROZEN)

    def start_final(self, config: Mapping[str, Any] | None = None) -> None:
        """Start final evaluation. If config is provided, verify it matches frozen."""
        if self.status != ExperimentStatus.FROZEN:
            raise ValueError(
                f"can only start final from FROZEN, current={self.status.value}"
            )
        if config is not None and self.frozen_config is not None:
            if not self.frozen_config.matches(config):
                raise ValueError(
                    "config does not match frozen config — create a new experiment ID"
                )
        self.transition_to(ExperimentStatus.RUNNING)

    def mark_qualified(self) -> None:
        self.transition_to(ExperimentStatus.QUALIFIED)

    def mark_failed(self, reason: str = "qualification") -> None:
        if reason == "execution":
            self.transition_to(ExperimentStatus.FAILED_EXECUTION)
        elif reason == "integrity":
            self.transition_to(ExperimentStatus.FAILED_INTEGRITY)
        else:
            self.transition_to(ExperimentStatus.FAILED_QUALIFICATION)

    def invalidate(self, reason: str = "") -> None:
        """Mark the experiment as invalidated."""
        # INVALIDATED can be reached from any non-terminal state
        if self.status == ExperimentStatus.INVALIDATED:
            return
        if not self.status.can_transition_to(ExperimentStatus.INVALIDATED):
            # Force transition for already-failed experiments
            self.status = ExperimentStatus.INVALIDATED
        else:
            self.transition_to(ExperimentStatus.INVALIDATED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "frozen_config": self.frozen_config.to_dict() if self.frozen_config else None,
            "history": list(self.history),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentState":
        with open(path) as f:
            data = json.load(f)
        state = cls(
            experiment_id=data["experiment_id"],
            status=ExperimentStatus(data.get("status", "DEVELOPMENT")),
            frozen_config=FrozenConfig.from_dict(data["frozen_config"])
            if data.get("frozen_config") else None,
            history=list(data.get("history", [])),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
        return state

    @classmethod
    def from_artifact_root(cls, artifact_root: str | Path) -> "ExperimentState":
        """Load or create experiment state from an artifact root."""
        p = Path(artifact_root) / "status.json"
        if p.exists():
            return cls.load(p)
        # Infer experiment_id from directory name
        eid = Path(artifact_root).name
        return cls(experiment_id=eid)


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Experiment registry
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RegistryEntry:
    """A single experiment in the experiment registry."""

    experiment_id: str
    version: str
    status: str
    artifact_root: str
    qualification_summary: dict[str, Any] = field(default_factory=dict)
    invalidated_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "version": self.version,
            "status": self.status,
            "artifact_root": self.artifact_root,
            "qualification_summary": dict(self.qualification_summary),
            "invalidated_reason": self.invalidated_reason,
        }


def load_registry(registry_path: str | Path) -> dict[str, Any]:
    """Load the experiment registry."""
    p = Path(registry_path)
    if not p.exists():
        return {"experiments": []}
    with open(p) as f:
        return json.load(f)


def save_registry(registry: dict[str, Any], registry_path: str | Path) -> None:
    """Save the experiment registry."""
    p = Path(registry_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(registry, f, indent=2)


def register_experiment(
    registry_path: str | Path,
    entry: RegistryEntry,
) -> None:
    """Add or update an experiment in the registry."""
    registry = load_registry(registry_path)
    experiments = registry.get("experiments", [])
    # Replace if exists
    experiments = [e for e in experiments if e.get("experiment_id") != entry.experiment_id]
    experiments.append(entry.to_dict())
    registry["experiments"] = experiments
    save_registry(registry, registry_path)


def invalidate_experiment(
    registry_path: str | Path,
    experiment_id: str,
    reason: str,
) -> None:
    """Mark an experiment as invalidated in the registry."""
    registry = load_registry(registry_path)
    for exp in registry.get("experiments", []):
        if exp.get("experiment_id") == experiment_id:
            exp["status"] = "INVALIDATED"
            exp["invalidated_reason"] = reason
    save_registry(registry, registry_path)


__all__ = [
    "ExperimentStatus",
    "FROZEN_CONFIG_FIELDS",
    "FrozenConfig",
    "ExperimentState",
    "RegistryEntry",
    "load_registry",
    "save_registry",
    "register_experiment",
    "invalidate_experiment",
]
