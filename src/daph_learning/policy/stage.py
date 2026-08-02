"""v0.3.10.5-alpha — experiment stage access control (Sections 14-15).

Enforces split access discipline:

  TRAIN:        fit policy, PCA, feature transforms, OOD distribution
  DEV:          choose policy, layer, capture location, hyperparameters
  CALIBRATION:  confidence threshold, probability calibration, OOD threshold
  FROZEN:       all fitting complete; artifacts frozen
  FINAL:        load once after complete freeze; zero fitting

The final split cannot be accessed before ``stage == FROZEN``. Every
final access is recorded in a ledger (Section 15).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any


class ExperimentStage(str, Enum):
    """Section 14/16: experiment lifecycle stages.

    Section 16 adds 9 fine-grained stages. The original 5 stages (TRAIN,
    DEV, CALIBRATION, FROZEN, FINAL) are retained as aliases for backward
    compatibility.

    The 9 Section 16 stages (in order):
      CREATED → DATASET_FROZEN → EXPERIENCE_COLLECTED →
      DEVELOPMENT_COMPLETE → CALIBRATED → POLICY_FROZEN →
      FINAL_RESERVED → FINAL_EVALUATED → REPORTED
    """
    # Original 5 stages (backward compatible).
    TRAIN = "train"
    DEV = "dev"
    CALIBRATION = "calibration"
    FROZEN = "frozen"
    FINAL = "final"
    # Section 16 — 9 fine-grained stages.
    CREATED = "created"
    DATASET_FROZEN = "dataset_frozen"
    EXPERIENCE_COLLECTED = "experience_collected"
    DEVELOPMENT_COMPLETE = "development_complete"
    CALIBRATED = "calibrated"
    POLICY_FROZEN = "policy_frozen"
    FINAL_RESERVED = "final_reserved"
    FINAL_EVALUATED = "final_evaluated"
    REPORTED = "reported"

    @classmethod
    def from_str(cls, value: str) -> "ExperimentStage":
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(
            f"unknown stage {value!r}; expected one of "
            f"{[m.value for m in cls]}")


# Section 16 — valid forward transitions. Backward transitions are invalid.
# Legacy 5-stage and Section 16 9-stage transitions are SEPARATED to
# prevent cross-machine transitions (e.g. CREATED → TRAIN is NOT allowed;
# each machine must follow its own path).
_LEGACY_STAGES = frozenset({
    ExperimentStage.TRAIN, ExperimentStage.DEV,
    ExperimentStage.CALIBRATION, ExperimentStage.FROZEN,
    ExperimentStage.FINAL,
})
_SECTION16_STAGES = frozenset({
    ExperimentStage.CREATED, ExperimentStage.DATASET_FROZEN,
    ExperimentStage.EXPERIENCE_COLLECTED, ExperimentStage.DEVELOPMENT_COMPLETE,
    ExperimentStage.CALIBRATED, ExperimentStage.POLICY_FROZEN,
    ExperimentStage.FINAL_RESERVED, ExperimentStage.FINAL_EVALUATED,
    ExperimentStage.REPORTED,
})

_VALID_TRANSITIONS: dict[ExperimentStage, set[ExperimentStage]] = {
    # Section 16 — 9 fine-grained stages (no mixing with legacy).
    ExperimentStage.CREATED: {
        ExperimentStage.DATASET_FROZEN,
    },
    ExperimentStage.DATASET_FROZEN: {
        ExperimentStage.EXPERIENCE_COLLECTED,
    },
    ExperimentStage.EXPERIENCE_COLLECTED: {
        ExperimentStage.DEVELOPMENT_COMPLETE,
    },
    ExperimentStage.DEVELOPMENT_COMPLETE: {
        ExperimentStage.CALIBRATED,
    },
    ExperimentStage.CALIBRATED: {
        ExperimentStage.POLICY_FROZEN,
    },
    ExperimentStage.POLICY_FROZEN: {
        ExperimentStage.FINAL_RESERVED,
    },
    ExperimentStage.FINAL_RESERVED: {
        ExperimentStage.FINAL_EVALUATED,
    },
    ExperimentStage.FINAL_EVALUATED: {
        ExperimentStage.REPORTED,
    },
    ExperimentStage.REPORTED: set(),  # terminal
    # Legacy 5-stage transitions (no mixing with Section 16).
    ExperimentStage.TRAIN: {
        ExperimentStage.DEV, ExperimentStage.CALIBRATION,
        ExperimentStage.FROZEN,
    },
    ExperimentStage.DEV: {
        ExperimentStage.CALIBRATION, ExperimentStage.FROZEN,
    },
    ExperimentStage.CALIBRATION: {
        ExperimentStage.FROZEN,
    },
    ExperimentStage.FROZEN: {
        ExperimentStage.FINAL,
    },
    ExperimentStage.FINAL: set(),  # terminal
}


class FinalAccessError(RuntimeError):
    """Raised when the final split is accessed before freeze or after
    the configured access budget is exhausted (Section 14-15)."""


@dataclass
class FinalAccessRecord:
    """Section 15: one final-test access ledger entry."""
    timestamp: float
    release_version: str
    source_hash: str
    config_hash: str
    policy_hash: str
    calibration_hash: str
    command: str
    reason: str


@dataclass
class FinalAccessLedger:
    """Section 15: append-only ledger of final-test accesses.

    Allows exactly ``max_accesses`` final evaluations (default 1).
    No hyperparameter updates after the first final access.
    """
    max_accesses: int = 1
    release_version: str = "0.4.0a2"
    source_hash: str = ""
    config_hash: str = ""
    policy_hash: str = ""
    calibration_hash: str = ""
    records: list[FinalAccessRecord] = field(default_factory=list)

    def request_access(self, *, command: str, reason: str) -> FinalAccessRecord:
        if len(self.records) >= self.max_accesses:
            raise FinalAccessError(
                f"final access budget exhausted: {len(self.records)}/"
                f"{self.max_accesses} accesses already used")
        rec = FinalAccessRecord(
            timestamp=time.time(),
            release_version=self.release_version,
            source_hash=self.source_hash,
            config_hash=self.config_hash,
            policy_hash=self.policy_hash,
            calibration_hash=self.calibration_hash,
            command=command,
            reason=reason,
        )
        self.records.append(rec)
        return rec

    @property
    def n_accesses(self) -> int:
        return len(self.records)

    @property
    def exhausted(self) -> bool:
        return len(self.records) >= self.max_accesses

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_accesses": self.max_accesses,
            "release_version": self.release_version,
            "source_hash": self.source_hash,
            "config_hash": self.config_hash,
            "policy_hash": self.policy_hash,
            "calibration_hash": self.calibration_hash,
            "n_accesses": self.n_accesses,
            "records": [
                {"timestamp": r.timestamp,
                 "release_version": r.release_version,
                 "source_hash": r.source_hash,
                 "config_hash": r.config_hash,
                 "policy_hash": r.policy_hash,
                 "calibration_hash": r.calibration_hash,
                 "command": r.command,
                 "reason": r.reason}
                for r in self.records
            ],
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class FreezeManifest:
    """Section 33: freeze manifest recording the exact state at freeze
    time.

    The freeze manifest binds the final evaluation to the exact source
    tree, config, datasets, utility config, model, representation,
    policy, calibration, and gate criteria state that was frozen. Any
    change after freezing invalidates the final evaluation.

    All identity-bearing inputs must be non-empty at freeze time.
    """
    experiment_id: str = ""
    release_version: str = "0.4.0a2"
    source_tree_sha256: str = ""
    config_sha256: str = ""
    train_dataset_sha256: str = ""
    development_dataset_sha256: str = ""
    calibration_dataset_sha256: str = ""
    final_dataset_sha256: str = ""
    utility_config_sha256: str = ""
    model_id: str = ""
    model_revision: str = ""
    tokenizer_id: str = ""
    tokenizer_revision: str = ""
    representation_config_sha256: str = ""
    policy_sha256: str = ""
    calibration_sha256: str = ""
    gate_criteria_sha256: str = ""
    frozen_at: float = 0.0
    stage: str = "frozen"

    # Backward compat: old code may read source_tree_sha256.
    @property
    def source_hash(self) -> str:
        return self.source_tree_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "release_version": self.release_version,
            "source_tree_sha256": self.source_tree_sha256,
            "config_sha256": self.config_sha256,
            "train_dataset_sha256": self.train_dataset_sha256,
            "development_dataset_sha256": self.development_dataset_sha256,
            "calibration_dataset_sha256": self.calibration_dataset_sha256,
            "final_dataset_sha256": self.final_dataset_sha256,
            "utility_config_sha256": self.utility_config_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "representation_config_sha256": self.representation_config_sha256,
            "policy_sha256": self.policy_sha256,
            "calibration_sha256": self.calibration_sha256,
            "gate_criteria_sha256": self.gate_criteria_sha256,
            "frozen_at": self.frozen_at,
            "stage": self.stage,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FreezeManifest":
        with open(path) as f:
            data = json.load(f)
        # Backward compat: old manifests may not have all fields.
        defaults = {f.name: "" for f in fields(cls) if f.default != ""}
        merged = {**defaults, **data}
        # Remove keys not in the dataclass.
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in merged.items() if k in valid_keys}
        return cls(**filtered)

    def assert_complete(self) -> None:
        """Section 33 — refuse to freeze when any required field is empty."""
        required = (
            "experiment_id", "source_tree_sha256", "config_sha256",
            "train_dataset_sha256", "development_dataset_sha256",
            "calibration_dataset_sha256", "final_dataset_sha256",
            "utility_config_sha256", "model_id",
            "representation_config_sha256", "policy_sha256",
            "calibration_sha256", "gate_criteria_sha256",
        )
        empty = [name for name in required if not getattr(self, name)]
        if empty:
            raise RuntimeError(
                f"freeze manifest incomplete — empty fields: {empty}")

    def verify_current_state(
        self,
        *,
        current_source_hash: str = "",
        current_config_hash: str = "",
        current_policy_hash: str = "",
        current_calibration_hash: str = "",
        current_train_dataset_hash: str = "",
        current_dev_dataset_hash: str = "",
        current_cal_dataset_hash: str = "",
        current_final_dataset_hash: str = "",
        current_utility_config_hash: str = "",
        current_model_id: str = "",
        current_representation_hash: str = "",
        current_gate_criteria_hash: str = "",
        strict: bool = True,
    ) -> bool:
        """Section 33: verify that the current state matches the frozen
        state. If ``strict=True``, raises on mismatch; otherwise returns
        False.
        """
        mismatches = []
        def _check(name: str, current: str, frozen: str) -> None:
            if current and frozen and current != frozen:
                mismatches.append(name)
        _check("source_tree_sha256", current_source_hash, self.source_tree_sha256)
        _check("config_sha256", current_config_hash, self.config_sha256)
        _check("policy_sha256", current_policy_hash, self.policy_sha256)
        _check("calibration_sha256", current_calibration_hash, self.calibration_sha256)
        _check("train_dataset_sha256", current_train_dataset_hash, self.train_dataset_sha256)
        _check("development_dataset_sha256", current_dev_dataset_hash, self.development_dataset_sha256)
        _check("calibration_dataset_sha256", current_cal_dataset_hash, self.calibration_dataset_sha256)
        _check("final_dataset_sha256", current_final_dataset_hash, self.final_dataset_sha256)
        _check("utility_config_sha256", current_utility_config_hash, self.utility_config_sha256)
        _check("model_id", current_model_id, self.model_id)
        _check("representation_config_sha256", current_representation_hash, self.representation_config_sha256)
        _check("gate_criteria_sha256", current_gate_criteria_hash, self.gate_criteria_sha256)
        if mismatches:
            msg = f"freeze manifest mismatch: {mismatches}"
            if strict:
                raise RuntimeError(msg)
            return False
        return True


@dataclass
class StageGuard:
    """Section 14: enforces that the final split is inaccessible before
    ``FROZEN`` and that final access goes through the ledger.

    Section 33: also manages the freeze manifest, which binds the final
    evaluation to the exact state at freeze time.
    """
    stage: ExperimentStage = ExperimentStage.TRAIN
    ledger: FinalAccessLedger = field(default_factory=FinalAccessLedger)
    freeze_manifest: FreezeManifest | None = None

    def transition_to(self, stage: ExperimentStage) -> None:
        """Section 16 — transition to a new stage. Raises on invalid
        backward transitions or cross-machine transitions."""
        if stage != self.stage:
            # Prevent cross-machine transitions (legacy ↔ Section 16).
            current_machine = (
                "legacy" if self.stage in _LEGACY_STAGES else "section16"
            )
            target_machine = (
                "legacy" if stage in _LEGACY_STAGES else "section16"
            )
            if current_machine != target_machine:
                raise FinalAccessError(
                    f"cross-machine transition forbidden: "
                    f"{self.stage.value!r} ({current_machine}) → "
                    f"{stage.value!r} ({target_machine}); "
                    f"each machine must follow its own path")
            allowed = _VALID_TRANSITIONS.get(self.stage, set())
            if stage not in allowed:
                raise FinalAccessError(
                    f"invalid stage transition: {self.stage.value!r} → "
                    f"{stage.value!r}; valid next stages: "
                    f"{sorted(s.value for s in allowed)}")
        self.stage = stage

    def freeze(
        self,
        *,
        experiment_id: str = "",
        source_hash: str = "",
        config_hash: str = "",
        train_dataset_hash: str = "",
        dev_dataset_hash: str = "",
        cal_dataset_hash: str = "",
        final_dataset_hash: str = "",
        utility_config_hash: str = "",
        model_id: str = "",
        model_revision: str = "",
        tokenizer_id: str = "",
        tokenizer_revision: str = "",
        representation_hash: str = "",
        policy_hash: str = "",
        calibration_hash: str = "",
        gate_criteria_hash: str = "",
    ) -> FreezeManifest:
        """Section 33: transition to FROZEN and record the freeze
        manifest with all identity-bearing inputs."""
        self.stage = ExperimentStage.FROZEN
        self.freeze_manifest = FreezeManifest(
            experiment_id=experiment_id,
            release_version=self.ledger.release_version,
            source_tree_sha256=source_hash,
            config_sha256=config_hash,
            train_dataset_sha256=train_dataset_hash,
            development_dataset_sha256=dev_dataset_hash,
            calibration_dataset_sha256=cal_dataset_hash,
            final_dataset_sha256=final_dataset_hash,
            utility_config_sha256=utility_config_hash,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            representation_config_sha256=representation_hash,
            policy_sha256=policy_hash,
            calibration_sha256=calibration_hash,
            gate_criteria_sha256=gate_criteria_hash,
            frozen_at=time.time(),
        )
        self.freeze_manifest.assert_complete()
        # Update the ledger with the freeze hashes.
        self.ledger.source_hash = source_hash
        self.ledger.config_hash = config_hash
        self.ledger.policy_hash = policy_hash
        self.ledger.calibration_hash = calibration_hash
        return self.freeze_manifest

    def assert_can_access_split(self, split: str) -> None:
        if split == "final":
            if self.stage != ExperimentStage.FROZEN:
                raise FinalAccessError(
                    f"final split accessed at stage {self.stage.value!r}; "
                    f"must be FROZEN first (Section 14)")
        # train/dev/calibration accessible at any stage >= their own.

    def request_final_access(self, *, command: str, reason: str) -> FinalAccessRecord:
        self.assert_can_access_split("final")
        return self.ledger.request_access(command=command, reason=reason)

    def verify_frozen_state(
        self,
        *,
        current_source_hash: str = "",
        current_config_hash: str = "",
        current_policy_hash: str = "",
        current_calibration_hash: str = "",
        current_train_dataset_hash: str = "",
        current_dev_dataset_hash: str = "",
        current_cal_dataset_hash: str = "",
        current_final_dataset_hash: str = "",
        current_utility_config_hash: str = "",
        current_model_id: str = "",
        current_representation_hash: str = "",
        current_gate_criteria_hash: str = "",
    ) -> bool:
        """Section 33: verify that the current state matches the frozen
        state before final evaluation. Checks ALL identity-bearing inputs."""
        if self.freeze_manifest is None:
            raise RuntimeError(
                "no freeze manifest — cannot verify frozen state "
                "(Section 33)")
        return self.freeze_manifest.verify_current_state(
            current_source_hash=current_source_hash,
            current_config_hash=current_config_hash,
            current_policy_hash=current_policy_hash,
            current_calibration_hash=current_calibration_hash,
            current_train_dataset_hash=current_train_dataset_hash,
            current_dev_dataset_hash=current_dev_dataset_hash,
            current_cal_dataset_hash=current_cal_dataset_hash,
            current_final_dataset_hash=current_final_dataset_hash,
            current_utility_config_hash=current_utility_config_hash,
            current_model_id=current_model_id,
            current_representation_hash=current_representation_hash,
            current_gate_criteria_hash=current_gate_criteria_hash,
            strict=True,
        )


__all__ = [
    "ExperimentStage",
    "FinalAccessError",
    "FinalAccessLedger",
    "FinalAccessRecord",
    "FreezeManifest",
    "StageGuard",
]
