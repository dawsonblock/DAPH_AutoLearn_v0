"""v0.3.10.3.2-alpha — experiment stage access control (Sections 14-15).

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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExperimentStage(str, Enum):
    """Section 14: experiment lifecycle stages."""
    TRAIN = "train"
    DEV = "dev"
    CALIBRATION = "calibration"
    FROZEN = "frozen"
    FINAL = "final"

    @classmethod
    def from_str(cls, value: str) -> "ExperimentStage":
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(
            f"unknown stage {value!r}; expected one of "
            f"{[m.value for m in cls]}")


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
    release_version: str = "0.3.10.3.2-alpha"
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
    tree, config, policy, and calibration state that was frozen. Any
    change after freezing invalidates the final evaluation.
    """
    release_version: str = "0.3.10.3.2-alpha"
    source_tree_sha256: str = ""
    config_sha256: str = ""
    policy_sha256: str = ""
    calibration_sha256: str = ""
    frozen_at: float = 0.0
    stage: str = "frozen"

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_version": self.release_version,
            "source_tree_sha256": self.source_tree_sha256,
            "config_sha256": self.config_sha256,
            "policy_sha256": self.policy_sha256,
            "calibration_sha256": self.calibration_sha256,
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
        return cls(**data)

    def verify_current_state(
        self,
        *,
        current_source_hash: str = "",
        current_config_hash: str = "",
        current_policy_hash: str = "",
        current_calibration_hash: str = "",
        strict: bool = True,
    ) -> bool:
        """Section 33: verify that the current state matches the frozen
        state. If ``strict=True``, raises on mismatch; otherwise returns
        False.
        """
        mismatches = []
        if current_source_hash and self.source_tree_sha256:
            if current_source_hash[:64] != self.source_tree_sha256[:64]:
                mismatches.append("source_tree_sha256")
        if current_config_hash and self.config_sha256:
            if current_config_hash != self.config_sha256:
                mismatches.append("config_sha256")
        if current_policy_hash and self.policy_sha256:
            if current_policy_hash != self.policy_sha256:
                mismatches.append("policy_sha256")
        if current_calibration_hash and self.calibration_sha256:
            if current_calibration_hash != self.calibration_sha256:
                mismatches.append("calibration_sha256")
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
        self.stage = stage

    def freeze(
        self,
        *,
        source_hash: str = "",
        config_hash: str = "",
        policy_hash: str = "",
        calibration_hash: str = "",
    ) -> FreezeManifest:
        """Section 33: transition to FROZEN and record the freeze
        manifest."""
        self.stage = ExperimentStage.FROZEN
        self.freeze_manifest = FreezeManifest(
            release_version=self.ledger.release_version,
            source_tree_sha256=source_hash,
            config_sha256=config_hash,
            policy_sha256=policy_hash,
            calibration_sha256=calibration_hash,
            frozen_at=time.time(),
        )
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
    ) -> bool:
        """Section 33: verify that the current state matches the frozen
        state before final evaluation."""
        if self.freeze_manifest is None:
            raise RuntimeError(
                "no freeze manifest — cannot verify frozen state "
                "(Section 33)")
        return self.freeze_manifest.verify_current_state(
            current_source_hash=current_source_hash,
            current_config_hash=current_config_hash,
            current_policy_hash=current_policy_hash,
            current_calibration_hash=current_calibration_hash,
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
