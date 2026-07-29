"""v0.3.10.3.1-alpha — experiment stage access control (Sections 14-15).

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
    release_version: str = "0.3.10.3.1-alpha"
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
class StageGuard:
    """Section 14: enforces that the final split is inaccessible before
    ``FROZEN`` and that final access goes through the ledger.
    """
    stage: ExperimentStage = ExperimentStage.TRAIN
    ledger: FinalAccessLedger = field(default_factory=FinalAccessLedger)

    def transition_to(self, stage: ExperimentStage) -> None:
        self.stage = stage

    def freeze(self) -> None:
        self.stage = ExperimentStage.FROZEN

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


__all__ = [
    "ExperimentStage",
    "FinalAccessError",
    "FinalAccessLedger",
    "FinalAccessRecord",
    "StageGuard",
]
