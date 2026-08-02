"""DAPH v0.4.0a3 — Final access guard and ledger.

Enforces that FINAL/FINAL_OOD data is only read during FINAL_RUNNING
or later states. Every access is logged to a persistent ledger.

This is the central mechanism for scientific phase isolation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daph_learning.executive.atomic_io import append_jsonl, atomic_write_json
from daph_learning.executive.lifecycle import ExperimentState, FinalAccessViolation


# Splits that are FINAL-sensitive
FINAL_SPLITS = {"final", "final_ood"}


@dataclass
class FinalAccessRecord:
    """A single final access ledger entry."""

    timestamp: str
    experiment_id: str
    stage: str
    split: str
    artifact: str
    purpose: str
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "split": self.split,
            "artifact": self.artifact,
            "purpose": self.purpose,
            "config_hash": self.config_hash,
        }


class FinalAccessGuard:
    """Central guard for FINAL split data access.

    Any code requesting FINAL-sensitive data must go through this guard.

    Usage::

        guard = FinalAccessGuard(state, ledger_path)
        guard.assert_can_read("final", artifact="counterfactuals/final.json",
                               purpose="primary final evaluation")
    """

    def __init__(
        self,
        state: ExperimentState,
        ledger_path: str | Path,
    ) -> None:
        self.state = state
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[FinalAccessRecord] = []

    def assert_can_read(
        self,
        split: str,
        *,
        artifact: str = "",
        purpose: str = "",
        stage: str = "",
    ) -> None:
        """Assert that the current state allows reading the given split.

        Raises FinalAccessViolation if the split is FINAL-sensitive and
        the experiment is not in FINAL_RUNNING or later.
        """
        if split not in FINAL_SPLITS:
            return  # Non-final splits are always accessible

        if not self.state.status.is_final_accessible:
            raise FinalAccessViolation(
                stage=stage or self.state.current_stage or "unknown",
                split=split,
                message=(
                    f"stage={stage or self.state.current_stage or 'unknown'} "
                    f"attempted to read split={split} "
                    f"(current state={self.state.status.value})"
                ),
            )

        # Record the access
        record = FinalAccessRecord(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            experiment_id=self.state.experiment_id,
            stage=stage or self.state.current_stage or "unknown",
            split=split,
            artifact=artifact,
            purpose=purpose,
            config_hash=self.state.frozen_config.config_hash if self.state.frozen_config else "",
        )
        self._records.append(record)
        append_jsonl(self.ledger_path, record.to_dict())

    def assert_can_read_final(self, artifact: str = "", purpose: str = "", stage: str = "") -> None:
        """Convenience method for reading the final split."""
        self.assert_can_read("final", artifact=artifact, purpose=purpose, stage=stage)

    def assert_can_read_ood(self, artifact: str = "", purpose: str = "", stage: str = "") -> None:
        """Convenience method for reading the final_ood split."""
        self.assert_can_read("final_ood", artifact=artifact, purpose=purpose, stage=stage)

    @property
    def records(self) -> list[FinalAccessRecord]:
        """All access records from this guard session."""
        return list(self._records)

    def get_ledger(self) -> list[dict]:
        """Load the full ledger from disk."""
        if not self.ledger_path.exists():
            return []
        records = []
        with open(self.ledger_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def verify_no_unauthorized_access(self) -> bool:
        """Verify that no FINAL access occurred before FINAL_RUNNING.

        Scans the ledger for any access records where the stage
        indicates a pre-final phase.
        """
        pre_final_stages = {
            "prepare", "counterfactuals", "development-counterfactuals",
            "representations", "development-representations",
            "train", "freeze-policy",
        }
        ledger = self.get_ledger()
        for record in ledger:
            if record.get("split") in FINAL_SPLITS:
                if record.get("stage", "") in pre_final_stages:
                    return False
        return True


def load_final_access_ledger(ledger_path: str | Path) -> list[dict]:
    """Load a final access ledger from disk."""
    p = Path(ledger_path)
    if not p.exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def check_final_isolation(ledger_path: str | Path) -> dict[str, Any]:
    """Check final isolation from a ledger file.

    Returns a dict with:
    - passed: bool
    - violations: list of violation records
    - n_accesses: total number of FINAL accesses
    """
    ledger = load_final_access_ledger(ledger_path)
    pre_final_stages = {
        "prepare", "counterfactuals", "development-counterfactuals",
        "representations", "development-representations",
        "train", "freeze-policy",
    }
    violations = []
    for record in ledger:
        if record.get("split") in FINAL_SPLITS:
            if record.get("stage", "") in pre_final_stages:
                violations.append(record)

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "n_accesses": len([r for r in ledger if r.get("split") in FINAL_SPLITS]),
    }


__all__ = [
    "FinalAccessRecord",
    "FinalAccessGuard",
    "FINAL_SPLITS",
    "load_final_access_ledger",
    "check_final_isolation",
]
