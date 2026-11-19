"""v0.3.10.1-alpha — atomic policy promotion (Section 46).

Promotion flow (Section 46):

1. write candidate artifact
2. evaluate candidate against incumbent on all gates
3. if all gates pass: atomically update the incumbent pointer
4. if any gate fails: incumbent remains unchanged (no partially-written
   promoted state)

The atomic update uses a temp-file + rename pattern so a crash during
the pointer update cannot leave a partially-written incumbent.

This module also provides :func:`rollback_incumbent` to restore a
prior incumbent from the lineage (Section 46 / G23).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..evaluation.capability_gate import GateResult


@dataclass
class PromotionResult:
    """Result of an atomic promotion attempt (Section 46)."""

    promoted: bool
    incumbent_path: str
    candidate_path: str
    gate_results: list[GateResult] = field(default_factory=list)
    reason: str = ""

    @property
    def all_gates_passed(self) -> bool:
        return all(g.passed for g in self.gate_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "incumbent_path": self.incumbent_path,
            "candidate_path": self.candidate_path,
            "all_gates_passed": self.all_gates_passed,
            "reason": self.reason,
            "gate_results": [g.to_dict() for g in self.gate_results],
        }


def atomic_write_json(path: str | Path, data: dict) -> None:
    """Atomically write JSON to ``path`` via temp-file + rename.

    The temp file is created in the same directory as ``path`` so the
    rename is atomic on POSIX. A crash during the write leaves the
    original file intact.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=str(p.parent), suffix=".tmp",
        prefix=p.stem + "_", delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2, default=str)
        tmp_path = tmp.name
    # Atomic rename (POSIX). On Windows, os.replace is atomic.
    os.replace(tmp_path, str(p))


def atomic_promote(
    candidate_artifact: dict,
    candidate_path: str | Path,
    incumbent_path: str | Path,
    *,
    gates: Sequence[GateResult],
) -> PromotionResult:
    """Atomically promote a candidate to incumbent (Section 46).

    Flow:
    1. Write the candidate artifact to ``candidate_path`` (always, for
       auditability).
    2. Check all gates. If any fail, return without touching the
       incumbent.
    3. If all gates pass, atomically write the candidate artifact to
       ``incumbent_path`` via temp-file + rename.

    Parameters
    ----------
    candidate_artifact : dict
        The candidate policy artifact (with provenance).
    candidate_path : str | Path
        Where to write the candidate artifact (always written).
    incumbent_path : str | Path
        Where the incumbent artifact lives. Atomically replaced on
        promotion.
    gates : sequence of GateResult
        All gates that must pass for promotion.

    Returns
    -------
    PromotionResult
    """
    # 1. Always write the candidate artifact (for auditability).
    atomic_write_json(candidate_path, candidate_artifact)
    gate_list = list(gates)
    # 2. Check all gates.
    if not all(g.passed for g in gate_list):
        failed = [g for g in gate_list if not g.passed]
        reasons = "; ".join(f"{g.reason}" for g in failed)
        return PromotionResult(
            promoted=False,
            incumbent_path=str(incumbent_path),
            candidate_path=str(candidate_path),
            gate_results=gate_list,
            reason=f"gate(s) failed: {reasons}",
        )
    # 3. All gates pass: atomically update the incumbent pointer.
    atomic_write_json(incumbent_path, candidate_artifact)
    return PromotionResult(
        promoted=True,
        incumbent_path=str(incumbent_path),
        candidate_path=str(candidate_path),
        gate_results=gate_list,
        reason="all gates passed; incumbent atomically updated",
    )


def rollback_incumbent(
    incumbent_path: str | Path,
    prior_artifact: dict,
) -> None:
    """Restore a prior incumbent artifact (Section 46 / G23).

    Atomically writes ``prior_artifact`` to ``incumbent_path``. Use this
    to roll back to a previous incumbent from the lineage if a promoted
    candidate is later found to be defective.
    """
    atomic_write_json(incumbent_path, prior_artifact)


def read_incumbent(incumbent_path: str | Path) -> dict:
    """Read the current incumbent artifact."""
    with open(incumbent_path) as f:
        return json.load(f)


__all__ = [
    "PromotionResult",
    "atomic_promote",
    "atomic_write_json",
    "read_incumbent",
    "rollback_incumbent",
]
