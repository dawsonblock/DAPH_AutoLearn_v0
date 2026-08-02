"""DAPH v0.4 — B5 qualification gates.

Freeze gates before final execution. Recommended primary qualification:

    Gate 1 — Beats best fixed:   LCB95(U_hidden - U_bestfixed) > 0
    Gate 2 — Beats surface:      LCB95(U_hidden - U_subtype) > 0
    Gate 3 — Beats sham:         LCB95(U_hidden - U_sham) > 0
    Gate 4 — Positive capture:   GapCapture > 0 (prefer ≥ 25%)
    Gate 5 — Breadth:            positive_group_fraction ≥ 65%, no catastrophic collapse
    Gate 6 — No leakage:         all leakage checks pass
    Gate 7 — Artifact integrity: all artifacts valid and reproducible

Failure of any hard gate means NOT QUALIFIED, not "mostly passed."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from daph_learning.executive.stats import (
    BootstrapResult,
    ShamComparisonResult,
    GroupResult,
    gap_capture,
    positive_group_fraction,
    worst_group_delta,
)


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Gate result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """Result of a single qualification gate."""

    gate_name: str
    passed: bool
    value: float | bool
    threshold: float | str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass
class QualificationResult:
    """Full B5 qualification result with all gates."""

    experiment_id: str
    gates: list[GateResult] = field(default_factory=list)
    overall_passed: bool = False
    failed_gates: list[str] = field(default_factory=list)

    # Primary metrics
    hidden_utility: float = 0.0
    best_fixed_utility: float = 0.0
    oracle_utility: float = 0.0
    gap_capture: float = 0.0
    selection_accuracy: float = 0.0
    positive_group_fraction: float = 0.0
    worst_group_delta: float = 0.0

    # Compute metrics
    avg_latency_ms: float = 0.0
    avg_generated_tokens: float = 0.0
    avg_llm_calls: float = 0.0
    compute_cost: float = 0.0

    # Bootstrap results
    bootstrap_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Sham results
    sham_result: dict[str, Any] = field(default_factory=dict)

    def add_gate(self, gate: GateResult) -> None:
        self.gates.append(gate)
        if not gate.passed:
            self.failed_gates.append(gate.gate_name)

    def evaluate_overall(self) -> None:
        """Evaluate whether all gates pass."""
        self.overall_passed = len(self.failed_gates) == 0 and len(self.gates) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "overall_passed": self.overall_passed,
            "n_gates": len(self.gates),
            "n_passed": len(self.gates) - len(self.failed_gates),
            "n_failed": len(self.failed_gates),
            "failed_gates": list(self.failed_gates),
            "gates": [g.to_dict() for g in self.gates],
            "primary_metrics": {
                "hidden_utility": self.hidden_utility,
                "best_fixed_utility": self.best_fixed_utility,
                "oracle_utility": self.oracle_utility,
                "gap_capture": self.gap_capture,
                "selection_accuracy": self.selection_accuracy,
                "positive_group_fraction": self.positive_group_fraction,
                "worst_group_delta": self.worst_group_delta,
            },
            "compute_metrics": {
                "avg_latency_ms": self.avg_latency_ms,
                "avg_generated_tokens": self.avg_generated_tokens,
                "avg_llm_calls": self.avg_llm_calls,
                "compute_cost": self.compute_cost,
            },
            "bootstrap_results": dict(self.bootstrap_results),
            "sham_result": dict(self.sham_result),
        }


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Frozen gate thresholds
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GateThresholds:
    """Frozen qualification gate thresholds.

    These must be frozen before final evaluation.
    """

    # Gate 1: LCB95(hidden - bestfixed) > 0
    min_lcb_vs_bestfixed: float = 0.0
    # Gate 2: LCB95(hidden - surface) > 0
    min_lcb_vs_surface: float = 0.0
    # Gate 3: LCB95(hidden - sham) > 0
    min_lcb_vs_sham: float = 0.0
    # Gate 4: GapCapture ≥ threshold
    min_gap_capture: float = 0.25
    # Gate 5: positive_group_fraction ≥ threshold
    min_positive_group_fraction: float = 0.65
    # Gate 5b: no catastrophic family collapse (worst_group_delta ≥ threshold)
    min_worst_group_delta: float = -0.20
    # Gate 6: all leakage checks pass (boolean)
    # Gate 7: all artifacts valid (boolean)

    def to_dict(self) -> dict[str, float]:
        return {
            "min_lcb_vs_bestfixed": self.min_lcb_vs_bestfixed,
            "min_lcb_vs_surface": self.min_lcb_vs_surface,
            "min_lcb_vs_sham": self.min_lcb_vs_sham,
            "min_gap_capture": self.min_gap_capture,
            "min_positive_group_fraction": self.min_positive_group_fraction,
            "min_worst_group_delta": self.min_worst_group_delta,
        }


DEFAULT_THRESHOLDS = GateThresholds()


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Gate evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_gates(
    *,
    experiment_id: str,
    # Bootstrap results for paired comparisons
    boot_hidden_vs_bestfixed: BootstrapResult,
    boot_hidden_vs_surface: BootstrapResult,
    # Sham comparison
    sham_result: ShamComparisonResult,
    # Group-local results
    group_results: list[GroupResult],
    # Primary metrics
    hidden_utility: float,
    best_fixed_utility: float,
    oracle_utility: float,
    selection_accuracy: float,
    # Leakage and integrity
    leakage_passed: bool,
    integrity_passed: bool,
    # Compute metrics
    avg_latency_ms: float = 0.0,
    avg_generated_tokens: float = 0.0,
    avg_llm_calls: float = 0.0,
    compute_cost: float = 0.0,
    # Thresholds
    thresholds: GateThresholds = DEFAULT_THRESHOLDS,
) -> QualificationResult:
    """Evaluate all seven B5 qualification gates.

    Parameters
    ----------
    boot_hidden_vs_bestfixed : BootstrapResult
        Paired group bootstrap of hidden vs best fixed policy.
    boot_hidden_vs_surface : BootstrapResult
        Paired group bootstrap of hidden vs strongest surface baseline.
    sham_result : ShamComparisonResult
        Matched sham evaluation result.
    group_results : list of GroupResult
        Per-group paired comparisons (hidden vs baseline).
    leakage_passed : bool
        Whether all leakage checks passed.
    integrity_passed : bool
        Whether all artifact integrity checks passed.
    thresholds : GateThresholds
        Frozen gate thresholds.
    """
    result = QualificationResult(experiment_id=experiment_id)
    result.hidden_utility = hidden_utility
    result.best_fixed_utility = best_fixed_utility
    result.oracle_utility = oracle_utility
    result.selection_accuracy = selection_accuracy
    result.avg_latency_ms = avg_latency_ms
    result.avg_generated_tokens = avg_generated_tokens
    result.avg_llm_calls = avg_llm_calls
    result.compute_cost = compute_cost

    # Gap capture
    gc = gap_capture(hidden_utility, best_fixed_utility, oracle_utility)
    result.gap_capture = gc

    # Positive group fraction
    pgf = positive_group_fraction(group_results)
    result.positive_group_fraction = pgf

    # Worst group delta
    wgd = worst_group_delta(group_results)
    result.worst_group_delta = wgd

    # Store bootstrap and sham results
    result.bootstrap_results = {
        "hidden_vs_bestfixed": boot_hidden_vs_bestfixed.to_dict(),
        "hidden_vs_surface": boot_hidden_vs_surface.to_dict(),
    }
    result.sham_result = sham_result.to_dict()

    # Gate 1: Beats best fixed
    g1 = GateResult(
        gate_name="gate_1_beats_best_fixed",
        passed=boot_hidden_vs_bestfixed.lcb_95 > thresholds.min_lcb_vs_bestfixed,
        value=boot_hidden_vs_bestfixed.lcb_95,
        threshold=f"LCB95 > {thresholds.min_lcb_vs_bestfixed}",
        detail=f"LCB95={boot_hidden_vs_bestfixed.lcb_95:.4f}, "
               f"P(Δ>0)={boot_hidden_vs_bestfixed.prob_positive:.3f}",
    )
    result.add_gate(g1)

    # Gate 2: Beats surface policy
    g2 = GateResult(
        gate_name="gate_2_beats_surface",
        passed=boot_hidden_vs_surface.lcb_95 > thresholds.min_lcb_vs_surface,
        value=boot_hidden_vs_surface.lcb_95,
        threshold=f"LCB95 > {thresholds.min_lcb_vs_surface}",
        detail=f"LCB95={boot_hidden_vs_surface.lcb_95:.4f}, "
               f"P(Δ>0)={boot_hidden_vs_surface.prob_positive:.3f}",
    )
    result.add_gate(g2)

    # Gate 3: Beats sham
    g3 = GateResult(
        gate_name="gate_3_beats_sham",
        passed=sham_result.hidden_vs_sham_lcb95 > thresholds.min_lcb_vs_sham,
        value=sham_result.hidden_vs_sham_lcb95,
        threshold=f"LCB95 > {thresholds.min_lcb_vs_sham}",
        detail=f"LCB95={sham_result.hidden_vs_sham_lcb95:.4f}, "
               f"P(hidden>sham)={sham_result.prob_hidden_gt_sham:.3f}",
    )
    result.add_gate(g3)

    # Gate 4: Positive oracle capture
    g4 = GateResult(
        gate_name="gate_4_positive_capture",
        passed=gc >= thresholds.min_gap_capture,
        value=gc,
        threshold=f"GapCapture ≥ {thresholds.min_gap_capture}",
        detail=f"GapCapture={gc:.4f}",
    )
    result.add_gate(g4)

    # Gate 5: Breadth (positive group fraction + no catastrophic collapse)
    g5 = GateResult(
        gate_name="gate_5_breadth",
        passed=(pgf >= thresholds.min_positive_group_fraction and
                wgd >= thresholds.min_worst_group_delta),
        value=pgf,
        threshold=f"pos_group_frac ≥ {thresholds.min_positive_group_fraction} "
                  f"AND worst_group_delta ≥ {thresholds.min_worst_group_delta}",
        detail=f"pos_group_frac={pgf:.3f}, worst_group_delta={wgd:.4f}",
    )
    result.add_gate(g5)

    # Gate 6: No leakage
    g6 = GateResult(
        gate_name="gate_6_no_leakage",
        passed=leakage_passed,
        value=leakage_passed,
        threshold="all leakage checks pass",
        detail="passed" if leakage_passed else "FAILED — leakage detected",
    )
    result.add_gate(g6)

    # Gate 7: Artifact integrity
    g7 = GateResult(
        gate_name="gate_7_artifact_integrity",
        passed=integrity_passed,
        value=integrity_passed,
        threshold="all artifacts valid and reproducible",
        detail="passed" if integrity_passed else "FAILED — artifact integrity violation",
    )
    result.add_gate(g7)

    result.evaluate_overall()
    return result


__all__ = [
    "GateResult",
    "QualificationResult",
    "GateThresholds",
    "DEFAULT_THRESHOLDS",
    "evaluate_gates",
]
