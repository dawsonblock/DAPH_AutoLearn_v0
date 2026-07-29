"""v0.3.10.3.1-alpha — release-gate claim contract + G1-G32 gate
registry (Sections 37-40).

Every scientific claim in the release is bound to one or more gates.
A gate is a named, testable assertion with:
  * an ID (G1-G32)
  * a description
  * a status (pass/fail/skipped)
  * the test(s) that verify it
  * the claim(s) it supports

The registry is the single source of truth for what the release
claims and what evidence backs each claim. The release report
(Section 41-48) is generated FROM this registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReleaseGate:
    """One release gate (G1-G32)."""
    gate_id: str
    section: str
    description: str
    claims_supported: tuple[str, ...]
    test_files: tuple[str, ...] = ()


# The full G1-G32 gate registry. Each gate maps to a section, a
# human-readable description, the claims it supports, and the test
# file(s) that verify it.
GATE_REGISTRY: list[ReleaseGate] = [
    ReleaseGate("G1", "1", "Version unified across all surfaces",
                ("version-consistency",),
                ("tests/test_version_claims_discipline.py",)),
    ReleaseGate("G2", "2", "Centroid zero-weight fails closed",
                ("zero-weight-fail-closed",),
                ("tests/test_zero_weight_policy_v0310_3_1.py",)),
    ReleaseGate("G3", "3", "One canonical utility function everywhere",
                ("canonical-utility",),
                ("tests/test_canonical_utility_v0310_3_1.py",)),
    ReleaseGate("G4", "4-5", "BackendOutcome distinguishes verified_correct from correct",
                ("outcome-contract",),
                ("tests/test_outcome_contract_v0310_3_1.py",)),
    ReleaseGate("G5", "4-5", "Confidence != quality semantics",
                ("confidence-quality-separation",),
                ("tests/test_outcome_contract_v0310_3_1.py",)),
    ReleaseGate("G6", "6-7", "True weighted vs unweighted ablation (different weights)",
                ("weighted-ablation",),
                ("tests/test_benchmark_stats_v0310_3_1.py",)),
    ReleaseGate("G7", "8", "No within-split prompt duplicates",
                ("within-split-dedup",),
                ("tests/test_benchmark_stats_v0310_3_1.py",
                 "tests/test_crossover_benchmark_v0310_3_1.py")),
    ReleaseGate("G8", "9", "Grouped bootstrap resamples groups not records",
                ("grouped-bootstrap",),
                ("tests/test_benchmark_stats_v0310_3_1.py",)),
    ReleaseGate("G9", "10-12", "Within-family crossover: 0.2 < P(symbolic) < 0.8",
                ("crossover-benchmark",),
                ("tests/test_crossover_benchmark_v0310_3_1.py",)),
    ReleaseGate("G10", "14-15", "Final split inaccessible before FROZEN",
                ("stage-access-control",),
                ("tests/test_stage_access_v0310_3_1.py",)),
    ReleaseGate("G11", "15", "Final access ledger records every access",
                ("final-access-ledger",),
                ("tests/test_stage_access_v0310_3_1.py",)),
    ReleaseGate("G12", "11", "No optimal backend encoded in task metadata",
                ("no-metadata-leakage",),
                ("tests/test_crossover_benchmark_v0310_3_1.py",)),
    ReleaseGate("G13", "13", "Strong hand router baseline exists",
                ("hand-router-baseline",),
                ("tests/test_benchmark_stats_v0310_3_1.py",)),
    ReleaseGate("G14", "20", "Fail closed on missing utility/verifier/policy",
                ("fail-closed",),
                ("tests/test_kl_failclosed_v0310_3_1.py",)),
    ReleaseGate("G15", "21-25", "Steering evaluated by ΔU(α) not P(symbolic)",
                ("steering-utility",),
                ("tests/test_steering_utility_v0310_3_1.py",)),
    ReleaseGate("G16", "23", "Beneficial/harmful flip classification",
                ("flip-classification",),
                ("tests/test_steering_utility_v0310_3_1.py",)),
    ReleaseGate("G17", "24", "Oracle alpha probe (DEV only diagnostic)",
                ("oracle-alpha",),
                ("tests/test_steering_utility_v0310_3_1.py",)),
    ReleaseGate("G18", "25", "Random control p_emp for steering gain",
                ("random-control",),
                ("tests/test_steering_utility_v0310_3_1.py",)),
    ReleaseGate("G19", "26", "Neutral KL release gate (3 cases)",
                ("kl-gate",),
                ("tests/test_kl_failclosed_v0310_3_1.py",)),
    ReleaseGate("G20", "27", "Verifier naming: exact vs constrained_answer",
                ("verifier-naming",),
                ("tests/test_v0310_2_real_gates.py",)),
    ReleaseGate("G21", "28", "Capture representation ablation (DEV only)",
                ("capture-ablation",),
                ("tests/test_capture_comparison_v0310_3_1.py",)),
    ReleaseGate("G22", "29", "Decisive policy class comparison",
                ("decisive-comparison",),
                ("tests/test_capture_comparison_v0310_3_1.py",)),
    ReleaseGate("G23", "30", "Tie-aware metrics (win/tie/loss)",
                ("tie-aware",),
                ("tests/test_capture_comparison_v0310_3_1.py",)),
    ReleaseGate("G24", "31", "ESS reported for weighted estimators",
                ("ess-reporting",),
                ("tests/test_benchmark_stats_v0310_3_1.py",
                 "tests/test_capture_comparison_v0310_3_1.py")),
    ReleaseGate("G25", "32", "Artifact directory discipline",
                ("artifact-discipline",),
                ("tests/test_artifact_integrity_v0310_3_1.py",)),
    ReleaseGate("G26", "33", "Source tree hash enforcement",
                ("source-hash",),
                ("tests/test_artifact_integrity_v0310_3_1.py",)),
    ReleaseGate("G27", "34", "Test collection hash recorded",
                ("test-collection-hash",),
                ("tests/test_artifact_integrity_v0310_3_1.py",)),
    ReleaseGate("G28", "35", "Re-run test report",
                ("rerun-test-report",),
                ("tests/test_artifact_integrity_v0310_3_1.py",)),
    ReleaseGate("G29", "37", "Real LLM backend integration (Qwen)",
                ("real-llm-integration",),
                ("tests/test_v0310_2_real_gates.py",)),
    ReleaseGate("G30", "38", "Real symbolic executor integration",
                ("real-symbolic-integration",),
                ("tests/test_v0310_2_real_gates.py",)),
    ReleaseGate("G31", "39", "Release-gate claim contract",
                ("claim-contract",),
                ("tests/test_release_gates_v0310_3_1.py",)),
    ReleaseGate("G32", "40", "All G1-G31 gates pass",
                ("all-gates-pass",),
                ("tests/test_release_gates_v0310_3_1.py",)),
]


@dataclass
class GateResult:
    """Result of evaluating one gate."""
    gate_id: str
    status: str  # "pass", "fail", "skipped"
    n_tests: int = 0
    error: str | None = None


@dataclass
class ReleaseGateReport:
    """Section 40: full release-gate report."""
    results: list[GateResult] = field(default_factory=list)
    n_pass: int = 0
    n_fail: int = 0
    n_skip: int = 0
    all_pass: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [
                {"gate_id": r.gate_id, "status": r.status,
                 "n_tests": r.n_tests, "error": r.error}
                for r in self.results],
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_skip": self.n_skip,
            "all_pass": self.all_pass,
        }


def get_gate(gate_id: str) -> ReleaseGate:
    """Look up a gate by ID."""
    for g in GATE_REGISTRY:
        if g.gate_id == gate_id:
            return g
    raise KeyError(f"unknown gate {gate_id!r}")


def all_gate_ids() -> list[str]:
    """Return all gate IDs in order."""
    return [g.gate_id for g in GATE_REGISTRY]


def all_claims() -> list[str]:
    """Return all claims supported by the gates."""
    seen = set()
    claims = []
    for g in GATE_REGISTRY:
        for c in g.claims_supported:
            if c not in seen:
                seen.add(c)
                claims.append(c)
    return claims


def gates_for_claim(claim: str) -> list[ReleaseGate]:
    """Return all gates that support a given claim."""
    return [g for g in GATE_REGISTRY if claim in g.claims_supported]


__all__ = [
    "GATE_REGISTRY",
    "GateResult",
    "ReleaseGate",
    "ReleaseGateReport",
    "all_claims",
    "all_gate_ids",
    "gates_for_claim",
    "get_gate",
]
