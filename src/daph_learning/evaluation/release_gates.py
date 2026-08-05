"""v0.3.10.3.2-alpha — release-gate claim contract + G1-G32 gate
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
                ("tests/test_all_version_surfaces_match.py",
                 "tests/test_version_claims_discipline.py")),
    ReleaseGate("G2", "2", "Canonical source-tree hash (64-char SHA-256)",
                ("canonical-source-hash",),
                ("tests/test_canonical_source_hash.py",),),
    ReleaseGate("G3", "3", "One canonical utility function everywhere",
                ("canonical-utility",),
                ("tests/test_canonical_utility_v0310_3_1.py",),),
    ReleaseGate("G4", "4-5", "BackendOutcome distinguishes verified_correct from correct",
                ("outcome-contract",),
                ("tests/test_outcome_contract_v0310_3_1.py",),),
    ReleaseGate("G5", "4-5", "Confidence != quality semantics",
                ("confidence-quality-separation",),
                ("tests/test_outcome_contract_v0310_3_1.py",),),
    ReleaseGate("G6", "6-8", "Benchmark arithmetic correctness (no truncation)",
                ("benchmark-arithmetic",),
                ("tests/test_benchmark_arithmetic_correctness.py",),),
    ReleaseGate("G7", "11", "No within-split prompt duplicates",
                ("within-split-dedup",),
                ("tests/test_no_duplicate_prompts_per_split.py",
                 "tests/test_crossover_benchmark_v0310_3_1.py")),
    ReleaseGate("G8", "9", "Grouped bootstrap resamples groups not records",
                ("grouped-bootstrap",),
                ("tests/test_benchmark_stats_v0310_3_1.py",
                 "tests/test_verifier_modes_ablations.py")),
    ReleaseGate("G9", "10", "Linguistic template split disjointness",
                ("template-disjointness",),
                ("tests/test_template_disjointness.py",),),
    ReleaseGate("G10", "14-15", "Final split inaccessible before FROZEN",
                ("stage-access-control",),
                ("tests/test_stage_freeze_final_guards.py",
                 "tests/test_stage_access_v0310_3_1.py")),
    ReleaseGate("G11", "15", "Final access ledger records every access",
                ("final-access-ledger",),
                ("tests/test_stage_freeze_final_guards.py",
                 "tests/test_stage_access_v0310_3_1.py")),
    ReleaseGate("G12", "11", "No optimal backend encoded in task metadata",
                ("no-metadata-leakage",),
                ("tests/test_within_subtype_crossover.py",
                 "tests/test_crossover_benchmark_v0310_3_1.py")),
    ReleaseGate("G13", "13", "Strong hand router baseline exists",
                ("hand-router-baseline",),
                ("tests/test_baseline_matrix.py",),),
    ReleaseGate("G14", "20", "Fail closed on missing utility/verifier/policy",
                ("fail-closed",),
                ("tests/test_kl_failclosed_v0310_3_1.py",
                 "tests/test_cli_path_completion.py")),
    ReleaseGate("G15", "21-25", "Steering evaluated by ΔU(α) not P(symbolic)",
                ("steering-utility",),
                ("tests/test_steering_utility_evidence.py",
                 "tests/test_steering_utility_v0310_3_1.py")),
    ReleaseGate("G16", "23", "Beneficial/harmful flip classification",
                ("flip-classification",),
                ("tests/test_steering_utility_evidence.py",
                 "tests/test_steering_utility_v0310_3_1.py")),
    ReleaseGate("G17", "24", "Oracle alpha probe (DEV only diagnostic)",
                ("oracle-alpha",),
                ("tests/test_steering_utility_evidence.py",
                 "tests/test_steering_utility_v0310_3_1.py")),
    ReleaseGate("G18", "25", "Random control p_emp for steering gain",
                ("random-control",),
                ("tests/test_steering_utility_evidence.py",
                 "tests/test_steering_utility_v0310_3_1.py")),
    ReleaseGate("G19", "26", "Neutral KL release gate (3 cases)",
                ("kl-gate",),
                ("tests/test_kl_failclosed_v0310_3_1.py",),),
    ReleaseGate("G20", "27", "Verifier naming: exact vs constrained_answer",
                ("verifier-naming",),
                ("tests/test_v0310_2_real_gates.py",
                 "tests/test_verifier_modes_ablations.py")),
    ReleaseGate("G21", "28", "Capture representation ablation (DEV only)",
                ("capture-ablation",),
                ("tests/test_steering_utility_evidence.py",
                 "tests/test_capture_comparison_v0310_3_1.py")),
    ReleaseGate("G22", "29", "Decisive policy class comparison",
                ("decisive-comparison",),
                ("tests/test_steering_utility_evidence.py",
                 "tests/test_capture_comparison_v0310_3_1.py")),
    ReleaseGate("G23", "30", "Tie-aware metrics (win/tie/loss)",
                ("tie-aware",),
                ("tests/test_capture_comparison_v0310_3_1.py",),),
    ReleaseGate("G24", "31", "ESS reported for weighted estimators",
                ("ess-reporting",),
                ("tests/test_benchmark_stats_v0310_3_1.py",
                 "tests/test_verifier_modes_ablations.py",
                 "tests/test_capture_comparison_v0310_3_1.py")),
    ReleaseGate("G25", "32", "Artifact directory discipline",
                ("artifact-discipline",),
                ("tests/test_current_artifact_tree_contains_no_stale_source_hash.py",
                 "tests/test_artifact_integrity_v0310_3_1.py")),
    ReleaseGate("G26", "33", "Source tree hash enforcement (canonical)",
                ("source-hash",),
                ("tests/test_canonical_source_hash.py",
                 "tests/test_artifact_integrity_v0310_3_1.py")),
    ReleaseGate("G27", "34", "Test collection hash recorded",
                ("test-collection-hash",),
                ("tests/test_artifact_integrity_v0310_3_1.py",),),
    ReleaseGate("G28", "35", "Re-run test report",
                ("rerun-test-report",),
                ("tests/test_artifact_integrity_v0310_3_1.py",),),
    ReleaseGate("G29", "37", "Real LLM backend integration (Qwen)",
                ("real-llm-integration",),
                ("tests/test_v0310_2_real_gates.py",),),
    ReleaseGate("G30", "38", "Real symbolic executor integration",
                ("real-symbolic-integration",),
                ("tests/test_v0310_2_real_gates.py",),),
    ReleaseGate("G31", "12-17", "Within-subtype crossover (>= 3 subtypes)",
                ("within-subtype-crossover",),
                ("tests/test_within_subtype_crossover.py",),),
    ReleaseGate("G32", "22-27", "CLI paths completed (real backends)",
                ("cli-path-completion",),
                ("tests/test_cli_path_completion.py",
                 "tests/test_cli_entrypoints.py")),
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
