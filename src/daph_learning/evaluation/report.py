"""Section 19 — Gate A report generator (v0.3.10.4-alpha).

One generator that reads ONLY validated artifacts and emits the full
required output set. ``GATE_A_RESULTS.md`` is generated from
machine-readable artifacts — no manually typed numbers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Comparator(str, Enum):
    """Typed threshold comparator for gate evaluation."""
    GT = "gt"    # strictly greater than
    GTE = "gte"  # greater than or equal
    LT = "lt"    # strictly less than
    LTE = "lte"  # less than or equal

    def evaluate(self, actual: float, threshold: float) -> bool:
        if self == Comparator.GT:
            return actual > threshold
        elif self == Comparator.GTE:
            return actual >= threshold
        elif self == Comparator.LT:
            return actual < threshold
        elif self == Comparator.LTE:
            return actual <= threshold
        return False


class GateStatus(str, Enum):
    """Status of a gate evaluation."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"  # precondition failure


@dataclass
class GateDecision:
    """Section 19 — gate-by-gate verdict + final PASS/FAIL."""
    passed: bool
    gate_verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)
    experiment_id: str = ""
    criteria_hash: str = ""
    overall_status: str = "PASS"  # PASS, FAIL, NOT_EVALUABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "experiment_id": self.experiment_id,
            "criteria_hash": self.criteria_hash,
            "overall_status": self.overall_status,
            "gate_verdicts": dict(self.gate_verdicts),
        }


def evaluate_gates(
    stats: dict[str, Any],
    criteria: dict[str, Any],
) -> GateDecision:
    """Section 18/19 — evaluate each preregistered gate against the
    computed statistics. Returns a GateDecision with per-gate verdicts.

    Precondition failures (insufficient groups, no crossover) yield
    NOT_EVALUABLE for all downstream gates, rather than counting them
    as passed.
    """
    gates = criteria.get("gates", {})
    primary = stats.get("primary_endpoint", {})
    route_dist = stats.get("route_distribution", {})
    dataset = stats.get("dataset", {})
    verdicts: dict[str, dict[str, Any]] = {}
    all_passed = True
    precondition_failed = False

    def _check(name: str, actual: float, threshold: float,
               comparator: Comparator = Comparator.GT) -> None:
        passed = comparator.evaluate(actual, threshold)
        verdicts[name] = {
            "actual": actual,
            "threshold": threshold,
            "comparator": comparator.value,
            "passed": passed,
        }
        if not passed:
            nonlocal all_passed
            all_passed = False

    def _not_evaluable(name: str, reason: str) -> None:
        nonlocal all_passed, precondition_failed
        precondition_failed = True
        all_passed = False
        verdicts[name] = {
            "actual": None,
            "threshold": None,
            "comparator": None,
            "passed": False,
            "status": "NOT_EVALUABLE",
            "reason": reason,
        }

    # --- Precondition checks ---
    # 1. Minimum independent groups.
    min_groups = criteria.get("dataset", {}).get("minimum_groups", 1)
    actual_groups = dataset.get("n_groups", 0)
    if actual_groups < min_groups:
        _not_evaluable(
            "precondition_minimum_groups",
            f"only {actual_groups} final groups, required {min_groups}")

    # 2. Meaningful crossover (both backends must win some tasks).
    oracle_sym_frac = route_dist.get("oracle_symbolic_fraction", 1.0)
    oracle_llm_frac = route_dist.get("oracle_llm_fraction", 0.0)
    if oracle_sym_frac >= 1.0 or oracle_llm_frac >= 1.0:
        _not_evaluable(
            "precondition_crossover",
            f"no backend crossover: oracle routes "
            f"{oracle_sym_frac:.1%} symbolic, {oracle_llm_frac:.1%} LLM")

    # --- Statistical gates (only if preconditions pass) ---
    if precondition_failed:
        overall_status = GateStatus.NOT_EVALUABLE.value
        # Mark all statistical gates as NOT_EVALUABLE.
        for gate_name in ("minimum_point_gain_vs_p0",
                          "require_lcb_vs_p0_above",
                          "require_lcb_vs_sham_above",
                          "minimum_oracle_gap_capture",
                          "minimum_positive_group_fraction",
                          "maximum_worst_subtype_regression",
                          "maximum_final_access_count"):
            if gate_name not in verdicts:
                verdicts[gate_name] = {
                    "actual": None, "threshold": None,
                    "comparator": None, "passed": False,
                    "status": "NOT_EVALUABLE",
                    "reason": "precondition failure",
                }
    else:
        overall_status = GateStatus.PASS.value if all_passed else GateStatus.FAIL.value

        _check("minimum_point_gain_vs_p0",
               primary.get("point_estimate", 0.0),
               float(gates.get("minimum_point_gain_vs_p0", 0.0)),
               Comparator.GT)
        _check("require_lcb_vs_p0_above",
               primary.get("ci_low", 0.0),
               float(gates.get("require_lcb_vs_p0_above", 0.0)),
               Comparator.GT)
        _check("require_lcb_vs_sham_above",
               stats.get("sham", {}).get("p1_minus_sham_ci_low", 0.0),
               float(gates.get("require_lcb_vs_sham_above", 0.0)),
               Comparator.GT)
        _check("minimum_oracle_gap_capture",
               stats.get("oracle_gap_capture", 0.0),
               float(gates.get("minimum_oracle_gap_capture", 0.0)),
               Comparator.GTE)  # >= threshold (not strict >)
        _check("minimum_positive_group_fraction",
               stats.get("positive_group_fraction", 0.0),
               float(gates.get("minimum_positive_group_fraction", 0.0)),
               Comparator.GT)
        _check("maximum_worst_subtype_regression",
               stats.get("worst_subtype_regression", 1.0),
               float(gates.get("maximum_worst_subtype_regression", 1.0)),
               Comparator.LTE)
        _check("maximum_final_access_count",
               float(stats.get("final_access_count", 0.0)),
               float(gates.get("maximum_final_access_count", 1)),
               Comparator.LTE)

        overall_status = GateStatus.PASS.value if all_passed else GateStatus.FAIL.value

    return GateDecision(
        passed=all_passed,
        gate_verdicts=verdicts,
        experiment_id=criteria.get("experiment_id", ""),
        criteria_hash=criteria.get("criteria_hash", ""),
        overall_status=overall_status,
    )


def generate_gate_a_results_md(
    decision: GateDecision,
    stats: dict[str, Any],
    criteria: dict[str, Any],
) -> str:
    """Section 19 — generate GATE_A_RESULTS.md from machine-readable
    artifacts. No manually typed numbers.
    """
    lines = [
        f"# Gate A Results — {decision.experiment_id}",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"**Criteria hash:** `{decision.criteria_hash}`",
        f"**Overall status:** {decision.overall_status}",
        "",
        f"## Final Verdict: {decision.overall_status}",
        "",
        "## Gate-by-Gate Verdicts",
        "",
        "| Gate | Actual | Threshold | Comparator | Passed |",
        "|------|--------|-----------|------------|--------|",
    ]
    for name, v in decision.gate_verdicts.items():
        actual = v.get("actual")
        threshold = v.get("threshold")
        comparator = v.get("comparator", "")
        status = v.get("status", "")
        if status == "NOT_EVALUABLE":
            lines.append(
                f"| {name} | N/A | N/A | N/A | NOT_EVALUABLE: {v.get('reason', '')} |")
        else:
            actual_str = f"{actual:.4f}" if actual is not None else "N/A"
            threshold_str = f"{threshold:.4f}" if threshold is not None else "N/A"
            lines.append(
                f"| {name} | {actual_str} | {threshold_str} | "
                f"{comparator} | {'YES' if v['passed'] else 'NO'} |")
    lines.extend([
        "",
        "## Primary Endpoint",
        "",
        f"- Estimand: {stats.get('primary_endpoint', {}).get('estimand', 'N/A')}",
        f"- Point estimate: {stats.get('primary_endpoint', {}).get('point_estimate', 'N/A')}",
        f"- 95% CI ({stats.get('primary_endpoint', {}).get('estimand', 'N/A')}): "
        f"[{stats.get('primary_endpoint', {}).get('ci_low', 'N/A')}, "
        f"{stats.get('primary_endpoint', {}).get('ci_high', 'N/A')}]",
        f"- CI label: {stats.get('primary_endpoint', {}).get('ci_label', stats.get('primary_endpoint', {}).get('estimand', 'N/A'))}",
        f"- Utility protocol: {stats.get('utility_protocol', 'N/A')}",
        "",
        "## Route Distribution",
        "",
        f"- P1 symbolic fraction: {stats.get('route_distribution', {}).get('p1_symbolic_fraction', 'N/A')}",
        f"- P1 LLM fraction: {stats.get('route_distribution', {}).get('p1_llm_fraction', 'N/A')}",
        f"- P1 abstain fraction: {stats.get('route_distribution', {}).get('p1_abstain_fraction', 'N/A')}",
        f"- Oracle symbolic fraction: {stats.get('route_distribution', {}).get('oracle_symbolic_fraction', 'N/A')}",
        f"- Oracle LLM fraction: {stats.get('route_distribution', {}).get('oracle_llm_fraction', 'N/A')}",
        f"- P1–oracle action agreement: {stats.get('route_distribution', {}).get('p1_oracle_action_agreement', 'N/A')}",
        f"- P1–always-symbolic agreement: {stats.get('route_distribution', {}).get('p1_always_symbolic_agreement', 'N/A')}",
        "",
        "## Sham Control",
        "",
        f"- P1 utility: {stats.get('sham', {}).get('p1_utility', 'N/A')}",
        f"- Mean sham utility: {stats.get('sham', {}).get('mean_sham_utility', 'N/A')}",
        f"- P1 minus sham (mean): {stats.get('sham', {}).get('p1_minus_sham_mean', 'N/A')}",
        f"- P1 minus sham 95% CI: "
        f"[{stats.get('sham', {}).get('p1_minus_sham_ci_low', 'N/A')}, "
        f"{stats.get('sham', {}).get('p1_minus_sham_ci_high', 'N/A')}]",
        f"- P1 percentile vs sham: {stats.get('sham', {}).get('p1_percentile_vs_sham', 'N/A')}%",
        f"- Sham seeds: {stats.get('sham', {}).get('n_seeds', 'N/A')}",
        f"- Training spec hash: `{stats.get('sham', {}).get('training_spec_hash', 'N/A')}`",
        "",
        "## Dataset",
        "",
        f"- N groups: {stats.get('dataset', {}).get('n_groups', 'N/A')}",
        f"- N tasks: {stats.get('dataset', {}).get('n_tasks', 'N/A')}",
        f"- Crossover subtypes: {stats.get('dataset', {}).get('n_crossover_subtypes', 'N/A')}",
        f"- Decisive fraction: {stats.get('dataset', {}).get('decisive_fraction', 'N/A')}",
        "",
        "## Baselines",
        "",
    ])
    for name, info in stats.get("baselines", {}).items():
        lines.append(f"- {name}: utility={info.get('utility', 'N/A')}")
    lines.extend([
        "",
        "## Final Access",
        "",
        f"- Access count: {stats.get('final_access_count', 'N/A')}",
        f"- Source hash: `{stats.get('source_hash', 'N/A')}`",
        "",
        "---",
        "This report was generated from machine-readable artifacts. "
        "No numbers were manually typed.",
    ])
    return "\n".join(lines)


def generate_report(
    out_dir: str | Path,
    *,
    stats: dict[str, Any],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    """Section 19 — generate the full report output set.

    Writes: ``gate_decision.json``, ``GATE_A_RESULTS.md``,
    ``experiment_results.json``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    decision = evaluate_gates(stats, criteria)

    # gate_decision.json
    (out / "gate_decision.json").write_text(
        json.dumps(decision.to_dict(), indent=2))

    # GATE_A_RESULTS.md
    md = generate_gate_a_results_md(decision, stats, criteria)
    (out / "GATE_A_RESULTS.md").write_text(md)

    # experiment_results.json
    (out / "experiment_results.json").write_text(
        json.dumps(stats, indent=2, default=str))

    return {
        "decision": decision.to_dict(),
        "output_dir": str(out),
        "files": ["gate_decision.json", "GATE_A_RESULTS.md",
                  "experiment_results.json"],
    }


__all__ = [
    "GateDecision",
    "evaluate_gates",
    "generate_gate_a_results_md",
    "generate_report",
]
