"""Section 19 — Gate A report generator (v0.3.10.4-alpha).

One generator that reads ONLY validated artifacts and emits the full
required output set. ``GATE_A_RESULTS.md`` is generated from
machine-readable artifacts — no manually typed numbers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GateDecision:
    """Section 19 — gate-by-gate verdict + final PASS/FAIL."""
    passed: bool
    gate_verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)
    experiment_id: str = ""
    criteria_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "experiment_id": self.experiment_id,
            "criteria_hash": self.criteria_hash,
            "gate_verdicts": dict(self.gate_verdicts),
        }


def evaluate_gates(
    stats: dict[str, Any],
    criteria: dict[str, Any],
) -> GateDecision:
    """Section 18/19 — evaluate each preregistered gate against the
    computed statistics. Returns a GateDecision with per-gate verdicts.
    """
    gates = criteria.get("gates", {})
    primary = stats.get("primary_endpoint", {})
    verdicts: dict[str, dict[str, Any]] = {}
    all_passed = True

    def _check(name: str, actual: float, threshold: float,
               direction: str = "above") -> None:
        if direction == "above":
            passed = actual > threshold
        elif direction == "below":
            passed = actual < threshold
        elif direction == "at_most":
            passed = actual <= threshold
        elif direction == "at_least":
            passed = actual >= threshold
        else:
            passed = False
        verdicts[name] = {
            "actual": actual,
            "threshold": threshold,
            "direction": direction,
            "passed": passed,
        }
        if not passed:
            nonlocal all_passed
            all_passed = False

    _check("minimum_point_gain_vs_p0",
           primary.get("point_estimate", 0.0),
           float(gates.get("minimum_point_gain_vs_p0", 0.0)))
    _check("require_lcb_vs_p0_above",
           primary.get("ci_low", 0.0),
           float(gates.get("require_lcb_vs_p0_above", 0.0)))
    _check("require_lcb_vs_sham_above",
           stats.get("sham", {}).get("p1_minus_sham_ci_low", 0.0),
           float(gates.get("require_lcb_vs_sham_above", 0.0)))
    _check("minimum_oracle_gap_capture",
           stats.get("oracle_gap_capture", 0.0),
           float(gates.get("minimum_oracle_gap_capture", 0.0)))
    _check("minimum_positive_group_fraction",
           stats.get("positive_group_fraction", 0.0),
           float(gates.get("minimum_positive_group_fraction", 0.0)))
    _check("maximum_worst_subtype_regression",
           stats.get("worst_subtype_regression", 1.0),
           float(gates.get("maximum_worst_subtype_regression", 1.0)),
           direction="at_most")
    _check("maximum_final_access_count",
           float(stats.get("final_access_count", 0.0)),
           float(gates.get("maximum_final_access_count", 1)),
           direction="at_most")

    return GateDecision(
        passed=all_passed,
        gate_verdicts=verdicts,
        experiment_id=criteria.get("experiment_id", ""),
        criteria_hash=criteria.get("criteria_hash", ""),
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
        "",
        f"## Final Verdict: {'PASS' if decision.passed else 'FAIL'}",
        "",
        "## Gate-by-Gate Verdicts",
        "",
        "| Gate | Actual | Threshold | Direction | Passed |",
        "|------|--------|-----------|-----------|--------|",
    ]
    for name, v in decision.gate_verdicts.items():
        lines.append(
            f"| {name} | {v['actual']:.4f} | {v['threshold']:.4f} | "
            f"{v['direction']} | {'YES' if v['passed'] else 'NO'} |")
    lines.extend([
        "",
        "## Primary Endpoint",
        "",
        f"- Estimand: {stats.get('primary_endpoint', {}).get('estimand', 'N/A')}",
        f"- Point estimate: {stats.get('primary_endpoint', {}).get('point_estimate', 'N/A')}",
        f"- 95% CI: [{stats.get('primary_endpoint', {}).get('ci_low', 'N/A')}, "
        f"{stats.get('primary_endpoint', {}).get('ci_high', 'N/A')}]",
        f"- CI label: {stats.get('primary_endpoint', {}).get('estimand', 'N/A')}",
        "",
        "## Sham Control",
        "",
        f"- P1 utility: {stats.get('sham', {}).get('p1_utility', 'N/A')}",
        f"- Mean sham utility: {stats.get('sham', {}).get('mean_sham_utility', 'N/A')}",
        f"- P1 minus sham (mean): {stats.get('sham', {}).get('p1_minus_sham_mean', 'N/A')}",
        f"- P1 percentile vs sham: {stats.get('sham', {}).get('p1_percentile_vs_sham', 'N/A')}%",
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
