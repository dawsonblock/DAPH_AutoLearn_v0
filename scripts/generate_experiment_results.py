#!/usr/bin/env python3
"""Generate EXPERIMENT_RESULTS.md from artifacts/current/pointer.json.

This script reads the current artifact pointer and the referenced bundle's
experiment_results.json and gate_decision.json to produce a single
authoritative top-level EXPERIMENT_RESULTS.md.

Usage:
    python scripts/generate_experiment_results.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def generate() -> str:
    pointer_path = REPO / "artifacts" / "current" / "pointer.json"
    if not pointer_path.exists():
        return _render_no_pointer()
    pointer = _load_json(pointer_path)
    target = pointer.get("target", "")
    if not target:
        return _render_no_pointer()

    # Resolve relative to artifacts/current/
    bundle_dir = (pointer_path.parent / target).resolve()
    if not bundle_dir.exists():
        return _render_missing_bundle(pointer, bundle_dir)

    results_path = bundle_dir / "experiment_results.json"
    gate_path = bundle_dir / "gate_decision.json"
    if not results_path.exists():
        return _render_missing_bundle(pointer, bundle_dir)

    stats = _load_json(results_path)
    gate = _load_json(gate_path) if gate_path.exists() else {}

    return _render_results(pointer, stats, gate)


def _render_no_pointer() -> str:
    return (
        "# Experiment Results — DAPH AutoLearn\n\n"
        "> **No current artifact pointer.** Gate A has not been evaluated.\n\n"
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_\n"
    )


def _render_missing_bundle(pointer: dict, bundle_dir: Path) -> str:
    return (
        "# Experiment Results — DAPH AutoLearn\n\n"
        f"> **Pointer target not found.** The current pointer references "
        f"`{pointer.get('target', '?')}` but the bundle directory does not "
        f"exist at `{bundle_dir}`.\n\n"
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_\n"
    )


def _render_results(pointer: dict, stats: dict, gate: dict) -> str:
    exp_id = stats.get("experiment_id", pointer.get("experiment_id", "unknown"))
    status = pointer.get("qualification_status", "UNKNOWN")
    evidence = pointer.get("evidence_level", "UNKNOWN")
    model_id = stats.get("model_id", "unknown")
    model_rev = stats.get("model_revision", "unknown")

    lines = [
        f"# Experiment Results — {exp_id}",
        "",
        f"**Generated from:** `artifacts/current/pointer.json`",
        f"**Generated at:** {datetime.now(timezone.utc).isoformat()}",
        f"**Experiment ID:** {exp_id}",
        f"**Qualification status:** **{status}**",
        f"**Evidence level:** {evidence}",
        f"**Model:** {model_id} (revision: `{model_rev}`)",
        "",
        "## Gate Decision",
        "",
    ]

    overall = gate.get("overall_status", status)
    lines.append(f"**Overall:** {overall}")
    lines.append("")

    verdicts = gate.get("gate_verdicts", {})
    if verdicts:
        lines.append("| Gate | Comparator | Threshold | Actual | Passed |")
        lines.append("|------|-----------|-----------|--------|--------|")
        for name, v in verdicts.items():
            comp = v.get("comparator", "?")
            thresh = v.get("threshold", "?")
            actual = v.get("actual", "?")
            passed = "PASS" if v.get("passed") else "FAIL"
            if isinstance(actual, float):
                actual = f"{actual:.4f}"
            if isinstance(thresh, float):
                thresh = f"{thresh:.4f}"
            lines.append(f"| {name} | {comp} | {thresh} | {actual} | {passed} |")
        lines.append("")

    pe = stats.get("primary_endpoint", {})
    if pe:
        lines.extend([
            "## Primary Endpoint (P1 − P0)",
            "",
            f"- **Estimand:** {pe.get('estimand', '?')}",
            f"- **Point estimate:** {pe.get('point_estimate', 0):.4f}",
            f"- **95% CI:** [{pe.get('ci_low', 0):.4f}, {pe.get('ci_high', 0):.4f}]",
            f"- **Bootstrap iterations:** {pe.get('n_iterations', '?')}",
            "",
        ])

    sham = stats.get("sham", {})
    if sham:
        lines.extend([
            "## Sham Comparison (P1 − Sham)",
            "",
            f"- **P1 utility:** {sham.get('p1_utility', 0):.4f}",
            f"- **Mean sham utility:** {sham.get('mean_sham_utility', 0):.4f}",
            f"- **P1 − sham mean:** {sham.get('p1_minus_sham_mean', 0):.4f}",
            f"- **P1 − sham 95% CI:** [{sham.get('p1_minus_sham_ci_low', 0):.4f}, {sham.get('p1_minus_sham_ci_high', 0):.4f}]",
            f"- **P1 percentile vs sham:** {sham.get('p1_percentile_vs_sham', 0):.1f}%",
            f"- **Sham seeds:** {sham.get('n_seeds', '?')}",
            "",
        ])

    rd = stats.get("route_distribution", {})
    if rd:
        lines.extend([
            "## Route Distribution",
            "",
            f"- P1 symbolic fraction: {rd.get('p1_symbolic_fraction', 0):.1%}",
            f"- P1 LLM fraction: {rd.get('p1_llm_fraction', 0):.1%}",
            f"- Oracle symbolic fraction: {rd.get('oracle_symbolic_fraction', 0):.1%}",
            f"- Oracle LLM fraction: {rd.get('oracle_llm_fraction', 0):.1%}",
            f"- P1-oracle action agreement: {rd.get('p1_oracle_action_agreement', 0):.1%}",
            "",
        ])

    lines.extend([
        "## Summary Metrics",
        "",
        f"- **Oracle gap capture:** {stats.get('oracle_gap_capture', 0):.4f}" if stats.get("oracle_gap_capture") is not None else "- **Oracle gap capture:** N/A",
        f"- **P1 utility:** {stats.get('p1_utility', 0):.4f}",
        f"- **P0 utility:** {stats.get('p0_utility', 0):.4f}",
        f"- **Oracle utility:** {stats.get('oracle_utility', 0):.4f}",
        f"- **Positive group fraction:** {stats.get('positive_group_fraction', 0):.1%}",
        f"- **Worst subtype regression:** {stats.get('worst_subtype_regression', 0):.4f}",
        "",
        "---",
        "",
        f"*This file is auto-generated from `artifacts/current/pointer.json`. "
        f"Do not edit manually — run `python scripts/generate_experiment_results.py` to regenerate.*",
    ])

    return "\n".join(lines) + "\n"


def main() -> int:
    content = generate()
    out_path = REPO / "EXPERIMENT_RESULTS.md"
    out_path.write_text(content)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
