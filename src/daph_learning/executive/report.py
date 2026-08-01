"""DAPH v0.4 — Generic executive qualification report generation.

Generates human-readable and machine-readable reports for arbitrary
N-action executive qualification experiments. This generalizes the
v0.3.x Gate A report (which was hard-coded for symbolic vs llm) to
work with any action space.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from daph_learning.executive.types import ActionSpace
from daph_learning.executive.qualification import (
    ExecutiveQualificationResult,
    ExecutiveTaskRecord,
)


def generate_markdown_report(
    result: ExecutiveQualificationResult,
    *,
    include_per_action: bool = True,
    include_per_group: bool = False,
    records: Sequence[ExecutiveTaskRecord] | None = None,
) -> str:
    """Generate a markdown report for a generic executive qualification.

    Parameters
    ----------
    result : ExecutiveQualificationResult
    include_per_action : bool
        Include per-action utility table.
    include_per_group : bool
        Include per-group breakdown (requires ``records``).
    records : Sequence[ExecutiveTaskRecord] | None
        Task records for per-group breakdown.
    """
    lines: list[str] = []
    space = result.action_space

    lines.append(f"# Executive Qualification Report")
    lines.append(f"")
    lines.append(f"**Experiment:** `{result.experiment_id}`")
    lines.append(f"**Date:** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**Tasks:** {result.n_tasks}  |  **Groups:** {result.n_groups}")
    lines.append(f"**Actions:** {', '.join(space.action_ids)}")
    if space.abstain_id:
        lines.append(f"**Abstain:** `{space.abstain_id}`")
    lines.append(f"")

    # Gate decision
    status = "PASS" if result.gate_passed else "FAIL"
    lines.append(f"## Gate Decision: {status}")
    lines.append(f"")
    if result.gate_failures:
        lines.append(f"Failures:")
        for f in result.gate_failures:
            lines.append(f"- {f}")
        lines.append(f"")

    # Primary endpoint
    lines.append(f"## Primary Endpoint: P1 - P0")
    lines.append(f"")
    p1p0 = result.p1_minus_p0
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Point estimate | {p1p0.get('point', 0):.4f} |")
    lines.append(f"| 95% LCB | {p1p0.get('lcb_95', 0):.4f} |")
    lines.append(f"| 95% UCB | {p1p0.get('ucb_95', 0):.4f} |")
    lines.append(f"| Bootstrap mean | {p1p0.get('bootstrap_mean', 0):.4f} |")
    lines.append(f"| Bootstrap std | {p1p0.get('bootstrap_std', 0):.4f} |")
    lines.append(f"")

    # Sham comparison
    if result.p1_minus_sham:
        lines.append(f"## Sham Control: P1 - Sham")
        lines.append(f"")
        p1s = result.p1_minus_sham
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        for k, v in p1s.items():
            lines.append(f"| {k} | {v:.4f} |")
        lines.append(f"")

    # Utility summary
    lines.append(f"## Utility Summary")
    lines.append(f"")
    lines.append(f"| Policy | Mean Utility |")
    lines.append(f"|--------|-------------|")
    lines.append(f"| P1 (learned) | {result.p1_mean_utility:.4f} |")
    lines.append(f"| P0 (baseline) | {result.p0_mean_utility:.4f} |")
    lines.append(f"| Oracle | {result.oracle_mean_utility:.4f} |")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Oracle gap (oracle - P0) | {result.oracle_gap:.4f} |")
    lines.append(f"| Oracle gap capture | {result.oracle_gap_capture:.1%} |")
    lines.append(f"| Positive group fraction | {result.positive_group_fraction:.1%} |")
    lines.append(f"| Worst subtype regression | {result.worst_subtype_regression:.4f} |")
    lines.append(f"")

    # Per-action utilities
    if include_per_action and result.always_action_utilities:
        lines.append(f"## Per-Action Utilities (always-action baseline)")
        lines.append(f"")
        lines.append(f"| Action | Always-Action Utility |")
        lines.append(f"|--------|---------------------|")
        for action_id in space.action_ids:
            u = result.always_action_utilities.get(action_id, 0.0)
            lines.append(f"| `{action_id}` | {u:.4f} |")
        lines.append(f"")

    # Per-group breakdown
    if include_per_group and records:
        lines.append(f"## Per-Group Breakdown")
        lines.append(f"")
        groups: dict[str, list[ExecutiveTaskRecord]] = {}
        for r in records:
            groups.setdefault(r.group_id, []).append(r)
        lines.append(f"| Group | N | P1 mean | P0 mean | P1-P0 |")
        lines.append(f"|-------|---|---------|---------|-------|")
        for g in sorted(groups):
            recs = groups[g]
            p1m = sum(r.p1_realized_utility for r in recs) / len(recs)
            p0m = sum(r.p0_realized_utility for r in recs) / len(recs)
            lines.append(f"| {g} | {len(recs)} | {p1m:.4f} | {p0m:.4f} | {p1m-p0m:.4f} |")
        lines.append(f"")

    return "\n".join(lines)


def generate_json_report(
    result: ExecutiveQualificationResult,
    *,
    records: Sequence[ExecutiveTaskRecord] | None = None,
) -> dict[str, Any]:
    """Generate a JSON-serializable report dict."""
    report = result.to_dict()
    if records:
        report["task_records"] = [r.to_dict() for r in records]
    report["report_generated_at"] = datetime.now(timezone.utc).isoformat()
    return report


def write_report(
    result: ExecutiveQualificationResult,
    output_dir: Path,
    *,
    records: Sequence[ExecutiveTaskRecord] | None = None,
    experiment_id: str | None = None,
) -> dict[str, Path]:
    """Write markdown and JSON reports to ``output_dir``.

    Returns dict of ``{"markdown": path, "json": path}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eid = experiment_id or result.experiment_id

    md_path = output_dir / f"{eid}_qualification_report.md"
    md_path.write_text(generate_markdown_report(
        result, include_per_group=True, records=records))

    json_path = output_dir / f"{eid}_qualification_report.json"
    json_path.write_text(json.dumps(
        generate_json_report(result, records=records), indent=2))

    return {"markdown": md_path, "json": json_path}
