"""v0.3.10.3.1-alpha — decisive policy class comparison (Sections 29-31).

Section 29: decisive comparison
  Compare policy classes (centroid, logistic, hand-router, random)
  on the SAME split with the SAME utility. A policy wins only when
  its mean utility exceeds the runner-up by more than the gap
  threshold AND the grouped-bootstrap CI excludes zero.

Section 30: tie-aware metrics
  When |ΔU| <= gap_threshold, the comparison is a TIE, not a win.
  Report win/tie/loss, not just win/loss.

Section 31: ESS
  Report effective sample size for each weighted estimator so a
  policy that "wins" with ESS=3 is not reported as decisive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from daph_learning.evaluation.grouped_stats import (
    effective_sample_size,
    grouped_bootstrap_mean_delta,
)


@dataclass(frozen=True)
class PolicyComparisonRow:
    """Section 29: one policy's summary on a split."""
    policy_id: str
    mean_utility: float
    mean_regret: float
    n_tasks: int
    ess: float  # Section 31


@dataclass
class PolicyComparisonResult:
    """Section 29-30: pairwise comparison between two policies."""
    policy_a: str
    policy_b: str
    mean_delta_a_minus_b: float
    ci_low: float
    ci_high: float
    verdict: str  # "A_wins", "B_wins", "tie"
    n_groups: int
    n_records: int


@dataclass
class PolicyComparisonReport:
    """Section 29-31: full multi-policy comparison."""
    rows: list[PolicyComparisonRow]
    pairwise: list[PolicyComparisonResult]
    best_policy_id: str | None
    best_is_decisive: bool
    gap_threshold: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [
                {"policy_id": r.policy_id, "mean_utility": r.mean_utility,
                 "mean_regret": r.mean_regret, "n_tasks": r.n_tasks,
                 "ess": r.ess}
                for r in self.rows],
            "pairwise": [
                {"policy_a": p.policy_a, "policy_b": p.policy_b,
                 "mean_delta": p.mean_delta_a_minus_b,
                 "ci_low": p.ci_low, "ci_high": p.ci_high,
                 "verdict": p.verdict, "n_groups": p.n_groups,
                 "n_records": p.n_records}
                for p in self.pairwise],
            "best_policy_id": self.best_policy_id,
            "best_is_decisive": self.best_is_decisive,
            "gap_threshold": self.gap_threshold,
            "notes": self.notes,
        }


def compare_policies(
    tasks: list[dict[str, Any]],
    *,
    policy_routes: dict[str, list[str]],
    utility_fn: Callable[[dict[str, Any], str], float],
    group_key: str = "template_id",
    gap_threshold: float = 0.01,
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> PolicyComparisonReport:
    """Section 29-31: decisive multi-policy comparison.

    Parameters
    ----------
    tasks : list of task mappings
    policy_routes : dict policy_id -> list of routes (one per task)
    utility_fn : callable(task, route) -> utility
    group_key : field for grouped bootstrap (Section 9)
    gap_threshold : |ΔU| below this is a tie (Section 30)
    n_bootstrap : bootstrap iterations
    seed : RNG seed
    """
    policy_ids = list(policy_routes.keys())
    rows: list[PolicyComparisonRow] = []
    # Per-policy utility arrays.
    utilities_by_policy: dict[str, list[float]] = {}
    for pid in policy_ids:
        routes = policy_routes[pid]
        us = [utility_fn(t, r) for t, r in zip(tasks, routes)]
        utilities_by_policy[pid] = us
        # ESS: uniform weights for now (policies don't weight here).
        ess = effective_sample_size(np.ones(len(us)))
        rows.append(PolicyComparisonRow(
            policy_id=pid,
            mean_utility=float(np.mean(us)),
            mean_regret=float(np.mean([
                max(utility_fn(t, "symbolic"), utility_fn(t, "llm")) - u
                for t, u in zip(tasks, us)])),
            n_tasks=len(tasks),
            ess=ess))

    # Pairwise comparisons.
    pairwise: list[PolicyComparisonResult] = []
    for i, pa in enumerate(policy_ids):
        for pb in policy_ids[i + 1:]:
            records = [
                {group_key: t.get("metadata", {}).get(group_key, t.get("task_id")),
                 "u_a": utilities_by_policy[pa][j],
                 "u_b": utilities_by_policy[pb][j]}
                for j, t in enumerate(tasks)]
            bs = grouped_bootstrap_mean_delta(
                records, group_key, "u_a", "u_b",
                n_bootstrap=n_bootstrap, seed=seed)
            delta = bs["mean_delta"]
            if delta > gap_threshold and bs["ci_low"] > 0:
                verdict = "A_wins"
            elif -delta > gap_threshold and bs["ci_high"] < 0:
                verdict = "B_wins"
            else:
                verdict = "tie"
            pairwise.append(PolicyComparisonResult(
                policy_a=pa, policy_b=pb,
                mean_delta_a_minus_b=delta,
                ci_low=bs["ci_low"], ci_high=bs["ci_high"],
                verdict=verdict, n_groups=bs["n_groups"],
                n_records=bs["n_records"]))

    # Determine best policy decisively.
    best_pid = max(rows, key=lambda r: r.mean_utility).policy_id
    best_is_decisive = True
    for p in pairwise:
        if p.policy_a == best_pid and p.verdict != "A_wins":
            best_is_decisive = False
        if p.policy_b == best_pid and p.verdict != "B_wins":
            best_is_decisive = False
    notes = []
    if not best_is_decisive:
        notes.append(
            f"best policy {best_pid!r} is not decisively ahead of all "
            f"others (at least one pairwise comparison is a tie or loss)")
    return PolicyComparisonReport(
        rows=rows, pairwise=pairwise,
        best_policy_id=best_pid,
        best_is_decisive=best_is_decisive,
        gap_threshold=gap_threshold,
        notes=notes)


__all__ = [
    "PolicyComparisonReport",
    "PolicyComparisonResult",
    "PolicyComparisonRow",
    "compare_policies",
]
