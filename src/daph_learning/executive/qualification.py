"""DAPH v0.4 — Generic executive qualification machinery.

This module generalizes the v0.3.x Gate A qualification to work with
arbitrary N-action executive qualification. The key generalizations:

1. **Oracle**: ``a*(s) = argmax_a U(s, a)`` over any action set
2. **Regret**: ``R(s, a) = U(s, a*) - U(s, a)`` for any chosen action
3. **Sham**: permute per-action utility vectors within bins, preserving
   the utility distribution while breaking the feature→utility link
4. **Bootstrap**: group-aware bootstrap over per-action realized utility
5. **Policy comparison**: P1 vs P0 vs always-action vs oracle

The existing v0.3.x binary qualification (qualification.py) remains
for backward compatibility. This module provides the generic equivalent.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from daph_learning.executive.types import (
    ActionSpace,
    CounterfactualSet,
    UtilityModel,
    ActionDecision,
    Regret,
    UtilityBreakdown,
)


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Per-Task Qualification Record
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutiveTaskRecord:
    """Canonical final-evaluation row for one task in a generic experiment.

    This generalizes ``FinalTaskRecord`` (which had ``symbolic_utility``,
    ``llm_utility``, ``symbolic_probability``, etc.) to arbitrary actions.

    Attributes
    ----------
    task_id : str
    group_id : str
    subtype : str
    split : str
    utilities : Mapping[str, float]
        Per-action utility ``U(s, a)`` for each candidate action.
    probabilities : Mapping[str, float]
        Per-action probability from the policy.
    selected_action : str
        The action the policy selected.
    oracle_action : str
        The action with the highest utility (argmax).
    p1_realized_utility : float
        Utility of the policy's selected action.
    p0_realized_utility : float
        Utility of the baseline (P0) selected action.
    oracle_utility : float
        Utility of the oracle action.
    regret : float
        ``oracle_utility - p1_realized_utility``
    """

    task_id: str
    group_id: str
    subtype: str
    split: str
    utilities: Mapping[str, float]
    probabilities: Mapping[str, float]
    selected_action: str
    oracle_action: str
    p1_realized_utility: float
    p0_realized_utility: float
    oracle_utility: float
    regret: float

    def utility(self, action_id: str) -> float:
        return self.utilities.get(action_id, 0.0)

    def probability(self, action_id: str) -> float:
        return self.probabilities.get(action_id, 0.0)

    @property
    def utility_gap(self) -> float:
        """Max utility minus the selected action's utility."""
        max_u = max(self.utilities.values()) if self.utilities else 0.0
        return max_u - self.p1_realized_utility

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "group_id": self.group_id,
            "subtype": self.subtype,
            "split": self.split,
            "utilities": dict(self.utilities),
            "probabilities": dict(self.probabilities),
            "selected_action": self.selected_action,
            "oracle_action": self.oracle_action,
            "p1_realized_utility": self.p1_realized_utility,
            "p0_realized_utility": self.p0_realized_utility,
            "oracle_utility": self.oracle_utility,
            "regret": self.regret,
        }


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Oracle Computation
# ──────────────────────────────────────────────────────────────────────

def compute_oracle_action(
    cf_set: CounterfactualSet,
    utility_model: UtilityModel,
) -> tuple[str, float]:
    """Compute the oracle action: ``a*(s) = argmax_a U(s, a)``."""
    return utility_model.best_action(cf_set)


def compute_oracle_utility(
    cf_sets: Sequence[CounterfactualSet],
    utility_model: UtilityModel,
) -> float:
    """Mean oracle utility across all tasks."""
    if not cf_sets:
        return 0.0
    total = 0.0
    for cf in cf_sets:
        _, best_u = utility_model.best_action(cf)
        total += best_u
    return total / len(cf_sets)


def compute_always_action_utility(
    cf_sets: Sequence[CounterfactualSet],
    action_id: str,
    utility_model: UtilityModel,
) -> float:
    """Mean utility of always selecting ``action_id``."""
    if not cf_sets:
        return 0.0
    total = 0.0
    for cf in cf_sets:
        breakdowns = utility_model.compute_all(cf)
        total += breakdowns[action_id].utility
    return total / len(cf_sets)


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Group-Aware Bootstrap
# ──────────────────────────────────────────────────────────────────────

def group_aware_bootstrap(
    records: Sequence[ExecutiveTaskRecord],
    *,
    bootstrap_iterations: int = 20000,
    seed: int = 20260731,
    group_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Group-aware bootstrap of P1 - P0 mean utility difference.

    Resamples at the group level (not the task level) to respect
    within-group correlation. Returns point estimate, LCB, UCB.

    Returns
    -------
    dict with keys: point, lcb_95, ucb_95, bootstrap_mean, bootstrap_std
    """
    if not records:
        return {"point": 0.0, "lcb_95": 0.0, "ucb_95": 0.0,
                "bootstrap_mean": 0.0, "bootstrap_std": 0.0}

    if group_ids is None:
        group_ids = [r.group_id for r in records]

    # Group records by group_id
    groups: dict[str, list[ExecutiveTaskRecord]] = {}
    for r, g in zip(records, group_ids):
        groups.setdefault(g, []).append(r)

    group_keys = list(groups.keys())
    n_groups = len(group_keys)

    # Per-group mean (P1 - P0)
    group_diffs = np.array([
        np.mean([r.p1_realized_utility - r.p0_realized_utility for r in groups[g]])
        for g in group_keys
    ])

    point = float(np.mean(group_diffs))

    rng = np.random.RandomState(seed)
    boot_means = np.empty(bootstrap_iterations)
    for i in range(bootstrap_iterations):
        sampled = rng.choice(n_groups, size=n_groups, replace=True)
        boot_means[i] = np.mean(group_diffs[sampled])

    lcb = float(np.percentile(boot_means, 2.5))
    ucb = float(np.percentile(boot_means, 97.5))

    return {
        "point": point,
        "lcb_95": lcb,
        "ucb_95": ucb,
        "bootstrap_mean": float(np.mean(boot_means)),
        "bootstrap_std": float(np.std(boot_means)),
    }


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Generic Sham Control
# ──────────────────────────────────────────────────────────────────────

def permute_action_utilities_within_bins(
    cf_sets: Sequence[CounterfactualSet],
    utility_model: UtilityModel,
    subtypes: Sequence[str],
    splits: Sequence[str],
    *,
    seed: int,
    decisive: Sequence[bool] | None = None,
) -> list[dict[str, float]]:
    """Permute per-action utility vectors within subtype×split bins.

    This is the generic sham control: it destroys the feature→utility
    association while preserving the utility distribution within each bin.

    For each task, the per-action utility vector ``{a: U(s, a)}`` is
    permuted across tasks within the same bin. This preserves the marginal
    distribution of utilities while breaking the link to features.

    Returns
    -------
    list of per-task permuted utility dicts ``{action_id: utility}``
    """
    n = len(cf_sets)
    if n < 2:
        return [utility_model.compute_all(cf) for cf in cf_sets]

    if decisive is None:
        decisive = [True] * n

    # Compute original per-action utilities
    original = []
    for cf in cf_sets:
        breakdowns = utility_model.compute_all(cf)
        original.append({a: bd.utility for a, bd in breakdowns.items()})

    # Get action_ids (from first task)
    action_ids = list(original[0].keys())

    # Build bins (same fallback hierarchy as v0.3.x sham)
    def _bin_key(subtype, split, dec):
        return f"{subtype}|{split}|{'decisive' if dec else 'nondecisive'}"

    bins: dict[str, list[int]] = {}
    for key_fn in (
        lambda i: _bin_key(str(subtypes[i]), str(splits[i]), bool(decisive[i])),
        lambda i: f"{subtypes[i]}|{splits[i]}",
        lambda i: str(subtypes[i]),
        lambda i: f"{'decisive' if decisive[i] else 'nondecisive'}",
        lambda i: "global",
    ):
        bins = {}
        for i in range(n):
            bins.setdefault(key_fn(i), []).append(i)
        if all(len(v) >= 2 for v in bins.values()):
            break

    # Permute within bins
    rng = np.random.RandomState(seed)
    permuted = [dict(u) for u in original]  # copy
    for key, indices in bins.items():
        if len(indices) <= 1:
            continue
        # Permute the utility vectors across tasks in this bin
        perm_idx = rng.permutation(len(indices))
        for j, idx in enumerate(indices):
            permuted[idx] = dict(original[indices[perm_idx[j]]])

    return permuted


# ──────────────────────────────────────────────────────────────────────
# Section 5 — Qualification Result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExecutiveQualificationResult:
    """Full qualification result for a generic executive experiment.

    This generalizes the v0.3.x Gate A result to arbitrary actions.
    """

    experiment_id: str
    action_space: ActionSpace
    n_tasks: int
    n_groups: int

    # Primary endpoint: P1 - P0
    p1_minus_p0: dict[str, float] = field(default_factory=dict)

    # Sham comparison: P1 - sham
    p1_minus_sham: dict[str, float] = field(default_factory=dict)

    # Oracle gap
    oracle_gap: float = 0.0
    oracle_gap_capture: float = 0.0  # (P1 - P0) / (oracle - P0)

    # Per-action utilities
    p1_mean_utility: float = 0.0
    p0_mean_utility: float = 0.0
    oracle_mean_utility: float = 0.0
    always_action_utilities: dict[str, float] = field(default_factory=dict)

    # Group-level positive fraction
    positive_group_fraction: float = 0.0

    # Worst subtype regression
    worst_subtype_regression: float = 0.0

    # Gate decision
    gate_passed: bool = False
    gate_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "action_space": self.action_space.to_dict(),
            "n_tasks": self.n_tasks,
            "n_groups": self.n_groups,
            "p1_minus_p0": self.p1_minus_p0,
            "p1_minus_sham": self.p1_minus_sham,
            "oracle_gap": self.oracle_gap,
            "oracle_gap_capture": self.oracle_gap_capture,
            "p1_mean_utility": self.p1_mean_utility,
            "p0_mean_utility": self.p0_mean_utility,
            "oracle_mean_utility": self.oracle_mean_utility,
            "always_action_utilities": dict(self.always_action_utilities),
            "positive_group_fraction": self.positive_group_fraction,
            "worst_subtype_regression": self.worst_subtype_regression,
            "gate_passed": self.gate_passed,
            "gate_failures": list(self.gate_failures),
        }


def evaluate_qualification(
    records: Sequence[ExecutiveTaskRecord],
    action_space: ActionSpace,
    *,
    experiment_id: str,
    bootstrap_iterations: int = 20000,
    bootstrap_seed: int = 20260731,
    n_sham_seeds: int = 20,
    sham_seed_base: int = 10000,
) -> ExecutiveQualificationResult:
    """Compute the full qualification result from task records.

    This is the generic equivalent of the v0.3.x Gate A evaluation.
    """
    n_tasks = len(records)
    if n_tasks == 0:
        return ExecutiveQualificationResult(
            experiment_id=experiment_id,
            action_space=action_space,
            n_tasks=0, n_groups=0,
        )

    group_ids = [r.group_id for r in records]
    n_groups = len(set(group_ids))

    # P1 - P0 bootstrap
    p1_p0 = group_aware_bootstrap(
        records,
        bootstrap_iterations=bootstrap_iterations,
        seed=bootstrap_seed,
        group_ids=group_ids,
    )

    # P1 mean utility
    p1_mean = float(np.mean([r.p1_realized_utility for r in records]))
    p0_mean = float(np.mean([r.p0_realized_utility for r in records]))
    oracle_mean = float(np.mean([r.oracle_utility for r in records]))

    # Oracle gap
    oracle_gap = oracle_mean - p0_mean
    p1_p0_point = p1_p0["point"]
    oracle_p0_gap = oracle_mean - p0_mean
    oracle_gap_capture = p1_p0_point / oracle_p0_gap if oracle_p0_gap > 1e-10 else 0.0

    # Always-action utilities
    always_utils = {}
    for action_id in action_space.action_ids:
        always_utils[action_id] = float(np.mean([
            r.utility(action_id) for r in records]))

    # Positive group fraction
    groups: dict[str, list[ExecutiveTaskRecord]] = {}
    for r in records:
        groups.setdefault(r.group_id, []).append(r)
    positive_groups = sum(
        1 for g, recs in groups.items()
        if np.mean([r.p1_realized_utility - r.p0_realized_utility for r in recs]) > 0
    )
    positive_group_fraction = positive_groups / n_groups if n_groups > 0 else 0.0

    # Worst subtype regression
    subtypes: dict[str, list[ExecutiveTaskRecord]] = {}
    for r in records:
        subtypes.setdefault(r.subtype, []).append(r)
    worst_regression = 0.0
    for st, recs in subtypes.items():
        mean_diff = np.mean([r.p1_realized_utility - r.p0_realized_utility for r in recs])
        if mean_diff < worst_regression:
            worst_regression = float(mean_diff)

    return ExecutiveQualificationResult(
        experiment_id=experiment_id,
        action_space=action_space,
        n_tasks=n_tasks,
        n_groups=n_groups,
        p1_minus_p0=p1_p0,
        p1_mean_utility=p1_mean,
        p0_mean_utility=p0_mean,
        oracle_mean_utility=oracle_mean,
        oracle_gap=oracle_gap,
        oracle_gap_capture=oracle_gap_capture,
        always_action_utilities=always_utils,
        positive_group_fraction=positive_group_fraction,
        worst_subtype_regression=worst_regression,
    )
