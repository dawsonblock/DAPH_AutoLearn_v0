"""DAPH v0.4 — Corrected qualification statistics.

This module implements the statistically valid qualification machinery
described in the B4 hardening protocol:

* **Group-local paired positive-group computation** (Section 6):
  For each evaluation group g, compute the paired delta
  Δ_g = mean_i [U_hidden,i - U_baseline,i] within the group.
  Do NOT compare a local group to a global baseline average.

* **Paired group bootstrap** (Section 7):
  Bootstrap groups (not arbitrary individual records).
  For each replicate: sample groups with replacement, include all tasks
  belonging to selected groups, compute mean(U_A - U_B).
  Report: mean delta, median delta, 2.5/97.5 percentiles, SE, P(Δ>0).

* **Matched sham evaluation** (Section 8):
  Generate multiple matched sham policies (50-100 seeds).
  For each sham: retain same train/dev/final tasks, retain identical
  features, destroy the state→utility mapping, train with same
  algorithm/hyperparameters, save per-task final utility.
  Perform group-aware paired comparisons between real hidden policy
  and each sham. Report: real hidden regret, sham mean/median/best
  regret, hidden-vs-sham paired delta, LCB95, P(hidden > sham).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Group-local paired positive-group computation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GroupResult:
    """Per-group paired comparison result."""

    group_id: str
    subtype: str
    n_tasks: int
    hidden_mean_utility: float
    baseline_mean_utility: float
    paired_delta: float
    delta_positive: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "subtype": self.subtype,
            "n_tasks": self.n_tasks,
            "hidden_mean_utility": self.hidden_mean_utility,
            "baseline_mean_utility": self.baseline_mean_utility,
            "paired_delta": self.paired_delta,
            "delta_positive": self.delta_positive,
        }


def compute_group_local_results(
    task_ids: Sequence[str],
    group_ids: Sequence[str],
    subtypes: Sequence[str],
    hidden_utilities: Sequence[float],
    baseline_utilities: Sequence[float],
) -> list[GroupResult]:
    """Compute group-local paired comparisons.

    For each evaluation group g::

        Δ_g = (1/|g|) Σ_{i∈g} [U_hidden,i - U_baseline,i]

    Then::

        positive_group_fraction = Σ_g 𝟙(Δ_g > 0) / G

    This does NOT compare a local group to a global baseline average.
    """
    n = len(task_ids)
    if n == 0:
        return []

    # Group tasks by group_id
    groups: dict[str, list[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(str(g), []).append(i)

    results: list[GroupResult] = []
    for g, indices in sorted(groups.items()):
        h_utils = [float(hidden_utilities[i]) for i in indices]
        b_utils = [float(baseline_utilities[i]) for i in indices]
        h_mean = float(np.mean(h_utils))
        b_mean = float(np.mean(b_utils))
        # Paired delta: mean of per-task differences within the group
        paired_diffs = [h - b for h, b in zip(h_utils, b_utils)]
        delta = float(np.mean(paired_diffs))
        results.append(GroupResult(
            group_id=g,
            subtype=str(subtypes[indices[0]]) if indices else "",
            n_tasks=len(indices),
            hidden_mean_utility=h_mean,
            baseline_mean_utility=b_mean,
            paired_delta=delta,
            delta_positive=delta > 0,
        ))

    return results


def positive_group_fraction(group_results: Sequence[GroupResult]) -> float:
    """Compute the fraction of groups with positive paired delta."""
    if not group_results:
        return 0.0
    positive = sum(1 for g in group_results if g.delta_positive)
    return positive / len(group_results)


def worst_group_delta(group_results: Sequence[GroupResult]) -> float:
    """Compute the minimum (worst) group-level paired delta."""
    if not group_results:
        return 0.0
    return min(g.paired_delta for g in group_results)


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Paired group bootstrap
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BootstrapResult:
    """Full bootstrap result for a paired policy comparison."""

    comparison: str  # e.g. "hidden_vs_bestfixed"
    n_replicates: int
    mean_delta: float
    median_delta: float
    lcb_95: float
    ucb_95: float
    std_error: float
    prob_positive: float
    point_estimate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison": self.comparison,
            "n_replicates": self.n_replicates,
            "point_estimate": self.point_estimate,
            "mean_delta": self.mean_delta,
            "median_delta": self.median_delta,
            "lcb_95": self.lcb_95,
            "ucb_95": self.ucb_95,
            "std_error": self.std_error,
            "prob_positive": self.prob_positive,
        }


def paired_group_bootstrap(
    utilities_a: np.ndarray,
    utilities_b: np.ndarray,
    group_ids: Sequence[str],
    *,
    comparison: str = "A_vs_B",
    n_replicates: int = 10000,
    seed: int = 20260731,
    estimand: str = "task_weighted",
) -> BootstrapResult:
    """Paired group-aware bootstrap of mean(U_A - U_B).

    For each bootstrap replicate b:
    1. Sample groups with replacement.
    2. Include all tasks belonging to selected groups.
    3. Calculate Δ_b over the resampled tasks.

    Two estimands are supported:

    ``task_weighted`` (default, recommended primary):
        Each replicate samples groups with replacement, includes every
        task belonging to each sampled group, and calculates the
        task-weighted mean paired utility delta. This preserves the
        task-level utility target while respecting grouped dependence.

    ``group_equal_weight``:
        Each replicate samples groups with replacement and calculates
        the unweighted mean of per-group mean deltas. This gives each
        group equal influence regardless of size. Use for research
        diagnostics only.

    Parameters
    ----------
    utilities_a, utilities_b : np.ndarray  [N]
        Per-task realized utilities for policy A and policy B.
    group_ids : sequence of str  [N]
        Group assignment for each task.
    comparison : str
        Name of the comparison for labeling.
    n_replicates : int
        Number of bootstrap replicates (default 10000, final ≥ 5000).
    seed : int
        Random seed.
    estimand : str
        ``"task_weighted"`` or ``"group_equal_weight"``.

    Returns
    -------
    BootstrapResult with mean, median, 2.5/97.5 percentiles, SE, P(Δ>0).
    """
    utilities_a = np.asarray(utilities_a, dtype=np.float64)
    utilities_b = np.asarray(utilities_b, dtype=np.float64)
    n = len(utilities_a)
    if n == 0 or n != len(utilities_b):
        return BootstrapResult(
            comparison=comparison, n_replicates=0,
            mean_delta=0.0, median_delta=0.0, lcb_95=0.0, ucb_95=0.0,
            std_error=0.0, prob_positive=0.0, point_estimate=0.0,
        )

    # Per-task paired differences
    diffs = utilities_a - utilities_b

    # Group tasks by group_id
    groups: dict[str, np.ndarray] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(str(g), []).append(i)
    group_keys = list(groups.keys())
    group_indices = [np.array(groups[g], dtype=int) for g in group_keys]
    group_sizes = np.array([len(idx) for idx in group_indices], dtype=np.float64)
    n_groups = len(group_keys)

    # Per-group mean differences
    group_diffs = np.array([
        float(np.mean(diffs[idx])) for idx in group_indices
    ])

    if estimand == "group_equal_weight":
        # Each group contributes equally
        point_estimate = float(np.mean(group_diffs))
    else:
        # task_weighted: weight each group by its size
        point_estimate = float(np.average(group_diffs, weights=group_sizes))

    # Bootstrap: sample groups with replacement
    rng = np.random.RandomState(seed)
    boot_deltas = np.empty(n_replicates, dtype=np.float64)

    for b in range(n_replicates):
        sampled = rng.choice(n_groups, size=n_groups, replace=True)
        if estimand == "group_equal_weight":
            boot_deltas[b] = np.mean(group_diffs[sampled])
        else:
            # task_weighted: weight by group sizes
            sampled_sizes = group_sizes[sampled]
            boot_deltas[b] = np.average(
                group_diffs[sampled], weights=sampled_sizes
            )

    return BootstrapResult(
        comparison=comparison,
        n_replicates=n_replicates,
        point_estimate=point_estimate,
        mean_delta=float(np.mean(boot_deltas)),
        median_delta=float(np.median(boot_deltas)),
        lcb_95=float(np.percentile(boot_deltas, 2.5)),
        ucb_95=float(np.percentile(boot_deltas, 97.5)),
        std_error=float(np.std(boot_deltas, ddof=1)) if n_replicates > 1 else 0.0,
        prob_positive=float(np.mean(boot_deltas > 0)),
    )


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Matched sham evaluation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ShamComparisonResult:
    """Result of comparing a real policy against matched shams."""

    n_shams: int
    real_hidden_regret: float
    sham_mean_regret: float
    sham_median_regret: float
    sham_best_regret: float  # lowest (best) sham regret
    sham_worst_regret: float
    # Paired comparison: hidden - sham (per-task utility difference)
    hidden_vs_sham_paired_delta: float
    hidden_vs_sham_lcb95: float
    hidden_vs_sham_ucb95: float
    prob_hidden_gt_sham: float
    # Per-sham regrets
    sham_regrets: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_shams": self.n_shams,
            "real_hidden_regret": self.real_hidden_regret,
            "sham_mean_regret": self.sham_mean_regret,
            "sham_median_regret": self.sham_median_regret,
            "sham_best_regret": self.sham_best_regret,
            "sham_worst_regret": self.sham_worst_regret,
            "hidden_vs_sham_paired_delta": self.hidden_vs_sham_paired_delta,
            "hidden_vs_sham_lcb95": self.hidden_vs_sham_lcb95,
            "hidden_vs_sham_ucb95": self.hidden_vs_sham_ucb95,
            "prob_hidden_gt_sham": self.prob_hidden_gt_sham,
            "sham_regrets": list(self.sham_regrets),
        }


def create_matched_sham_utilities(
    utilities: np.ndarray,
    subtypes: Sequence[str],
    split_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Create sham utilities by shuffling within subtype × split bins.

    This destroys the mapping between features and action advantage
    while preserving the marginal distribution of utilities within
    each bin.

    Parameters
    ----------
    utilities : np.ndarray  [N, n_actions]
        Real counterfactual utilities.
    subtypes : sequence of str  [N]
    split_ids : np.ndarray  [N]
    seed : int
    """
    rng = np.random.RandomState(seed)
    sham = utilities.copy()
    n = len(utilities)

    bins: dict[tuple[str, int], list[int]] = {}
    for i in range(n):
        key = (str(subtypes[i]), int(split_ids[i]))
        bins.setdefault(key, []).append(i)

    for key, indices in bins.items():
        if len(indices) <= 1:
            continue
        perm = rng.permutation(len(indices))
        sham[indices] = utilities[[indices[p] for p in perm]]

    return sham


def run_matched_sham_evaluation(
    train_features: np.ndarray,
    train_utilities: np.ndarray,
    test_features: np.ndarray,
    test_utilities: np.ndarray,
    test_group_ids: Sequence[str],
    train_subtypes: Sequence[str],
    test_subtypes: Sequence[str],
    train_split_ids: np.ndarray,
    test_split_ids: np.ndarray,
    real_hidden_predictions: np.ndarray,
    policy_cls,
    policy_kwargs: dict | None = None,
    *,
    n_shams: int = 50,
    seed_base: int = 10000,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 20260731,
) -> ShamComparisonResult:
    """Run matched sham evaluation with paired comparison.

    For each sham:
    1. Retain exactly the same train/dev/final tasks.
    2. Retain identical features.
    3. Destroy the meaningful mapping between state and utility.
    4. Train with the same algorithm and hyperparameters.
    5. Save per-task final utility.

    Then perform group-aware paired comparisons between real hidden
    policy and each sham.

    Parameters
    ----------
    real_hidden_predictions : np.ndarray  [N_test]
        Action indices selected by the real hidden policy on the test set.
    """
    from daph_learning.executive.q_policy import compute_regret, mean_regret

    if policy_kwargs is None:
        policy_kwargs = {}

    n_test = len(test_utilities)
    action_ids = policy_kwargs.get(
        "action_ids",
        [f"a{i}" for i in range(train_utilities.shape[1])],
    )

    # Real hidden policy: per-task realized utility and regret
    real_hidden_utils = test_utilities[
        np.arange(n_test), real_hidden_predictions
    ]
    real_regret = mean_regret(real_hidden_predictions, test_utilities)

    # Oracle utility per task
    oracle_utils = test_utilities.max(axis=1)

    # Run shams
    sham_regrets: list[float] = []
    # For paired comparison, we need per-task utility for each sham
    sham_per_task_utils: list[np.ndarray] = []

    for s in range(n_shams):
        seed = seed_base + s
        sham_train_utils = create_matched_sham_utilities(
            train_utilities, train_subtypes, train_split_ids, seed
        )

        policy = policy_cls(**policy_kwargs)
        policy.fit(train_features, sham_train_utils)
        sham_preds = policy.predict(test_features)

        sham_utils = test_utilities[np.arange(n_test), sham_preds]
        sham_per_task_utils.append(sham_utils)
        sham_regrets.append(mean_regret(sham_preds, test_utilities))

    sham_regrets_arr = np.array(sham_regrets)

    # Paired comparison: hidden vs sham
    # For each sham, compute the paired delta (hidden_util - sham_util)
    # per task, then bootstrap at the group level.
    # We aggregate across all shams: for each task, compute the mean
    # sham utility, then compare hidden vs mean_sham.
    mean_sham_utils = np.mean(sham_per_task_utils, axis=0)

    boot = paired_group_bootstrap(
        real_hidden_utils,
        mean_sham_utils,
        test_group_ids,
        comparison="hidden_vs_sham",
        n_replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    return ShamComparisonResult(
        n_shams=n_shams,
        real_hidden_regret=float(real_regret),
        sham_mean_regret=float(sham_regrets_arr.mean()),
        sham_median_regret=float(np.median(sham_regrets_arr)),
        sham_best_regret=float(sham_regrets_arr.min()),
        sham_worst_regret=float(sham_regrets_arr.max()),
        hidden_vs_sham_paired_delta=boot.point_estimate,
        hidden_vs_sham_lcb95=boot.lcb_95,
        hidden_vs_sham_ucb95=boot.ucb_95,
        prob_hidden_gt_sham=boot.prob_positive,
        sham_regrets=sham_regrets_arr.tolist(),
    )


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Gap capture metric
# ──────────────────────────────────────────────────────────────────────

def gap_capture(
    policy_utility: float,
    best_fixed_utility: float,
    oracle_utility: float,
) -> float:
    """Compute oracle gap capture.

    GapCapture = (U_π - U_bestfixed) / (U_oracle - U_bestfixed)

    Returns 0.0 if the denominator is not positive.
    """
    denom = oracle_utility - best_fixed_utility
    if denom <= 1e-10:
        return 0.0
    return (policy_utility - best_fixed_utility) / denom


# ──────────────────────────────────────────────────────────────────────
# Section 5 — Selection accuracy
# ──────────────────────────────────────────────────────────────────────

def selection_accuracy(
    predicted_actions: np.ndarray,
    oracle_actions: np.ndarray,
) -> float:
    """Fraction of tasks where chosen action equals oracle action."""
    if len(predicted_actions) == 0:
        return 0.0
    return float(np.mean(predicted_actions == oracle_actions))


# ──────────────────────────────────────────────────────────────────────
# Section 6 — Abstention / margin analysis
# ──────────────────────────────────────────────────────────────────────

def margin_analysis(
    predicted_q: np.ndarray,
    oracle_actions: np.ndarray,
    utilities: np.ndarray,
    *,
    n_buckets: int = 5,
) -> dict[str, Any]:
    """Analyze policy performance as a function of Q-margin.

    For each task, margin = Q_(1) - Q_(2) (top minus second).

    Reports:
    * accuracy of selected action vs oracle action by margin bucket
    * regret by confidence bucket
    * utility gain by confidence bucket
    """
    n = len(predicted_q)
    if n == 0:
        return {"buckets": []}

    # Sort by margin (descending = most confident first)
    sorted_q = np.sort(predicted_q, axis=1)
    margins = sorted_q[:, -1] - sorted_q[:, -2] if predicted_q.shape[1] > 1 else \
        np.ones(n)
    predicted_actions = np.argmax(predicted_q, axis=1)

    # Per-task metrics
    oracle_utils = utilities.max(axis=1)
    selected_utils = utilities[np.arange(n), predicted_actions]
    regrets = oracle_utils - selected_utils
    correct = (predicted_actions == oracle_actions).astype(float)

    # Bucket by margin
    sorted_idx = np.argsort(margins)
    bucket_size = max(1, n // n_buckets)
    buckets = []
    for b in range(n_buckets):
        start = b * bucket_size
        end = (b + 1) * bucket_size if b < n_buckets - 1 else n
        idx = sorted_idx[start:end]
        if len(idx) == 0:
            continue
        buckets.append({
            "bucket": b,
            "n_tasks": len(idx),
            "margin_range": [
                float(margins[idx[0]]),
                float(margins[idx[-1]]),
            ],
            "mean_margin": float(np.mean(margins[idx])),
            "selection_accuracy": float(np.mean(correct[idx])),
            "mean_regret": float(np.mean(regrets[idx])),
            "mean_utility": float(np.mean(selected_utils[idx])),
        })

    return {"buckets": buckets}


# ──────────────────────────────────────────────────────────────────────
# Section 7 — Canonical PairedPolicyComparison
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairedPolicyComparison:
    """Canonical comparison between two policies.

    All policy comparisons (hidden vs fixed, hidden vs surface,
    hidden vs sham, ridge vs linear, etc.) must use this common
    representation so that different experiments cannot quietly
    define confidence differently.
    """

    policy_a: str
    policy_b: str
    point_delta: float
    bootstrap_mean: float
    bootstrap_median: float
    lcb95: float
    ucb95: float
    standard_error: float
    probability_positive: float
    positive_group_fraction: float
    worst_group_delta: float
    group_count: int
    task_count: int
    estimand: str = "task_weighted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "point_delta": self.point_delta,
            "bootstrap_mean": self.bootstrap_mean,
            "bootstrap_median": self.bootstrap_median,
            "lcb95": self.lcb95,
            "ucb95": self.ucb95,
            "standard_error": self.standard_error,
            "probability_positive": self.probability_positive,
            "positive_group_fraction": self.positive_group_fraction,
            "worst_group_delta": self.worst_group_delta,
            "group_count": self.group_count,
            "task_count": self.task_count,
            "estimand": self.estimand,
        }


def make_paired_comparison(
    policy_a: str,
    policy_b: str,
    utilities_a: np.ndarray,
    utilities_b: np.ndarray,
    group_ids: Sequence[str],
    subtypes: Sequence[str] | None = None,
    *,
    n_replicates: int = 10000,
    seed: int = 20260731,
    estimand: str = "task_weighted",
) -> PairedPolicyComparison:
    """Build a canonical PairedPolicyComparison from per-task utilities.

    This is the single entry point for all policy-vs-policy comparisons.
    It runs the paired group bootstrap and computes group-local
    statistics in one call.
    """
    boot = paired_group_bootstrap(
        utilities_a, utilities_b, group_ids,
        comparison=f"{policy_a}_vs_{policy_b}",
        n_replicates=n_replicates,
        seed=seed,
        estimand=estimand,
    )
    if subtypes is None:
        subtypes = ["unknown"] * len(utilities_a)
    group_results = compute_group_local_results(
        task_ids=[f"t{i}" for i in range(len(utilities_a))],
        group_ids=group_ids,
        subtypes=subtypes,
        hidden_utilities=utilities_a,
        baseline_utilities=utilities_b,
    )
    return PairedPolicyComparison(
        policy_a=policy_a,
        policy_b=policy_b,
        point_delta=boot.point_estimate,
        bootstrap_mean=boot.mean_delta,
        bootstrap_median=boot.median_delta,
        lcb95=boot.lcb_95,
        ucb95=boot.ucb_95,
        standard_error=boot.std_error,
        probability_positive=boot.prob_positive,
        positive_group_fraction=positive_group_fraction(group_results),
        worst_group_delta=worst_group_delta(group_results),
        group_count=len(group_results),
        task_count=len(utilities_a),
        estimand=estimand,
    )


# ──────────────────────────────────────────────────────────────────────
# Section 8 — Action advantage margin
# ──────────────────────────────────────────────────────────────────────

def action_advantage_margin(utilities: np.ndarray) -> dict[str, Any]:
    """Compute the per-task advantage margin M_i = U_best - U_second.

    Reports distribution statistics and threshold fractions.
    """
    n = len(utilities)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0,
                "frac_gt_0.01": 0.0, "frac_gt_0.02": 0.0,
                "frac_gt_0.05": 0.0, "frac_gt_0.10": 0.0}
    sorted_u = np.sort(utilities, axis=1)
    if utilities.shape[1] < 2:
        margins = np.zeros(n)
    else:
        margins = sorted_u[:, -1] - sorted_u[:, -2]
    return {
        "mean": float(np.mean(margins)),
        "median": float(np.median(margins)),
        "p25": float(np.percentile(margins, 25)),
        "p75": float(np.percentile(margins, 75)),
        "frac_gt_0.01": float(np.mean(margins > 0.01)),
        "frac_gt_0.02": float(np.mean(margins > 0.02)),
        "frac_gt_0.05": float(np.mean(margins > 0.05)),
        "frac_gt_0.10": float(np.mean(margins > 0.10)),
    }


__all__ = [
    "GroupResult",
    "compute_group_local_results",
    "positive_group_fraction",
    "worst_group_delta",
    "BootstrapResult",
    "paired_group_bootstrap",
    "PairedPolicyComparison",
    "make_paired_comparison",
    "ShamComparisonResult",
    "create_matched_sham_utilities",
    "run_matched_sham_evaluation",
    "gap_capture",
    "selection_accuracy",
    "margin_analysis",
    "action_advantage_margin",
]
