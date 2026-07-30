"""v0.3.10.3.2-alpha — group-aware statistics (Sections 9, 31).

Even with no exact duplicates, tasks may share templates or latent
generators. Standard bootstrap that resamples individual tasks
overstates the effective sample size when many tasks are near
duplicates. This module provides grouped bootstrap that resamples
groups (templates/families) instead of individual tasks.

Also provides effective sample size (ESS) for weighted estimators
(Section 31) and weight diagnostics.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def grouped_bootstrap_mean_delta(
    records: Sequence[Mapping[str, Any]],
    group_key: str,
    value_a_key: str,
    value_b_key: str,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Grouped bootstrap of ``mean(value_a) - mean(value_b)`` (Section 9).

    Resamples GROUPS (identified by ``group_key``) with replacement,
    not individual records, so that near-duplicate tasks sharing a
    template/generator are kept together. This avoids overstating N.

    Parameters
    ----------
    records : sequence of mappings
        Each record must have ``group_key``, ``value_a_key``,
        ``value_b_key``.
    group_key : str
        Field identifying the group (e.g. ``"template_id"``,
        ``"group_id"``).
    value_a_key, value_b_key : str
        Fields whose mean difference is bootstrapped (e.g. utility
        under policy A vs policy B).
    n_bootstrap : int
    seed : int

    Returns
    -------
    dict with ``mean_delta``, ``ci_low``, ``ci_high``, ``n_groups``,
    ``n_records``.
    """
    rng = np.random.default_rng(seed)
    # Group records by group_key.
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for r in records:
        g = r.get(group_key)
        groups.setdefault(g, []).append(r)
    group_list = list(groups.values())
    n_groups = len(group_list)
    if n_groups == 0:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "n_groups": 0, "n_records": 0}
    a_vals = np.array([float(r[value_a_key]) for r in records])
    b_vals = np.array([float(r[value_b_key]) for r in records])
    mean_delta = float(a_vals.mean() - b_vals.mean())
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_groups, size=n_groups)
        a_sample = []
        b_sample = []
        for j in idx:
            for r in group_list[j]:
                a_sample.append(float(r[value_a_key]))
                b_sample.append(float(r[value_b_key]))
        deltas[i] = np.mean(a_sample) - np.mean(b_sample)
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    return {
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_groups": n_groups,
        "n_records": len(records),
    }


def effective_sample_size(weights: np.ndarray | Sequence[float]) -> float:
    """Weighted effective sample size (Section 31).

    ``ESS = (sum w_i)^2 / sum w_i^2``

    Exposes cases where weighting leaves only a tiny effective dataset.
    Returns 0.0 for an all-zero weight array.
    """
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    sq = float((w ** 2).sum())
    if sq <= 0.0:
        return 0.0
    return total * total / sq


def weight_diagnostics(weights: np.ndarray | Sequence[float]) -> dict[str, float]:
    """Per-policy weight diagnostics (Section 6).

    Returns effective sample weight sum, min, max, mean, zero-weight
    fraction, and ESS.
    """
    w = np.asarray(weights, dtype=np.float64)
    n = w.shape[0]
    return {
        "n": int(n),
        "weight_sum": float(w.sum()),
        "min_weight": float(w.min()) if n else 0.0,
        "max_weight": float(w.max()) if n else 0.0,
        "mean_weight": float(w.mean()) if n else 0.0,
        "zero_weight_fraction": float((w == 0.0).mean()) if n else 0.0,
        "ess": effective_sample_size(w),
    }


__all__ = [
    "cluster_robust_mean_test",
    "effective_sample_size",
    "grouped_bootstrap_mean_delta",
    "grouped_bootstrap_utility",
    "grouped_permutation_test",
    "leave_one_group_out",
    "weight_diagnostics",
]


# ------------------------------------------------------------------
# Section 17: additional group-aware statistics
# ------------------------------------------------------------------

def grouped_bootstrap_utility(
    records: Sequence[Mapping[str, Any]],
    group_key: str,
    value_key: str,
    *,
    n_bootstrap: int = 20000,
    confidence_level: float = 0.95,
    seed: int = 0,
    estimand: str = "group_weighted",
) -> dict[str, Any]:
    """Section 17 — grouped bootstrap of mean utility with labeled CI.

    Parameters
    ----------
    estimand : str
        ``"group_weighted"`` (primary, default) — each group gets equal
        weight regardless of size. ``"task_weighted"`` (secondary) —
        each task gets equal weight.
    """
    rng = np.random.default_rng(seed)
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for r in records:
        g = r.get(group_key)
        groups.setdefault(g, []).append(r)
    group_list = list(groups.values())
    n_groups = len(group_list)
    if n_groups == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "n_groups": 0, "n_records": 0, "estimand": estimand}

    if estimand == "group_weighted":
        # Each group's mean, then average across groups.
        group_means = np.array([
            np.mean([float(r[value_key]) for r in g]) for g in group_list])
        point_estimate = float(group_means.mean())
        deltas = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n_groups, size=n_groups)
            deltas[i] = group_means[idx].mean()
    else:
        # Task-weighted: pool all tasks, resample groups.
        all_vals = np.array([float(r[value_key]) for r in records])
        point_estimate = float(all_vals.mean())
        deltas = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n_groups, size=n_groups)
            sample = []
            for j in idx:
                for r in group_list[j]:
                    sample.append(float(r[value_key]))
            deltas[i] = np.mean(sample)

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(deltas, 100 * alpha / 2))
    ci_high = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return {
        "mean": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_groups": n_groups,
        "n_records": len(records),
        "estimand": estimand,
        "confidence_level": confidence_level,
    }


def grouped_permutation_test(
    records: Sequence[Mapping[str, Any]],
    group_key: str,
    value_a_key: str,
    value_b_key: str,
    *,
    n_permutations: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Section 17 — grouped permutation test for mean(a) - mean(b).

    Permutes group labels (not individual records) to build the null
    distribution. Returns the observed delta, p-value, and null
    distribution summary.
    """
    rng = np.random.default_rng(seed)
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for r in records:
        g = r.get(group_key)
        groups.setdefault(g, []).append(r)
    group_list = list(groups.values())
    n_groups = len(group_list)
    if n_groups == 0:
        return {"observed_delta": 0.0, "p_value": 1.0, "n_groups": 0}

    # Observed delta.
    a_vals = np.array([float(r[value_a_key]) for r in records])
    b_vals = np.array([float(r[value_b_key]) for r in records])
    observed = float(a_vals.mean() - b_vals.mean())

    # Permutation: for each group, randomly swap a/b labels.
    null_deltas = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        a_sample = []
        b_sample = []
        for g in group_list:
            if rng.random() < 0.5:
                for r in g:
                    a_sample.append(float(r[value_a_key]))
                    b_sample.append(float(r[value_b_key]))
            else:
                for r in g:
                    a_sample.append(float(r[value_b_key]))
                    b_sample.append(float(r[value_a_key]))
        null_deltas[i] = np.mean(a_sample) - np.mean(b_sample)

    p_value = float(np.mean(np.abs(null_deltas) >= abs(observed)))
    return {
        "observed_delta": observed,
        "p_value": p_value,
        "n_groups": n_groups,
        "n_permutations": n_permutations,
        "null_mean": float(null_deltas.mean()),
        "null_std": float(null_deltas.std()),
    }


def leave_one_group_out(
    records: Sequence[Mapping[str, Any]],
    group_key: str,
    value_key: str,
    *,
    evaluate_fn=None,
) -> dict[str, Any]:
    """Section 17 — leave-one-group-out cross-validation.

    For each group, holds out that group and evaluates on it using a
    model trained on the remaining groups. Returns per-group results
    and the overall LOO mean.
    """
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for r in records:
        g = r.get(group_key)
        groups.setdefault(g, []).append(r)
    group_ids = list(groups.keys())
    n_groups = len(group_ids)
    if n_groups == 0:
        return {"loo_mean": 0.0, "per_group": {}, "n_groups": 0}

    per_group: dict[str, float] = {}
    for gid in group_ids:
        held_out = groups[gid]
        if evaluate_fn is None:
            # Default: just report the group's mean value.
            vals = [float(r[value_key]) for r in held_out]
            per_group[str(gid)] = float(np.mean(vals))
        else:
            train = [r for g, rs in groups.items() if g != gid for r in rs]
            per_group[str(gid)] = float(evaluate_fn(train, held_out))

    loo_mean = float(np.mean(list(per_group.values())))
    return {
        "loo_mean": loo_mean,
        "per_group": per_group,
        "n_groups": n_groups,
    }


def cluster_robust_mean_test(
    records: Sequence[Mapping[str, Any]],
    group_key: str,
    value_key: str,
    *,
    null_value: float = 0.0,
) -> dict[str, Any]:
    """Section 17 — cluster-robust standard error test.

    Uses the cluster-robust variance estimator (Liang-Zellner) to test
    whether the mean of ``value_key`` differs from ``null_value``.
    """
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for r in records:
        g = r.get(group_key)
        groups.setdefault(g, []).append(r)
    group_list = list(groups.values())
    n_groups = len(group_list)
    if n_groups == 0:
        return {"mean": 0.0, "se": 0.0, "t_stat": 0.0, "n_groups": 0}

    # Group means.
    group_means = np.array([
        np.mean([float(r[value_key]) for r in g]) for g in group_list])
    overall_mean = float(group_means.mean())

    # Cluster-robust SE: sqrt(var(group_means) / n_groups).
    se = float(np.std(group_means, ddof=1) / np.sqrt(n_groups))
    t_stat = (overall_mean - null_value) / se if se > 0 else 0.0

    return {
        "mean": overall_mean,
        "se": se,
        "t_stat": float(t_stat),
        "n_groups": n_groups,
        "null_value": null_value,
    }
