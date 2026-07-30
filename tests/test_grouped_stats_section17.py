"""Section 17 — group-aware statistical suite tests."""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.evaluation.grouped_stats import (
    cluster_robust_mean_test,
    grouped_bootstrap_mean_delta,
    grouped_bootstrap_utility,
    grouped_permutation_test,
    leave_one_group_out,
)


def _records(n_groups=10, n_per_group=5, seed=42):
    rng = np.random.default_rng(seed)
    records = []
    for g in range(n_groups):
        for i in range(n_per_group):
            records.append({
                "group_id": f"g{g}",
                "u_p1": float(rng.normal(0.1, 0.1)),
                "u_p0": float(rng.normal(0.0, 0.1)),
                "utility": float(rng.normal(0.1, 0.1)),
            })
    return records


def test_grouped_bootstrap_utility_group_weighted():
    records = _records()
    result = grouped_bootstrap_utility(
        records, "group_id", "utility",
        n_bootstrap=1000, estimand="group_weighted")
    assert "mean" in result
    assert "ci_low" in result
    assert "ci_high" in result
    assert result["estimand"] == "group_weighted"
    assert result["n_groups"] == 10


def test_grouped_bootstrap_utility_task_weighted():
    records = _records()
    result = grouped_bootstrap_utility(
        records, "group_id", "utility",
        n_bootstrap=1000, estimand="task_weighted")
    assert result["estimand"] == "task_weighted"


def test_grouped_bootstrap_ci_brackets_mean():
    records = _records()
    result = grouped_bootstrap_utility(
        records, "group_id", "utility", n_bootstrap=1000)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_grouped_permutation_test():
    records = _records()
    result = grouped_permutation_test(
        records, "group_id", "u_p1", "u_p0",
        n_permutations=500)
    assert "observed_delta" in result
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0


def test_leave_one_group_out():
    records = _records()
    result = leave_one_group_out(records, "group_id", "utility")
    assert result["n_groups"] == 10
    assert len(result["per_group"]) == 10
    assert "loo_mean" in result


def test_cluster_robust_mean_test():
    records = _records()
    result = cluster_robust_mean_test(records, "group_id", "utility")
    assert "mean" in result
    assert "se" in result
    assert "t_stat" in result
    assert result["n_groups"] == 10
    assert result["se"] > 0


def test_empty_records():
    result = grouped_bootstrap_utility([], "group_id", "utility")
    assert result["n_groups"] == 0
    assert result["mean"] == 0.0


def test_existing_grouped_bootstrap_mean_delta_still_works():
    records = _records()
    result = grouped_bootstrap_mean_delta(
        records, "group_id", "u_p1", "u_p0", n_bootstrap=100)
    assert "mean_delta" in result
    assert "ci_low" in result
