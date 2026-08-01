"""Tests for DAPH v0.4 generic executive qualification.

Verifies oracle computation, group-aware bootstrap, sham permutation,
and full qualification evaluation work for arbitrary N-action spaces.
"""

from __future__ import annotations

import pytest
import numpy as np

from daph_learning.executive import (
    ActionDescriptor,
    ActionSpace,
    ExecutiveState,
    ActionExecution,
    CounterfactualSet,
    UtilityModel,
    ExecutiveTaskRecord,
    compute_oracle_action,
    compute_oracle_utility,
    compute_always_action_utility,
    group_aware_bootstrap,
    permute_action_utilities_within_bins,
    evaluate_qualification,
    binary_action_space,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

def _make_cf_set(task_id, sym_correct, llm_correct, sym_fast=True):
    """Helper: build a binary counterfactual set."""
    state = ExecutiveState(
        task_id=task_id, prompt=f"prompt_{task_id}",
        task_metadata={"subtype": "A", "group_id": "g1", "split": "test"},
    )
    e_sym = ActionExecution(
        action_id="action.symbolic_arithmetic",
        selected=sym_correct, executed=True,
        verified_correct=sym_correct,
        verifier_name="numeric",
        latency_ms=10.0 if sym_fast else 100.0,
        compute_cost=0.05,
    )
    e_llm = ActionExecution(
        action_id="action.llm_direct",
        selected=not sym_correct, executed=True,
        verified_correct=llm_correct,
        verifier_name="numeric",
        latency_ms=500.0,
        compute_cost=0.20,
    )
    return CounterfactualSet(
        state=state,
        executions={
            "action.symbolic_arithmetic": e_sym,
            "action.llm_direct": e_llm,
        },
        selected_action="action.symbolic_arithmetic" if sym_correct else "action.llm_direct",
    )


def _make_records(n=20, n_groups=4, action_space=None):
    """Generate synthetic ExecutiveTaskRecords for testing."""
    if action_space is None:
        action_space = binary_action_space()
    sym_id = "action.symbolic_arithmetic"
    llm_id = "action.llm_direct"
    records = []
    rng = np.random.RandomState(42)
    for i in range(n):
        group_id = f"g{i % n_groups}"
        subtype = "A" if i % 2 == 0 else "B"
        # Make P1 better than P0 on average
        sym_u = rng.uniform(0.3, 0.9)
        llm_u = rng.uniform(0.2, 0.7)
        # P1 selects symbolic, P0 always selects llm
        p1_u = sym_u
        p0_u = llm_u
        oracle_u = max(sym_u, llm_u)
        oracle_a = sym_id if sym_u >= llm_u else llm_id
        records.append(ExecutiveTaskRecord(
            task_id=f"t{i}",
            group_id=group_id,
            subtype=subtype,
            split="test",
            utilities={sym_id: sym_u, llm_id: llm_u},
            probabilities={sym_id: 0.6, llm_id: 0.4},
            selected_action=sym_id,
            oracle_action=oracle_a,
            p1_realized_utility=p1_u,
            p0_realized_utility=p0_u,
            oracle_utility=oracle_u,
            regret=oracle_u - p1_u,
        ))
    return records


# ──────────────────────────────────────────────────────────────────────
# Oracle Tests
# ──────────────────────────────────────────────────────────────────────

class TestOracle:
    def test_oracle_selects_best(self):
        cf = _make_cf_set("t1", sym_correct=True, llm_correct=False)
        um = UtilityModel(lambda_time=0.0, lambda_compute=0.0, lambda_risk=0.0)
        best_id, best_u = compute_oracle_action(cf, um)
        assert best_id == "action.symbolic_arithmetic"
        assert best_u == pytest.approx(1.0)

    def test_oracle_selects_llm_when_better(self):
        cf = _make_cf_set("t1", sym_correct=False, llm_correct=True)
        um = UtilityModel(lambda_time=0.0, lambda_compute=0.0, lambda_risk=0.0)
        best_id, best_u = compute_oracle_action(cf, um)
        assert best_id == "action.llm_direct"
        assert best_u == pytest.approx(1.0)

    def test_oracle_utility_mean(self):
        cf_sets = [
            _make_cf_set("t1", sym_correct=True, llm_correct=False),
            _make_cf_set("t2", sym_correct=False, llm_correct=True),
        ]
        um = UtilityModel(lambda_time=0.0, lambda_compute=0.0, lambda_risk=0.0)
        mean_u = compute_oracle_utility(cf_sets, um)
        assert mean_u == pytest.approx(1.0)

    def test_always_action_utility(self):
        cf_sets = [
            _make_cf_set("t1", sym_correct=True, llm_correct=False),
            _make_cf_set("t2", sym_correct=False, llm_correct=True),
        ]
        um = UtilityModel(lambda_time=0.0, lambda_compute=0.0, lambda_risk=0.0)
        always_sym = compute_always_action_utility(cf_sets, "action.symbolic_arithmetic", um)
        always_llm = compute_always_action_utility(cf_sets, "action.llm_direct", um)
        assert always_sym == pytest.approx(0.5)
        assert always_llm == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────
# Bootstrap Tests
# ──────────────────────────────────────────────────────────────────────

class TestBootstrap:
    def test_basic_bootstrap(self):
        records = _make_records(n=20, n_groups=4)
        result = group_aware_bootstrap(records, bootstrap_iterations=1000, seed=42)
        assert "point" in result
        assert "lcb_95" in result
        assert "ucb_95" in result
        assert result["lcb_95"] <= result["point"] <= result["ucb_95"]

    def test_empty_records(self):
        result = group_aware_bootstrap([], bootstrap_iterations=100)
        assert result["point"] == 0.0

    def test_positive_p1_p0(self):
        records = _make_records(n=20, n_groups=4)
        # P1 (symbolic) is better than P0 (llm) by construction
        result = group_aware_bootstrap(records, bootstrap_iterations=1000, seed=42)
        assert result["point"] > 0


# ──────────────────────────────────────────────────────────────────────
# Sham Tests
# ──────────────────────────────────────────────────────────────────────

class TestSham:
    def test_permute_utilities_preserves_distribution(self):
        cf_sets = [_make_cf_set(f"t{i}", i % 2 == 0, i % 3 == 0) for i in range(20)]
        um = UtilityModel(lambda_time=0.01, lambda_compute=0.0, lambda_risk=0.0)
        subtypes = ["A" if i % 2 == 0 else "B" for i in range(20)]
        splits = ["test"] * 20

        permuted = permute_action_utilities_within_bins(
            cf_sets, um, subtypes, splits, seed=42)

        # The multiset of utilities should be preserved
        original_utils = []
        for cf in cf_sets:
            bds = um.compute_all(cf)
            original_utils.extend(bd.utility for bd in bds.values())

        permuted_utils = []
        for pu in permuted:
            permuted_utils.extend(pu.values())

        assert sorted(original_utils) == sorted(permuted_utils)

    def test_permute_breaks_feature_link(self):
        # With enough tasks, the permuted utilities should differ from original
        cf_sets = [_make_cf_set(f"t{i}", i % 2 == 0, i % 3 == 0) for i in range(20)]
        um = UtilityModel(lambda_time=0.01, lambda_compute=0.0, lambda_risk=0.0)
        subtypes = ["A"] * 20
        splits = ["test"] * 20

        permuted = permute_action_utilities_within_bins(
            cf_sets, um, subtypes, splits, seed=123)

        # At least some should differ
        n_diff = 0
        for cf, pu in zip(cf_sets, permuted):
            bds = um.compute_all(cf)
            for a in pu:
                if abs(bds[a].utility - pu[a]) > 1e-10:
                    n_diff += 1
                    break
        assert n_diff > 0, "permutation did not change anything"


# ──────────────────────────────────────────────────────────────────────
# Full Qualification Tests
# ──────────────────────────────────────────────────────────────────────

class TestQualification:
    def test_basic_evaluation(self):
        records = _make_records(n=40, n_groups=8)
        space = binary_action_space()
        result = evaluate_qualification(
            records, space,
            experiment_id="test_exp",
            bootstrap_iterations=1000,
        )
        assert result.n_tasks == 40
        assert result.n_groups == 8
        assert result.p1_mean_utility > 0
        assert result.p0_mean_utility > 0
        assert result.oracle_mean_utility >= result.p1_mean_utility
        assert "point" in result.p1_minus_p0
        assert "action.symbolic_arithmetic" in result.always_action_utilities

    def test_empty_records(self):
        space = binary_action_space()
        result = evaluate_qualification(
            [], space, experiment_id="empty")
        assert result.n_tasks == 0
        assert result.gate_passed is False

    def test_three_action_space(self):
        """Test with a 3-action space (not just binary)."""
        space = ActionSpace(actions=(
            ActionDescriptor(action_id="reasoning.direct", cost_estimate=0.1),
            ActionDescriptor(action_id="retrieval.vector", cost_estimate=0.05),
            ActionDescriptor(action_id="reasoning.decompose", cost_estimate=0.3),
        ))
        rng = np.random.RandomState(42)
        records = []
        for i in range(30):
            utils = {
                "reasoning.direct": rng.uniform(0.3, 0.9),
                "retrieval.vector": rng.uniform(0.2, 0.7),
                "reasoning.decompose": rng.uniform(0.1, 0.6),
            }
            selected = "reasoning.direct"
            oracle_a = max(utils, key=utils.get)
            records.append(ExecutiveTaskRecord(
                task_id=f"t{i}",
                group_id=f"g{i % 5}",
                subtype="A" if i % 2 == 0 else "B",
                split="test",
                utilities=utils,
                probabilities={"reasoning.direct": 0.5, "retrieval.vector": 0.3, "reasoning.decompose": 0.2},
                selected_action=selected,
                oracle_action=oracle_a,
                p1_realized_utility=utils[selected],
                p0_realized_utility=utils["retrieval.vector"],
                oracle_utility=utils[oracle_a],
                regret=utils[oracle_a] - utils[selected],
            ))
        result = evaluate_qualification(
            records, space,
            experiment_id="three_action_test",
            bootstrap_iterations=500,
        )
        assert result.n_tasks == 30
        assert len(result.always_action_utilities) == 3
        assert "reasoning.direct" in result.always_action_utilities
        assert "retrieval.vector" in result.always_action_utilities
        assert "reasoning.decompose" in result.always_action_utilities

    def test_to_dict(self):
        records = _make_records(n=10, n_groups=2)
        space = binary_action_space()
        result = evaluate_qualification(
            records, space,
            experiment_id="dict_test",
            bootstrap_iterations=100,
        )
        d = result.to_dict()
        assert d["experiment_id"] == "dict_test"
        assert d["n_tasks"] == 10
        assert "action_space" in d
        assert "p1_minus_p0" in d
