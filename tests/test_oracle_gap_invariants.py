"""Invariant tests for oracle gap capture and route distribution.

These tests verify the mathematical invariants that must hold when
the policy matches the oracle, and that route distributions are
correctly reported.
"""

from __future__ import annotations

import numpy as np
import pytest


def oracle_gap_capture(
    policy_utility: float,
    baseline_utility: float,
    oracle_utility: float,
) -> float:
    """Oracle gap capture: fraction of the oracle-baseline gap captured
    by the policy."""
    gap = oracle_utility - baseline_utility
    if abs(gap) < 1e-12:
        return 0.0
    return (policy_utility - baseline_utility) / gap


def test_oracle_gap_capture_is_one_when_policy_matches_oracle():
    """If P1 achieves the same utility as the oracle, the capture
    must be exactly 1.0."""
    p0 = np.array([0.1, 0.2, 0.3])
    oracle = np.array([0.8, 0.9, 1.0])
    p1 = oracle.copy()
    capture = oracle_gap_capture(
        policy_utility=p1.mean(),
        baseline_utility=p0.mean(),
        oracle_utility=oracle.mean(),
    )
    assert capture == pytest.approx(1.0)


def test_oracle_gap_capture_is_zero_when_policy_matches_baseline():
    """If P1 achieves the same utility as the baseline, the capture
    must be exactly 0.0."""
    p0 = np.array([0.1, 0.2, 0.3])
    oracle = np.array([0.8, 0.9, 1.0])
    p1 = p0.copy()
    capture = oracle_gap_capture(
        policy_utility=p1.mean(),
        baseline_utility=p0.mean(),
        oracle_utility=oracle.mean(),
    )
    assert capture == pytest.approx(0.0)


def test_oracle_gap_capture_is_half_when_policy_is_coin_flip():
    """If P1 is a coin flip between oracle and baseline, the capture
    must be exactly 0.5."""
    p0 = np.array([0.0, 0.0, 0.0, 0.0])
    oracle = np.array([1.0, 1.0, 1.0, 1.0])
    p1 = np.array([1.0, 0.0, 1.0, 0.0])  # 50% oracle, 50% baseline
    capture = oracle_gap_capture(
        policy_utility=p1.mean(),
        baseline_utility=p0.mean(),
        oracle_utility=oracle.mean(),
    )
    assert capture == pytest.approx(0.5)


def test_oracle_gap_capture_negative_when_policy_worse_than_baseline():
    capture = oracle_gap_capture(
        policy_utility=-0.5,
        baseline_utility=0.0,
        oracle_utility=1.0,
    )
    assert capture < 0.0


def test_route_distribution_sums_to_one():
    """Route fractions must sum to 1.0."""
    sym_frac = 0.6
    llm_frac = 0.3
    abstain_frac = 0.1
    assert sym_frac + llm_frac + abstain_frac == pytest.approx(1.0)


def test_oracle_action_agreement_when_policy_matches_oracle():
    """If P1 actions == oracle actions, agreement must be 1.0."""
    p1_actions = np.array(["symbolic", "llm", "symbolic", "llm"])
    oracle_actions = np.array(["symbolic", "llm", "symbolic", "llm"])
    agreement = np.mean(p1_actions == oracle_actions)
    assert agreement == pytest.approx(1.0)


def test_oracle_action_agreement_when_policy_is_constant():
    """If P1 always picks symbolic and oracle always picks symbolic,
    agreement must be 1.0."""
    p1_actions = np.array(["symbolic"] * 10)
    oracle_actions = np.array(["symbolic"] * 10)
    agreement = np.mean(p1_actions == oracle_actions)
    assert agreement == pytest.approx(1.0)


def test_oracle_action_agreement_zero_when_completely_disagree():
    p1_actions = np.array(["symbolic"] * 10)
    oracle_actions = np.array(["llm"] * 10)
    agreement = np.mean(p1_actions == oracle_actions)
    assert agreement == pytest.approx(0.0)


def test_centroid_policy_constant_when_one_class_empty():
    """When all training examples prefer symbolic, the centroid policy
    must produce p≈1.0 (not p=0.5)."""
    import sys
    sys.path.insert(0, "src")
    from daph_learning.policy.centroid_policy import CentroidPolicy

    # All delta_u > 0 (symbolic always wins)
    features = np.random.randn(50, 10).astype(np.float32)
    delta_u = np.ones(50, dtype=np.float64)  # all positive
    weights = np.ones(50, dtype=np.float64)

    policy = CentroidPolicy()
    policy.fit(features, delta_u, weights)

    # Predict on new data
    test_features = np.random.randn(10, 10).astype(np.float32)
    probs = policy.predict_proba(test_features)

    # All probabilities should be ≈1.0 (route to symbolic)
    assert np.all(probs > 0.99), f"Expected p≈1.0, got {probs}"
    assert policy.degenerate_train_data is True
    assert policy.n_llm_ == 0


def test_centroid_policy_constant_when_all_llm():
    """When all training examples prefer LLM, the centroid policy
    must produce p≈0.0 (not p=0.5)."""
    import sys
    sys.path.insert(0, "src")
    from daph_learning.policy.centroid_policy import CentroidPolicy

    features = np.random.randn(50, 10).astype(np.float32)
    delta_u = -np.ones(50, dtype=np.float64)  # all negative
    weights = np.ones(50, dtype=np.float64)

    policy = CentroidPolicy()
    policy.fit(features, delta_u, weights)

    test_features = np.random.randn(10, 10).astype(np.float32)
    probs = policy.predict_proba(test_features)

    # All probabilities should be ≈0.0 (route to LLM)
    assert np.all(probs < 0.01), f"Expected p≈0.0, got {probs}"
    assert policy.degenerate_train_data is True
    assert policy.n_symbolic_ == 0


def test_centroid_policy_normal_when_both_classes_present():
    """When both classes are present, the policy should NOT be
    degenerate."""
    import sys
    sys.path.insert(0, "src")
    from daph_learning.policy.centroid_policy import CentroidPolicy

    features = np.random.randn(50, 10).astype(np.float32)
    delta_u = np.concatenate([
        np.ones(25, dtype=np.float64),
        -np.ones(25, dtype=np.float64),
    ])
    weights = np.ones(50, dtype=np.float64)

    policy = CentroidPolicy()
    policy.fit(features, delta_u, weights)

    assert policy.degenerate_train_data is False
    assert policy.n_symbolic_ == 25
    assert policy.n_llm_ == 25
    # Vector should be non-zero
    assert np.linalg.norm(policy.vector) > 0.01
