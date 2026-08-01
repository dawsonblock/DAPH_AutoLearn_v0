"""Tests for DAPH v0.4 executive experience builder and training targets."""

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
    ExecutiveExperience,
    build_executive_experiences,
    experiences_to_training_arrays,
    combine_action_confidences,
    ExecutiveTrainingTargets,
    build_executive_training_targets,
    estimate_uncertainty,
)


def _make_cf_sets(n=30, n_actions=3, seed=42):
    """Generate synthetic counterfactual sets."""
    rng = np.random.RandomState(seed)
    action_ids = [f"action_{i}" for i in range(n_actions)]
    space = ActionSpace(actions=tuple(
        ActionDescriptor(action_id=aid) for aid in action_ids))

    cf_sets = []
    for i in range(n):
        state = ExecutiveState(
            task_id=f"t{i}", prompt=f"prompt_{i}",
            task_metadata={"subtype": "A", "group_id": f"g{i%5}", "split": "test"},
        )
        executions = {}
        for j, aid in enumerate(action_ids):
            correct = (j == i % n_actions)
            executions[aid] = ActionExecution(
                action_id=aid,
                selected=(j == i % n_actions),
                executed=True,
                verified_correct=correct,
                verifier_name="numeric",
                latency_ms=100.0 + j * 200 + rng.randn() * 20,
                compute_cost=0.1 * (j + 1),
            )
        cf_sets.append(CounterfactualSet(
            state=state,
            executions=executions,
            selected_action=action_ids[i % n_actions],
        ))
    return cf_sets, space


class TestExecutiveExperience:
    def test_build_experiences(self):
        cf_sets, space = _make_cf_sets(n=20, n_actions=3)
        um = UtilityModel(lambda_time=0.01, lambda_compute=0.1, lambda_risk=1.0)
        experiences = build_executive_experiences(cf_sets, um, space)
        assert len(experiences) == 20
        for exp in experiences:
            assert isinstance(exp, ExecutiveExperience)
            assert exp.best_action in space.action_ids
            assert exp.best_utility >= -1.0
            assert exp.sample_weight == 1.0  # default

    def test_utility_vector(self):
        cf_sets, space = _make_cf_sets(n=5, n_actions=3)
        um = UtilityModel()
        experiences = build_executive_experiences(cf_sets, um, space)
        vec = experiences[0].utility_vector(space.action_ids)
        assert vec.shape == (3,)

    def test_custom_weight_fn(self):
        cf_sets, space = _make_cf_sets(n=10, n_actions=3)
        um = UtilityModel()
        experiences = build_executive_experiences(
            cf_sets, um, space,
            weight_fn=lambda cf, u: 2.0)
        assert all(e.sample_weight == 2.0 for e in experiences)

    def test_to_dict(self):
        cf_sets, space = _make_cf_sets(n=3, n_actions=2)
        um = UtilityModel()
        experiences = build_executive_experiences(cf_sets, um, space)
        d = experiences[0].to_dict()
        assert "state" in d
        assert "utilities" in d
        assert "best_action" in d


class TestTrainingArrays:
    def test_arrays_shape(self):
        cf_sets, space = _make_cf_sets(n=30, n_actions=3)
        um = UtilityModel()
        experiences = build_executive_experiences(cf_sets, um, space)
        feats = np.random.randn(30, 5).astype(np.float32)
        arrays = experiences_to_training_arrays(experiences, space, features=feats)
        assert arrays["utilities"].shape == (30, 3)
        assert arrays["weights"].shape == (30,)
        assert arrays["best_actions"].shape == (30,)
        assert arrays["features"].shape == (30, 5)

    def test_arrays_without_features(self):
        cf_sets, space = _make_cf_sets(n=10, n_actions=2)
        um = UtilityModel()
        experiences = build_executive_experiences(cf_sets, um, space)
        arrays = experiences_to_training_arrays(experiences, space)
        assert "features" not in arrays
        assert arrays["utilities"].shape == (10, 2)


class TestConfidenceCombination:
    def test_product_2actions(self):
        c = combine_action_confidences({"a": 0.8, "b": 0.6}, mode="product")
        assert c == pytest.approx(0.48)

    def test_product_3actions(self):
        c = combine_action_confidences({"a": 0.9, "b": 0.8, "c": 0.7}, mode="product")
        assert c == pytest.approx(0.504)

    def test_min(self):
        c = combine_action_confidences({"a": 0.8, "b": 0.6, "c": 0.9}, mode="min")
        assert c == pytest.approx(0.6)

    def test_geometric_mean(self):
        c = combine_action_confidences({"a": 0.8, "b": 0.2}, mode="geometric_mean")
        assert c == pytest.approx(np.sqrt(0.16))

    def test_empty(self):
        assert combine_action_confidences({}) == 0.0

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            combine_action_confidences({"a": 1.5})


class TestTrainingTargets:
    def test_build_targets(self):
        cf_sets, space = _make_cf_sets(n=20, n_actions=3)
        um = UtilityModel()
        experiences = build_executive_experiences(cf_sets, um, space)
        targets = build_executive_training_targets(experiences, space)
        assert targets.n_examples == 20
        assert targets.n_actions == 3
        assert targets.utilities.shape == (20, 3)
        assert targets.weights.shape == (20,)

    def test_abstention_band(self):
        cf_sets, space = _make_cf_sets(n=20, n_actions=3)
        um = UtilityModel()
        experiences = build_executive_experiences(cf_sets, um, space)
        # Large band → all ties → all weights 0
        targets = build_executive_training_targets(
            experiences, space, abstention_band=100.0)
        assert np.all(targets.weights == 0.0)

    def test_weight_mode_utility(self):
        cf_sets, space = _make_cf_sets(n=20, n_actions=3)
        um = UtilityModel()
        experiences = build_executive_experiences(cf_sets, um, space)
        targets = build_executive_training_targets(
            experiences, space, weight_mode="utility")
        # Weights should be non-negative
        assert np.all(targets.weights >= 0)

    def test_weight_mode_gap(self):
        cf_sets, space = _make_cf_sets(n=20, n_actions=3)
        um = UtilityModel(lambda_time=0.01)
        experiences = build_executive_experiences(cf_sets, um, space)
        targets = build_executive_training_targets(
            experiences, space, weight_mode="gap")
        assert np.all(targets.weights >= 0)


class TestUncertainty:
    def test_single_replicate(self):
        utils = np.array([[0.9, 0.1, 0.3], [0.4, 0.5, 0.1]])
        sigma = estimate_uncertainty(utils)
        assert sigma.shape == (2,)
        assert np.all(sigma > 0)

    def test_multi_replicate(self):
        utils = np.random.randn(10, 3, 5)  # 10 examples, 3 actions, 5 replicates
        sigma = estimate_uncertainty(utils, n_replicates=5)
        assert sigma.shape == (10,)
        assert np.all(sigma > 0)
