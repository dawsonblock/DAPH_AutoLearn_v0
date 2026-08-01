"""Tests for DAPH v0.4 generic executive policy models.

Verifies that ExecutiveCentroidPolicy and ExecutiveLogisticPolicy
work correctly for arbitrary N-action spaces, including:
- 2-action (binary compatibility with v0.3.x)
- 3-action (the first new v0.4 use case)
- 4-action (general case)
"""

from __future__ import annotations

import json
import pytest
import numpy as np

from daph_learning.executive import (
    ActionDescriptor,
    ActionSpace,
    ActionDecision,
    ExecutiveCentroidPolicy,
    ExecutiveLogisticPolicy,
    make_executive_policy,
    binary_action_space,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

def _make_3action_space():
    return ActionSpace(actions=(
        ActionDescriptor(action_id="reasoning.direct", cost_estimate=0.1),
        ActionDescriptor(action_id="retrieval.vector", cost_estimate=0.05),
        ActionDescriptor(action_id="reasoning.decompose", cost_estimate=0.3),
    ))

def _make_synthetic_data(n=60, d=8, n_actions=3, seed=42):
    """Generate synthetic features and per-action utilities with cluster structure."""
    rng = np.random.RandomState(seed)
    features = rng.randn(n, d)
    utils = np.zeros((n, n_actions))
    for i in range(n):
        cluster = i % n_actions
        utils[i, cluster] = 1.0 + rng.randn() * 0.15
        for j in range(n_actions):
            if j != cluster:
                utils[i, j] = 0.2 + rng.randn() * 0.1
    weights = np.ones(n)
    return features, utils, weights


# ──────────────────────────────────────────────────────────────────────
# ExecutiveCentroidPolicy Tests
# ──────────────────────────────────────────────────────────────────────

class TestExecutiveCentroidPolicy:
    def test_fit_3action(self):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveCentroidPolicy()
        model.fit(feats, utils, w, space)
        assert len(model.action_ids) == 3
        assert model.centroids.shape == (3, 8)
        assert model.n_train_ == 60
        assert sum(model.class_counts.values()) <= 60  # ties excluded

    def test_predict_proba_shape(self):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveCentroidPolicy()
        model.fit(feats, utils, w, space)
        probs = model.predict_proba(feats)
        assert probs.shape == (60, 3)
        # Probabilities sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_single(self):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveCentroidPolicy()
        model.fit(feats, utils, w, space)
        probs = model.predict_proba(feats[0])
        assert probs.shape == (3,)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_predict_decision(self):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveCentroidPolicy()
        model.fit(feats, utils, w, space)
        decisions = model.predict_decision(feats[:5], space, state_ids=["s1","s2","s3","s4","s5"])
        assert len(decisions) == 5
        for d in decisions:
            assert isinstance(d, ActionDecision)
            assert d.selected_action in space.action_ids
            assert d.confidence > 0

    def test_binary_compatibility(self):
        """2-action centroid should produce probabilities summing to 1."""
        space = binary_action_space()
        n, d = 40, 8
        rng = np.random.RandomState(42)
        feats = rng.randn(n, d)
        utils = np.zeros((n, 2))
        for i in range(n):
            if i % 2 == 0:
                utils[i] = [0.9, 0.2]
            else:
                utils[i] = [0.2, 0.9]
        w = np.ones(n)
        model = ExecutiveCentroidPolicy()
        model.fit(feats, utils, w, space)
        probs = model.predict_proba(feats)
        assert probs.shape == (n, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_save_load(self, tmp_path):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveCentroidPolicy()
        model.fit(feats, utils, w, space)
        path = str(tmp_path / "centroid.json")
        model.save(path)
        loaded = ExecutiveCentroidPolicy.load(path)
        assert loaded.action_ids == model.action_ids
        np.testing.assert_allclose(loaded.centroids, model.centroids)
        assert loaded.n_train_ == model.n_train_

    def test_not_fitted_raises(self):
        model = ExecutiveCentroidPolicy()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.zeros(5))

    def test_degenerate_data(self):
        """All ties → degenerate flag set."""
        space = _make_3action_space()
        n, d = 20, 5
        feats = np.random.randn(n, d)
        # All utilities equal → all ties
        utils = np.ones((n, 3)) * 0.5
        w = np.ones(n)
        model = ExecutiveCentroidPolicy()
        model.fit(feats, utils, w, space, gap_threshold=0.01)
        assert model.degenerate_train_data is True

    def test_shape_mismatch_raises(self):
        space = _make_3action_space()
        feats = np.random.randn(10, 5)
        utils = np.zeros((10, 2))  # wrong n_actions
        w = np.ones(10)
        model = ExecutiveCentroidPolicy()
        with pytest.raises(ValueError, match="action_utilities shape"):
            model.fit(feats, utils, w, space)

    def test_estimator_name(self):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveCentroidPolicy()
        model.fit(feats, utils, w, space)
        assert model.estimator_name() == "executive_centroid"

    def test_estimator_name_with_fallback(self):
        space = _make_3action_space()
        n, d = 30, 5
        rng = np.random.RandomState(42)
        feats = rng.randn(n, d)
        utils = np.zeros((n, 3))
        # Make one action always win, with zero weight for another
        for i in range(n):
            utils[i, 0] = 1.0
            utils[i, 1] = 0.1
            utils[i, 2] = 0.1
        w = np.ones(n)
        w[:10] = 0  # zero weight for first 10 → may trigger fallback
        model = ExecutiveCentroidPolicy()
        model.fit(feats, utils, w, space, eps=1e-15)
        # With zero weight on some examples, fallback may trigger
        if model.weight_fallback:
            assert "with_unweighted_fallback" in model.estimator_name()


# ──────────────────────────────────────────────────────────────────────
# ExecutiveLogisticPolicy Tests
# ──────────────────────────────────────────────────────────────────────

class TestExecutiveLogisticPolicy:
    def test_fit_3action(self):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveLogisticPolicy()
        model.fit(feats, utils, w, space, n_iter=100)
        assert len(model.action_ids) == 3
        assert model.weights.shape == (8, 3)
        assert model.bias.shape == (3,)
        assert model.n_train_ == 60

    def test_predict_proba_shape(self):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveLogisticPolicy()
        model.fit(feats, utils, w, space, n_iter=100)
        probs = model.predict_proba(feats)
        assert probs.shape == (60, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_single(self):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveLogisticPolicy()
        model.fit(feats, utils, w, space, n_iter=100)
        probs = model.predict_proba(feats[0])
        assert probs.shape == (3,)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_predict_decision(self):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveLogisticPolicy()
        model.fit(feats, utils, w, space, n_iter=100)
        decisions = model.predict_decision(feats[:3], space)
        assert len(decisions) == 3
        for d in decisions:
            assert d.selected_action in space.action_ids

    def test_convergence(self):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveLogisticPolicy()
        model.fit(feats, utils, w, space, n_iter=500, tol=1e-8)
        # Loss should decrease
        assert len(model.convergence_history) > 0
        assert model.convergence_history[-1] < model.convergence_history[0]

    def test_binary_compatibility(self):
        """2-action logistic should work and sum to 1."""
        space = binary_action_space()
        n, d = 40, 8
        rng = np.random.RandomState(42)
        feats = rng.randn(n, d)
        utils = np.zeros((n, 2))
        for i in range(n):
            if i % 2 == 0:
                utils[i] = [0.9, 0.2]
            else:
                utils[i] = [0.2, 0.9]
        w = np.ones(n)
        model = ExecutiveLogisticPolicy()
        model.fit(feats, utils, w, space, n_iter=200)
        probs = model.predict_proba(feats)
        assert probs.shape == (n, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_save_load(self, tmp_path):
        space = _make_3action_space()
        feats, utils, w = _make_synthetic_data(n_actions=3)
        model = ExecutiveLogisticPolicy()
        model.fit(feats, utils, w, space, n_iter=100)
        path = str(tmp_path / "logistic.json")
        model.save(path)
        loaded = ExecutiveLogisticPolicy.load(path)
        assert loaded.action_ids == model.action_ids
        np.testing.assert_allclose(loaded.weights, model.weights)
        assert loaded.n_train_ == model.n_train_

    def test_not_fitted_raises(self):
        model = ExecutiveLogisticPolicy()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.zeros(5))

    def test_estimator_name(self):
        model = ExecutiveLogisticPolicy()
        assert model.estimator_name() == "executive_logistic"


# ──────────────────────────────────────────────────────────────────────
# Factory Tests
# ──────────────────────────────────────────────────────────────────────

class TestFactory:
    def test_make_centroid(self):
        p = make_executive_policy("centroid")
        assert isinstance(p, ExecutiveCentroidPolicy)

    def test_make_logistic(self):
        p = make_executive_policy("logistic")
        assert isinstance(p, ExecutiveLogisticPolicy)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            make_executive_policy("unknown")


# ──────────────────────────────────────────────────────────────────────
# 4-Action General Case Tests
# ──────────────────────────────────────────────────────────────────────

class TestFourActions:
    def test_centroid_4action(self):
        space = ActionSpace(actions=(
            ActionDescriptor(action_id="a"),
            ActionDescriptor(action_id="b"),
            ActionDescriptor(action_id="c"),
            ActionDescriptor(action_id="d"),
        ))
        feats, utils, w = _make_synthetic_data(n=80, d=10, n_actions=4)
        model = ExecutiveCentroidPolicy()
        model.fit(feats, utils, w, space)
        probs = model.predict_proba(feats)
        assert probs.shape == (80, 4)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_logistic_4action(self):
        space = ActionSpace(actions=(
            ActionDescriptor(action_id="a"),
            ActionDescriptor(action_id="b"),
            ActionDescriptor(action_id="c"),
            ActionDescriptor(action_id="d"),
        ))
        feats, utils, w = _make_synthetic_data(n=80, d=10, n_actions=4)
        model = ExecutiveLogisticPolicy()
        model.fit(feats, utils, w, space, n_iter=200)
        probs = model.predict_proba(feats)
        assert probs.shape == (80, 4)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
