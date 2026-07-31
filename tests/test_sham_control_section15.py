"""Section 15 — sham control tests."""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.evaluation.sham import (
    PolicyTrainingSpec,
    ShamResult,
    run_sham_control,
    shuffle_labels_within_bins,
)


def _spec():
    return PolicyTrainingSpec(
        feature_schema_hash="abc",
        policy_class="logistic",
        regularization=0.01,
        optimizer="adam",
        seed=42,
        target_mode="signal_to_noise",
        calibration_method="isotonic",
    )


def test_shuffle_preserves_bin_counts():
    labels = np.array([1, 0, 1, 0, 1, 0])
    subtypes = np.array(["A", "A", "A", "B", "B", "B"])
    splits = np.array(["train"] * 6)
    decisive = np.array([True, True, False, True, True, False])
    shuffled, n = shuffle_labels_within_bins(
        labels, subtypes, splits, decisive, seed=42)
    assert shuffled.sum() == labels.sum()
    assert n > 0


def test_shuffle_returns_count():
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    subtypes = np.array(["A"] * 8)
    splits = np.array(["train"] * 8)
    decisive = np.array([True] * 8)
    shuffled, n = shuffle_labels_within_bins(
        labels, subtypes, splits, decisive, seed=99)
    assert shuffled.sum() == labels.sum()
    assert n == 8


def test_shuffle_fallback_hierarchy():
    """Single-item bins should be merged via the fallback hierarchy."""
    labels = np.array([1, 0, 1, 0])
    subtypes = np.array(["A", "B", "C", "D"])  # each subtype has 1 item
    splits = np.array(["train"] * 4)
    decisive = np.array([True] * 4)
    shuffled, n = shuffle_labels_within_bins(
        labels, subtypes, splits, decisive, seed=42)
    # Fallback to global bin → all 4 items shuffled.
    assert n == 4
    assert shuffled.sum() == labels.sum()


def test_run_sham_control():
    np.random.seed(42)
    n = 100
    labels = np.random.randint(0, 2, n)
    subtypes = np.random.choice(["A", "B", "C"], n)
    splits = np.array(["train"] * n)
    decisive = np.array([True] * n)
    features = np.random.randn(n, 4)

    def train_fn(X, y):
        return {"mean": float(y.mean())}

    def eval_fn(model):
        return model["mean"]

    result = run_sham_control(
        labels, subtypes, splits, decisive, features,
        p1_utility=0.6, train_fn=train_fn, evaluate_fn=eval_fn,
        training_spec=_spec(), n_seeds=10, master_seed=42)

    assert result.n_seeds == 10
    assert len(result.sham_utilities) == 10
    assert 0.0 <= result.p1_percentile_vs_sham <= 100.0
    assert result.training_spec_hash != ""
    assert result.feature_matrix_hash != ""
    assert result.n_shuffled > 0


def test_sham_signal_destroyed():
    np.random.seed(42)
    n = 200
    labels = np.random.randint(0, 2, n)
    subtypes = np.array(["A"] * n)
    splits = np.array(["train"] * n)
    decisive = np.array([True] * n)
    features = np.random.randn(n, 4)

    def train_fn(X, y):
        return float(y.mean())

    def eval_fn(m):
        return m

    result = run_sham_control(
        labels, subtypes, splits, decisive, features,
        p1_utility=0.5, train_fn=train_fn, evaluate_fn=eval_fn,
        training_spec=_spec(), n_seeds=20, master_seed=42)
    assert abs(result.mean_sham_utility - 0.5) < 0.1


def test_to_dict_serializable():
    result = ShamResult(n_seeds=5, sham_utilities=[0.1, 0.2, 0.3, 0.4, 0.5])
    d = result.to_dict()
    assert "n_seeds" in d
    assert "training_spec_hash" in d


def test_policy_training_spec_hash_deterministic():
    s1 = _spec()
    s2 = _spec()
    assert s1.spec_hash == s2.spec_hash
    s3 = PolicyTrainingSpec(
        feature_schema_hash="abc", policy_class="logistic",
        regularization=0.01, optimizer="adam", seed=99,
        target_mode="signal_to_noise", calibration_method="isotonic")
    assert s1.spec_hash != s3.spec_hash  # different seed
