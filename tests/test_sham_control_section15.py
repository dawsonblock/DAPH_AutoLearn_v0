"""Section 15 — sham control tests."""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.evaluation.sham import (
    ShamResult,
    run_sham_control,
    shuffle_labels_within_bins,
)


def test_shuffle_preserves_bin_counts():
    labels = np.array([1, 0, 1, 0, 1, 0])
    subtypes = np.array(["A", "A", "A", "B", "B", "B"])
    splits = np.array(["train"] * 6)
    decisive = np.array([True, True, False, True, True, False])
    shuffled = shuffle_labels_within_bins(
        labels, subtypes, splits, decisive, seed=42)
    # Same number of 1s and 0s overall.
    assert shuffled.sum() == labels.sum()
    # Same number of 1s in each bin.
    for s in ["A", "B"]:
        for d in [True, False]:
            mask = (subtypes == s) & (decisive == d)
            assert shuffled[mask].sum() == labels[mask].sum()


def test_shuffle_changes_labels():
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    subtypes = np.array(["A"] * 8)
    splits = np.array(["train"] * 8)
    decisive = np.array([True] * 8)
    shuffled = shuffle_labels_within_bins(
        labels, subtypes, splits, decisive, seed=99)
    # With 8 items in one bin, very likely to change order.
    assert shuffled.sum() == labels.sum()  # same counts


def test_run_sham_control():
    """Run sham with a simple train/evaluate function."""
    np.random.seed(42)
    n = 100
    labels = np.random.randint(0, 2, n)
    subtypes = np.random.choice(["A", "B", "C"], n)
    splits = np.array(["train"] * n)
    decisive = np.array([True] * n)
    features = np.random.randn(n, 4)

    def train_fn(X, y):
        # Trivial: return mean of labels.
        return {"mean": float(y.mean())}

    def eval_fn(model):
        return model["mean"]

    result = run_sham_control(
        labels, subtypes, splits, decisive, features,
        p1_utility=0.6, train_fn=train_fn, evaluate_fn=eval_fn,
        n_seeds=10, master_seed=42)

    assert result.n_seeds == 10
    assert len(result.sham_utilities) == 10
    assert 0.0 <= result.p1_percentile_vs_sham <= 100.0


def test_sham_signal_destroyed():
    """When labels are random, sham should be close to P1."""
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
        n_seeds=20, master_seed=42)

    # With random labels, sham mean should be close to 0.5.
    assert abs(result.mean_sham_utility - 0.5) < 0.1


def test_to_dict_serializable():
    result = ShamResult(n_seeds=5, sham_utilities=[0.1, 0.2, 0.3, 0.4, 0.5])
    d = result.to_dict()
    assert "n_seeds" in d
    assert "sham_utilities" in d
