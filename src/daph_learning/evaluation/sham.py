"""Section 15 — multi-seed sham control (v0.3.10.4-alpha).

Primary sham: shuffle target labels within ``subtype × split ×
decisive/nondecisive`` bins, preserving class balance and subtype
prevalence. Secondary sham: permute utility gaps within subtype.

≥20 sham seeds; report mean sham utility, distribution, 95% sham
interval, percentile of P1 vs sham, and group-aware P1-minus-sham
interval.

The sham uses the SAME features, model class, regularization, optimizer,
example count, weight distribution, and calibration as P1 — only the
feature→winner mapping is destroyed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ShamResult:
    """Section 15 — result of a multi-seed sham control."""
    n_seeds: int
    sham_utilities: list[float] = field(default_factory=list)
    mean_sham_utility: float = 0.0
    std_sham_utility: float = 0.0
    sham_ci_lower: float = 0.0
    sham_ci_upper: float = 0.0
    p1_utility: float = 0.0
    p1_minus_sham_mean: float = 0.0
    p1_percentile_vs_sham: float = 0.0
    procedure: str = "subtype_split_decisive_shuffle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_seeds": self.n_seeds,
            "sham_utilities": list(self.sham_utilities),
            "mean_sham_utility": self.mean_sham_utility,
            "std_sham_utility": self.std_sham_utility,
            "sham_ci_lower": self.sham_ci_lower,
            "sham_ci_upper": self.sham_ci_upper,
            "p1_utility": self.p1_utility,
            "p1_minus_sham_mean": self.p1_minus_sham_mean,
            "p1_percentile_vs_sham": self.p1_percentile_vs_sham,
            "procedure": self.procedure,
        }


def _bin_key(subtype: str, split: str, decisive: bool) -> str:
    return f"{subtype}|{split}|{'decisive' if decisive else 'nondecisive'}"


def shuffle_labels_within_bins(
    labels: np.ndarray,
    subtypes: np.ndarray,
    splits: np.ndarray,
    decisive: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Section 15 — shuffle labels within subtype×split×decisive bins.

    Preserves class balance within each bin. Returns a new label array
    with the same shape, where labels have been permuted within bins.
    """
    rng = np.random.RandomState(seed)
    shuffled = labels.copy()
    bins: dict[str, list[int]] = {}
    for i in range(len(labels)):
        key = _bin_key(str(subtypes[i]), str(splits[i]), bool(decisive[i]))
        bins.setdefault(key, []).append(i)
    for key, indices in bins.items():
        if len(indices) > 1:
            shuffled[indices] = rng.permutation(shuffled[indices])
    return shuffled


def run_sham_control(
    labels: np.ndarray,
    subtypes: np.ndarray,
    splits: np.ndarray,
    decisive: np.ndarray,
    features: np.ndarray,
    *,
    p1_utility: float,
    train_fn,
    evaluate_fn,
    n_seeds: int = 20,
    master_seed: int = 42,
) -> ShamResult:
    """Section 15 — run multi-seed sham control.

    Parameters
    ----------
    labels : np.ndarray
        Original training labels (e.g. preferred_action as 0/1).
    subtypes, splits, decisive : np.ndarray
        Grouping arrays for binning.
    features : np.ndarray
        Feature matrix (same as P1).
    p1_utility : float
        The P1 (real policy) utility on the final split.
    train_fn : callable
        ``train_fn(features, labels) -> model`` — trains a policy.
    evaluate_fn : callable
        ``evaluate_fn(model) -> float`` — evaluates on the final split.
    n_seeds : int
        Number of sham seeds (≥20 recommended).
    master_seed : int
        Master seed for reproducibility.

    Returns
    -------
    ShamResult
    """
    sham_utilities: list[float] = []
    for i in range(n_seeds):
        seed = master_seed + i
        sham_labels = shuffle_labels_within_bins(
            labels, subtypes, splits, decisive, seed=seed)
        model = train_fn(features, sham_labels)
        utility = evaluate_fn(model)
        sham_utilities.append(float(utility))

    sham_arr = np.array(sham_utilities)
    mean_sham = float(np.mean(sham_arr))
    std_sham = float(np.std(sham_arr, ddof=1)) if n_seeds > 1 else 0.0
    ci_lower = float(np.percentile(sham_arr, 2.5))
    ci_upper = float(np.percentile(sham_arr, 97.5))

    # Percentile of P1 vs sham distribution.
    n_below = int(np.sum(sham_arr < p1_utility))
    percentile = (n_below / n_seeds) * 100.0

    return ShamResult(
        n_seeds=n_seeds,
        sham_utilities=sham_utilities,
        mean_sham_utility=mean_sham,
        std_sham_utility=std_sham,
        sham_ci_lower=ci_lower,
        sham_ci_upper=ci_upper,
        p1_utility=p1_utility,
        p1_minus_sham_mean=p1_utility - mean_sham,
        p1_percentile_vs_sham=percentile,
    )


__all__ = [
    "ShamResult",
    "run_sham_control",
    "shuffle_labels_within_bins",
]
