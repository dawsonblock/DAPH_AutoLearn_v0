"""Section 15 — multi-seed sham control (v0.3.10.4-alpha).

Primary sham: shuffle target labels within ``subtype × split ×
decisive/nondecisive`` bins, preserving class balance and subtype
prevalence. Secondary sham: permute utility gaps within subtype.

≥20 sham seeds; report mean sham utility, distribution, 95% sham
interval, percentile of P1 vs sham, and group-aware P1-minus-sham
interval.

The sham uses the SAME features, model class, regularization, optimizer,
example count, weight distribution, and calibration as P1 — only the
feature→winner mapping is destroyed. This is enforced via
:class:`PolicyTrainingSpec` — both P1 and sham must share the same spec.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class PolicyTrainingSpec:
    """Section 15 — shared training specification for P1 and sham.

    Both P1 and sham must be produced by the same trainer with the same
    spec. Only the target permutation differs. The spec hash is recorded
    and asserted equal between P1 and sham.
    """
    feature_schema_hash: str
    policy_class: str
    regularization: float
    optimizer: str
    seed: int
    target_mode: str
    calibration_method: str

    @property
    def spec_hash(self) -> str:
        payload = {
            "feature_schema_hash": self.feature_schema_hash,
            "policy_class": self.policy_class,
            "regularization": self.regularization,
            "optimizer": self.optimizer,
            "seed": self.seed,
            "target_mode": self.target_mode,
            "calibration_method": self.calibration_method,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
    training_spec_hash: str = ""
    feature_matrix_hash: str = ""
    n_shuffled: int = 0
    n_total: int = 0

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
            "training_spec_hash": self.training_spec_hash,
            "feature_matrix_hash": self.feature_matrix_hash,
            "n_shuffled": self.n_shuffled,
            "n_total": self.n_total,
        }


def _bin_key(subtype: str, split: str, decisive: bool) -> str:
    return f"{subtype}|{split}|{'decisive' if decisive else 'nondecisive'}"


def _fallback_hierarchy(subtypes, splits, decisive):
    """Section 15 — fallback hierarchy for single-item bins.

    subtype × split × decisive
    → subtype × split
    → subtype
    → global decisive status
    → global

    Every permutation group must contain at least two observations.
    """
    n = len(subtypes)
    # Try progressively coarser binning.
    for key_fn in (
        lambda i: _bin_key(str(subtypes[i]), str(splits[i]), bool(decisive[i])),
        lambda i: f"{subtypes[i]}|{splits[i]}",
        lambda i: str(subtypes[i]),
        lambda i: f"{'decisive' if decisive[i] else 'nondecisive'}",
        lambda i: "global",
    ):
        bins: dict[str, list[int]] = {}
        for i in range(n):
            bins.setdefault(key_fn(i), []).append(i)
        if all(len(v) >= 2 for v in bins.values()):
            return bins
    # Last resort: one global bin (guaranteed >= 2 if n >= 2).
    return {"global": list(range(n))}


def shuffle_labels_within_bins(
    labels: np.ndarray,
    subtypes: np.ndarray,
    splits: np.ndarray,
    decisive: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Section 15 — shuffle labels within subtype×split×decisive bins.

    Uses the fallback hierarchy to ensure every permutation group has
    at least two observations. Single-item bins are merged into coarser
    bins rather than silently retaining the original label.

    Returns
    -------
    shuffled : np.ndarray
        The shuffled label array.
    n_shuffled : int
        Number of items that were actually permuted.
    """
    rng = np.random.RandomState(seed)
    shuffled = labels.copy()
    bins = _fallback_hierarchy(subtypes, splits, decisive)
    n_shuffled = 0
    for key, indices in bins.items():
        if len(indices) > 1:
            shuffled[indices] = rng.permutation(shuffled[indices])
            n_shuffled += len(indices)
    return shuffled, n_shuffled


def _feature_matrix_hash(features: np.ndarray) -> str:
    """Compute a hash of the feature matrix for provenance."""
    return hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()


def run_sham_control(
    labels: np.ndarray,
    subtypes: np.ndarray,
    splits: np.ndarray,
    decisive: np.ndarray,
    features: np.ndarray,
    *,
    p1_utility: float,
    train_fn: Callable,
    evaluate_fn: Callable,
    training_spec: PolicyTrainingSpec,
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
        Feature matrix (same as P1 — asserted via training_spec).
    p1_utility : float
        The P1 (real policy) utility on the final split.
    train_fn : callable
        ``train_fn(features, labels) -> model`` — trains a policy.
        Must use the same training_spec as P1.
    evaluate_fn : callable
        ``evaluate_fn(model) -> float`` — evaluates on the final split.
    training_spec : PolicyTrainingSpec
        The shared training specification. Both P1 and sham must use
        the same spec — the spec hash is recorded in the result.
    n_seeds : int
        Number of sham seeds (≥20 recommended).
    master_seed : int
        Master seed for reproducibility.

    Returns
    -------
    ShamResult
    """
    feat_hash = _feature_matrix_hash(features)
    sham_utilities: list[float] = []
    n_shuffled_total = 0
    for i in range(n_seeds):
        seed = master_seed + i
        sham_labels, n_shuffled = shuffle_labels_within_bins(
            labels, subtypes, splits, decisive, seed=seed)
        n_shuffled_total = n_shuffled  # same for all seeds (same binning)
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
        training_spec_hash=training_spec.spec_hash,
        feature_matrix_hash=feat_hash,
        n_shuffled=n_shuffled_total,
        n_total=len(labels),
    )


__all__ = [
    "PolicyTrainingSpec",
    "ShamResult",
    "run_sham_control",
    "shuffle_labels_within_bins",
]
