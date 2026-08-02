"""DAPH v0.4 — Matched hidden-state sham for B4.

Destroys the mapping between hidden vectors and action advantage
while preserving the marginal distribution of utilities within
subtype × difficulty × split bins.

Trains 20-50 sham policies and compares:
    P_hidden_real - P_hidden_sham
"""

from __future__ import annotations

import numpy as np
from typing import Any


def create_sham_utilities(
    utilities: np.ndarray,
    subtypes: list[str],
    split_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Create sham utilities by shuffling within subtype × split bins.

    This destroys the mapping between features and action advantage
    while preserving the marginal distribution of utilities within
    each bin.

    Parameters
    ----------
    utilities : np.ndarray  [N, n_actions]
        Real counterfactual utilities.
    subtypes : list[str]  [N]
        Subtype labels for each task.
    split_ids : np.ndarray  [N]
        Split identifier (0=train, 1=dev, 2=final) for each task.
    seed : int
        Random seed for shuffling.

    Returns
    -------
    np.ndarray  [N, n_actions]
        Sham utilities (same shape, shuffled within bins).
    """
    rng = np.random.RandomState(seed)
    sham = utilities.copy()

    # Group by (subtype, split) and shuffle within each group
    n = len(utilities)
    bins: dict[tuple[str, int], list[int]] = {}
    for i in range(n):
        key = (subtypes[i], int(split_ids[i]))
        bins.setdefault(key, []).append(i)

    for key, indices in bins.items():
        if len(indices) <= 1:
            continue
        # Shuffle the utility rows within this bin
        perm = rng.permutation(len(indices))
        sham[indices] = utilities[[indices[p] for p in perm]]

    return sham


def run_sham_experiment(
    train_features: np.ndarray,
    train_utilities: np.ndarray,
    test_features: np.ndarray,
    test_utilities: np.ndarray,
    train_subtypes: list[str],
    test_subtypes: list[str],
    train_split_ids: np.ndarray,
    test_split_ids: np.ndarray,
    policy_cls,
    policy_kwargs: dict | None = None,
    n_shams: int = 50,
    seed_base: int = 10000,
) -> dict:
    """Run multiple sham experiments and return aggregate statistics.

    Parameters
    ----------
    train_features, test_features : np.ndarray
        Hidden-state features (already PCA-reduced).
    train_utilities, test_utilities : np.ndarray  [N, n_actions]
        Real counterfactual utilities.
    train_subtypes, test_subtypes : list[str]
    train_split_ids, test_split_ids : np.ndarray
    policy_cls : type
        Policy class (QRegressionPolicy or similar).
    policy_kwargs : dict
    n_shams : int
        Number of sham policies to train.
    seed_base : int

    Returns
    -------
    dict with sham results
    """
    from daph_learning.executive.q_policy import mean_regret

    if policy_kwargs is None:
        policy_kwargs = {}

    sham_regrets = []
    action_ids = policy_kwargs.get("action_ids", [f"a{i}" for i in range(train_utilities.shape[1])])

    for s in range(n_shams):
        seed = seed_base + s
        sham_train_utils = create_sham_utilities(
            train_utilities, train_subtypes, train_split_ids, seed
        )

        # Train policy on sham utilities
        policy = policy_cls(**policy_kwargs)
        policy.fit(train_features, sham_train_utils)

        # Evaluate on REAL test utilities
        test_preds = policy.predict(test_features)
        regret = mean_regret(test_preds, test_utilities)
        sham_regrets.append(regret)

        if (s + 1) % 10 == 0:
            print(f"    Sham {s+1}/{n_shams}: regret={regret:.4f}")

    sham_regrets = np.array(sham_regrets)
    return {
        "n_shams": n_shams,
        "sham_regrets": sham_regrets.tolist(),
        "sham_mean_regret": float(sham_regrets.mean()),
        "sham_std_regret": float(sham_regrets.std()),
        "sham_median_regret": float(np.median(sham_regrets)),
        "sham_p05_regret": float(np.percentile(sham_regrets, 5)),
        "sham_p95_regret": float(np.percentile(sham_regrets, 95)),
    }


__all__ = ["create_sham_utilities", "run_sham_experiment"]
