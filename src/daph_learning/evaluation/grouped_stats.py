"""v0.3.10.3.2-alpha — group-aware statistics (Sections 9, 31).

Even with no exact duplicates, tasks may share templates or latent
generators. Standard bootstrap that resamples individual tasks
overstates the effective sample size when many tasks are near
duplicates. This module provides grouped bootstrap that resamples
groups (templates/families) instead of individual tasks.

Also provides effective sample size (ESS) for weighted estimators
(Section 31) and weight diagnostics.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def grouped_bootstrap_mean_delta(
    records: Sequence[Mapping[str, Any]],
    group_key: str,
    value_a_key: str,
    value_b_key: str,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Grouped bootstrap of ``mean(value_a) - mean(value_b)`` (Section 9).

    Resamples GROUPS (identified by ``group_key``) with replacement,
    not individual records, so that near-duplicate tasks sharing a
    template/generator are kept together. This avoids overstating N.

    Parameters
    ----------
    records : sequence of mappings
        Each record must have ``group_key``, ``value_a_key``,
        ``value_b_key``.
    group_key : str
        Field identifying the group (e.g. ``"template_id"``,
        ``"group_id"``).
    value_a_key, value_b_key : str
        Fields whose mean difference is bootstrapped (e.g. utility
        under policy A vs policy B).
    n_bootstrap : int
    seed : int

    Returns
    -------
    dict with ``mean_delta``, ``ci_low``, ``ci_high``, ``n_groups``,
    ``n_records``.
    """
    rng = np.random.default_rng(seed)
    # Group records by group_key.
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for r in records:
        g = r.get(group_key)
        groups.setdefault(g, []).append(r)
    group_list = list(groups.values())
    n_groups = len(group_list)
    if n_groups == 0:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "n_groups": 0, "n_records": 0}
    a_vals = np.array([float(r[value_a_key]) for r in records])
    b_vals = np.array([float(r[value_b_key]) for r in records])
    mean_delta = float(a_vals.mean() - b_vals.mean())
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_groups, size=n_groups)
        a_sample = []
        b_sample = []
        for j in idx:
            for r in group_list[j]:
                a_sample.append(float(r[value_a_key]))
                b_sample.append(float(r[value_b_key]))
        deltas[i] = np.mean(a_sample) - np.mean(b_sample)
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    return {
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_groups": n_groups,
        "n_records": len(records),
    }


def effective_sample_size(weights: np.ndarray | Sequence[float]) -> float:
    """Weighted effective sample size (Section 31).

    ``ESS = (sum w_i)^2 / sum w_i^2``

    Exposes cases where weighting leaves only a tiny effective dataset.
    Returns 0.0 for an all-zero weight array.
    """
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    sq = float((w ** 2).sum())
    if sq <= 0.0:
        return 0.0
    return total * total / sq


def weight_diagnostics(weights: np.ndarray | Sequence[float]) -> dict[str, float]:
    """Per-policy weight diagnostics (Section 6).

    Returns effective sample weight sum, min, max, mean, zero-weight
    fraction, and ESS.
    """
    w = np.asarray(weights, dtype=np.float64)
    n = w.shape[0]
    return {
        "n": int(n),
        "weight_sum": float(w.sum()),
        "min_weight": float(w.min()) if n else 0.0,
        "max_weight": float(w.max()) if n else 0.0,
        "mean_weight": float(w.mean()) if n else 0.0,
        "zero_weight_fraction": float((w == 0.0).mean()) if n else 0.0,
        "ess": effective_sample_size(w),
    }


__all__ = [
    "effective_sample_size",
    "grouped_bootstrap_mean_delta",
    "weight_diagnostics",
]
