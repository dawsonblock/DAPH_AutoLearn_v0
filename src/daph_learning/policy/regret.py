"""v0.3.10 — regret metrics and paired bootstrap (Sections 2, 19).

Regret is the **primary scientific metric**, not routing accuracy.

Per-task regret::

    Regret_i = max_a U_i(a) - U_i(π(x_i))

Mean regret::

    R̄ = (1/N) Σ_i Regret_i

A policy can have high classification accuracy but make its mistakes on
tasks where the utility penalty is enormous. The learning objective and
evaluation metric must preserve this distinction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def per_task_regret(
    policy_utilities: np.ndarray | Sequence[float],
    oracle_utilities: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Per-task regret: ``max_a U(a) - U(π(x))``.

    Parameters
    ----------
    policy_utilities : array-like
        ``U_i(π(x_i))`` — the utility of the action the policy actually
        selected. Shape ``[N]``.
    oracle_utilities : array-like
        ``max_a U_i(a)`` — the best achievable utility per task. Shape
        ``[N]``. This is the oracle counterfactual, used ONLY for scoring
        regret AFTER the policy acts. It must NEVER replace the policy's
        actual decision (Section 18).

    Returns
    -------
    np.ndarray
        Shape ``[N]`` per-task regrets (>= 0).
    """
    pu = np.asarray(policy_utilities, dtype=np.float64)
    ou = np.asarray(oracle_utilities, dtype=np.float64)
    if pu.shape != ou.shape:
        raise ValueError("policy_utilities and oracle_utilities must have the same shape")
    if len(pu) == 0:
        raise ValueError("empty input")
    regrets = ou - pu
    # Regret should be non-negative (policy cannot beat the oracle).
    # Clamp tiny negative values from floating-point error to 0.
    regrets = np.maximum(regrets, 0.0)
    return regrets


def mean_regret(
    policy_utilities: np.ndarray | Sequence[float],
    oracle_utilities: np.ndarray | Sequence[float],
) -> float:
    """Mean regret: ``(1/N) Σ Regret_i``."""
    return float(per_task_regret(policy_utilities, oracle_utilities).mean())


def paired_bootstrap_mean_ci(
    deltas: np.ndarray | Sequence[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired bootstrap confidence interval for the mean of ``deltas``
    (Section 19).

    Parameters
    ----------
    deltas : array-like
        Per-task deltas (e.g. ``U_candidate - U_incumbent`` or
        ``Regret_incumbent - Regret_candidate``).
    n_bootstrap : int
    alpha : float
        Significance level (returns a ``1 - alpha`` CI).
    seed : int

    Returns
    -------
    tuple[float, float]
        ``(low, high)`` bounds of the bootstrap CI.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)
    if n == 0:
        raise ValueError("empty delta array")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("deltas contain NaN/Inf")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = deltas[idx].mean()
    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return low, high


def paired_promotion_statistics(
    candidate_utilities: np.ndarray | Sequence[float],
    incumbent_utilities: np.ndarray | Sequence[float],
    oracle_utilities: np.ndarray | Sequence[float] | None = None,
    *,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Paired promotion statistics (Section 19).

    Reports mean/median utility delta, mean regret delta, win/tie/loss
    rates, and bootstrap CI.
    """
    cu = np.asarray(candidate_utilities, dtype=np.float64)
    iu = np.asarray(incumbent_utilities, dtype=np.float64)
    if cu.shape != iu.shape:
        raise ValueError("candidate/incumbent utility arrays must match")
    n = len(cu)
    deltas = cu - iu
    wins = int(np.sum(deltas > 1e-9))
    losses = int(np.sum(deltas < -1e-9))
    ties = n - wins - losses
    low, high = paired_bootstrap_mean_ci(deltas, n_bootstrap, alpha, seed)
    result = {
        "n": n,
        "mean_utility_delta": float(deltas.mean()),
        "median_utility_delta": float(np.median(deltas)),
        "candidate_win_rate": wins / n if n else 0.0,
        "incumbent_win_rate": losses / n if n else 0.0,
        "tie_rate": ties / n if n else 0.0,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
    }
    if oracle_utilities is not None:
        ou = np.asarray(oracle_utilities, dtype=np.float64)
        cand_regret = per_task_regret(cu, ou)
        inc_regret = per_task_regret(iu, ou)
        regret_deltas = inc_regret - cand_regret  # positive => candidate better
        r_low, r_high = paired_bootstrap_mean_ci(regret_deltas, n_bootstrap, alpha, seed)
        result["mean_regret_delta"] = float(regret_deltas.mean())
        result["mean_candidate_regret"] = float(cand_regret.mean())
        result["mean_incumbent_regret"] = float(inc_regret.mean())
        result["regret_bootstrap_ci_low"] = r_low
        result["regret_bootstrap_ci_high"] = r_high
    return result


__all__ = [
    "mean_regret",
    "paired_bootstrap_mean_ci",
    "paired_promotion_statistics",
    "per_task_regret",
]
