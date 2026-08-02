"""DAPH v0.4.0a3 — B5 adaptive compute diagnostics.

Because B5 is specifically about whether more reasoning compute is
worthwhile, this module provides:
* Empirical crossover analysis (winner distribution per family)
* Action advantage margin
* Compute budget frontier (Pareto table)
* THINK-FAST delta analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def empirical_crossover_analysis(
    utilities: np.ndarray,
    task_families: Sequence[str],
    action_ids: Sequence[str],
) -> dict[str, Any]:
    """Calculate empirical winner distribution per task family.

    For each task family, report what fraction of tasks each action wins.
    Flag low-information families where one action dominates >90%.

    Parameters
    ----------
    utilities : np.ndarray  [N, n_actions]
        Per-task counterfactual utilities.
    task_families : sequence of str  [N]
        Family assignment for each task.
    action_ids : sequence of str
        Action names.
    """
    n_actions = len(action_ids)
    families = sorted(set(task_families))
    winner_dist = {}
    low_info_families = []

    for family in families:
        idx = [i for i, f in enumerate(task_families) if f == family]
        if not idx:
            continue
        family_utils = utilities[idx]
        winners = np.argmax(family_utils, axis=1)
        dist = {}
        for a, aid in enumerate(action_ids):
            frac = float(np.mean(winners == a))
            dist[aid] = round(frac, 4)
        winner_dist[family] = dist

        # Flag low-information families (one action > 90%)
        max_frac = max(dist.values())
        if max_frac > 0.90:
            low_info_families.append({
                "family": family,
                "dominant_action": max(dist, key=dist.get),
                "dominant_fraction": max_frac,
            })

    return {
        "winner_distribution": winner_dist,
        "low_information_families": low_info_families,
        "n_families": len(families),
    }


def think_fast_delta_analysis(
    utilities: np.ndarray,
    task_families: Sequence[str],
    difficulties: Sequence[str],
    prompt_lengths: Sequence[int],
    fast_idx: int,
    think_idx: int,
    decompose_idx: int,
    retrieve_idx: int,
) -> dict[str, Any]:
    """Analyze THINK-FAST utility delta.

    ΔU_THINK-FAST = U_THINK - U_FAST

    Reports:
    * fraction THINK improves correctness
    * fraction THINK lowers utility due to cost
    * fraction FAST already sufficient
    * fraction DECOMPOSE dominates both
    * fraction RETRIEVE dominates both
    """
    n = len(utilities)
    if n == 0:
        return {}

    u_fast = utilities[:, fast_idx]
    u_think = utilities[:, think_idx]
    u_decompose = utilities[:, decompose_idx]
    u_retrieve = utilities[:, retrieve_idx]

    delta_tf = u_think - u_fast

    # Fraction THINK improves (delta > 0.01)
    frac_think_improves = float(np.mean(delta_tf > 0.01))
    # Fraction THINK lowers utility (delta < -0.01)
    frac_think_lowers = float(np.mean(delta_tf < -0.01))
    # Fraction FAST already sufficient (|delta| <= 0.01)
    frac_fast_sufficient = float(np.mean(np.abs(delta_tf) <= 0.01))

    # Fraction DECOMPOSE dominates both THINK and FAST
    decompose_dominates = float(np.mean(
        (u_decompose > u_think) & (u_decompose > u_fast)
    ))
    # Fraction RETRIEVE dominates both
    retrieve_dominates = float(np.mean(
        (u_retrieve > u_think) & (u_retrieve > u_fast)
    ))

    # Analyze delta against families
    family_deltas = {}
    for family in sorted(set(task_families)):
        idx = [i for i, f in enumerate(task_families) if f == family]
        if idx:
            family_deltas[family] = float(np.mean(delta_tf[idx]))

    # Analyze delta against difficulty
    difficulty_deltas = {}
    for diff in sorted(set(difficulties)):
        idx = [i for i, d in enumerate(difficulties) if d == diff]
        if idx:
            difficulty_deltas[diff] = float(np.mean(delta_tf[idx]))

    # Analyze delta against prompt length buckets
    lengths = np.array(prompt_lengths)
    median_len = float(np.median(lengths))
    short_idx = lengths <= median_len
    long_idx = lengths > median_len
    length_deltas = {
        "short": float(np.mean(delta_tf[short_idx])) if short_idx.any() else 0.0,
        "long": float(np.mean(delta_tf[long_idx])) if long_idx.any() else 0.0,
    }

    return {
        "mean_delta": float(np.mean(delta_tf)),
        "median_delta": float(np.median(delta_tf)),
        "frac_think_improves": frac_think_improves,
        "frac_think_lowers": frac_think_lowers,
        "frac_fast_sufficient": frac_fast_sufficient,
        "frac_decompose_dominates": decompose_dominates,
        "frac_retrieve_dominates": retrieve_dominates,
        "by_family": family_deltas,
        "by_difficulty": difficulty_deltas,
        "by_prompt_length": length_deltas,
    }


def compute_budget_frontier(
    policy_results: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Build a Pareto-style compute budget frontier table.

    For each policy, report:
    * mean verified accuracy
    * mean utility
    * mean latency
    * mean tokens
    * mean LLM calls
    * oracle regret

    Parameters
    ----------
    policy_results : dict
        Maps policy name to dict with keys:
        accuracy, utility, latency_ms, tokens, llm_calls, oracle_utility
    """
    frontier = []
    for name, metrics in sorted(policy_results.items()):
        oracle_utility = metrics.get("oracle_utility", 0.0)
        regret = oracle_utility - metrics.get("utility", 0.0)
        frontier.append({
            "policy": name,
            "accuracy": metrics.get("accuracy", 0.0),
            "utility": metrics.get("utility", 0.0),
            "latency_ms": metrics.get("latency_ms", 0.0),
            "tokens": metrics.get("tokens", 0.0),
            "llm_calls": metrics.get("llm_calls", 0.0),
            "oracle_regret": regret,
        })

    # Sort by utility descending
    frontier.sort(key=lambda x: x["utility"], reverse=True)
    return frontier


__all__ = [
    "empirical_crossover_analysis",
    "think_fast_delta_analysis",
    "compute_budget_frontier",
]
