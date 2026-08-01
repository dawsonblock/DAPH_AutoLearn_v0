"""DAPH v0.4 — Generic executive experience builder and training targets.

This module provides the generic equivalent of:
- ``policy/learner.py::build_counterfactual_experiences`` (binary)
- ``policy/targets.py::build_uncertainty_aware_targets`` (binary)
- ``policy/confidence.py::combine_paired_confidence`` (binary)

All three are generalized to work with arbitrary N-action spaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from daph_learning.executive.types import (
    ActionSpace,
    ExecutiveState,
    ActionExecution,
    CounterfactualSet,
    UtilityModel,
    UtilityBreakdown,
    ActionDecision,
)


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Generic Experience Builder
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutiveExperience:
    """A single counterfactual experience for N-action learning.

    Generalizes ``CounterfactualExperience`` (which held ``symbolic``
    and ``llm`` BackendOutcomes) to arbitrary actions.

    Attributes
    ----------
    state : ExecutiveState
    utilities : Mapping[str, UtilityBreakdown]
        Per-action utility breakdowns.
    best_action : str
        ``argmax_a U(s, a)`` (the oracle action).
    best_utility : float
    regret : float
        ``best_utility - utility_of_selected``
    sample_weight : float
    """

    state: ExecutiveState
    utilities: dict[str, UtilityBreakdown]
    best_action: str
    best_utility: float
    regret: float
    sample_weight: float

    @property
    def task_id(self) -> str:
        return self.state.task_id

    def utility_vector(self, action_ids: Sequence[str]) -> np.ndarray:
        """Return utilities as a vector in the given action order."""
        return np.array([self.utilities[a].utility for a in action_ids])

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "utilities": {a: bd.to_dict() for a, bd in self.utilities.items()},
            "best_action": self.best_action,
            "best_utility": self.best_utility,
            "regret": self.regret,
            "sample_weight": self.sample_weight,
        }


def build_executive_experiences(
    cf_sets: Sequence[CounterfactualSet],
    utility_model: UtilityModel,
    action_space: ActionSpace,
    *,
    weight_fn: Callable[[CounterfactualSet, dict[str, UtilityBreakdown]], float] | None = None,
) -> list[ExecutiveExperience]:
    """Build executive experiences from counterfactual sets.

    This is the generic equivalent of ``build_counterfactual_experiences``
    in ``policy/learner.py``, generalized to N actions.

    Parameters
    ----------
    cf_sets : Sequence[CounterfactualSet]
    utility_model : UtilityModel
    action_space : ActionSpace
    weight_fn : callable, optional
        ``(cf_set, utilities) → sample_weight``. Default: uniform weight 1.0.

    Returns
    -------
    list[ExecutiveExperience]
    """
    if weight_fn is None:
        weight_fn = lambda cf, u: 1.0

    experiences = []
    for cf in cf_sets:
        breakdowns = utility_model.compute_all(cf)
        best_id = max(breakdowns, key=lambda a: breakdowns[a].utility)
        best_u = breakdowns[best_id].utility

        # Selected action's utility (for regret)
        selected = cf.selected_action
        if selected is not None and selected in breakdowns:
            selected_u = breakdowns[selected].utility
        else:
            selected_u = best_u  # no selection → no regret

        weight = weight_fn(cf, breakdowns)
        experiences.append(ExecutiveExperience(
            state=cf.state,
            utilities=dict(breakdowns),
            best_action=best_id,
            best_utility=best_u,
            regret=best_u - selected_u,
            sample_weight=weight,
        ))
    return experiences


def experiences_to_training_arrays(
    experiences: Sequence[ExecutiveExperience],
    action_space: ActionSpace,
    features: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Convert experiences to training arrays for policy fitting.

    Parameters
    ----------
    experiences : Sequence[ExecutiveExperience]
    action_space : ActionSpace
    features : np.ndarray | None
        Feature matrix ``[N, D]``. If None, only utilities and weights
        are returned (features must be provided separately for fitting).

    Returns
    -------
    dict with keys:
        - "utilities": np.ndarray  shape [N, n_actions]
        - "weights": np.ndarray  shape [N]
        - "best_actions": np.ndarray  shape [N] (action index)
        - "features": np.ndarray  shape [N, D] (if features provided)
    """
    n = len(experiences)
    n_actions = len(action_space.action_ids)
    action_ids = action_space.action_ids

    utilities = np.zeros((n, n_actions), dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    best_actions = np.zeros(n, dtype=np.int64)

    action_to_idx = {a: i for i, a in enumerate(action_ids)}

    for i, exp in enumerate(experiences):
        for j, action_id in enumerate(action_ids):
            utilities[i, j] = exp.utilities[action_id].utility
        weights[i] = exp.sample_weight
        best_actions[i] = action_to_idx.get(exp.best_action, 0)

    result = {
        "utilities": utilities,
        "weights": weights,
        "best_actions": best_actions,
    }
    if features is not None:
        result["features"] = np.asarray(features, dtype=np.float32)
    return result


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Generic Confidence Combination
# ──────────────────────────────────────────────────────────────────────

def combine_action_confidences(
    confidences: dict[str, float],
    mode: str = "product",
) -> float:
    """Combine per-action confidences into a single example confidence.

    Generalizes ``combine_paired_confidence`` (which takes exactly 2
    confidences) to N actions.

    Parameters
    ----------
    confidences : dict[str, float]
        Per-action confidence in ``[0, 1]``.
    mode : str
        ``"product"``, ``"min"``, or ``"geometric_mean"``.
    """
    values = list(confidences.values())
    if not values:
        return 0.0
    for v in values:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {v}")
    if mode == "product":
        result = 1.0
        for v in values:
            result *= v
        return float(result)
    if mode == "min":
        return float(min(values))
    if mode == "geometric_mean":
        log_sum = sum(math.log(max(v, 1e-12)) for v in values)
        return float(math.exp(log_sum / len(values)))
    raise ValueError(f"unknown confidence combine mode: {mode!r}")


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Generic Training Targets
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExecutiveTrainingTargets:
    """Training targets for N-action executive policy.

    Generalizes ``TrainingTargets`` (which held binary ``delta_u``,
    ``weights``, ``targets``) to N actions.

    Attributes
    ----------
    utilities : np.ndarray  shape [N, n_actions]
        Per-action utility for each example.
    weights : np.ndarray  shape [N]
        Sample weights.
    best_action_idx : np.ndarray  shape [N]
        Index of the best action for each example.
    action_ids : tuple[str, ...]
    sigma : np.ndarray | None  shape [N]
        Per-example uncertainty (if computed).
    """

    utilities: np.ndarray
    weights: np.ndarray
    best_action_idx: np.ndarray
    action_ids: tuple[str, ...]
    sigma: np.ndarray | None = None

    @property
    def n_examples(self) -> int:
        return self.utilities.shape[0]

    @property
    def n_actions(self) -> int:
        return self.utilities.shape[1]


def build_executive_training_targets(
    experiences: Sequence[ExecutiveExperience],
    action_space: ActionSpace,
    *,
    abstention_band: float = 0.0,
    weight_mode: str = "uniform",
) -> ExecutiveTrainingTargets:
    """Build training targets from executive experiences.

    Generalizes ``build_uncertainty_aware_targets`` to N actions.

    Parameters
    ----------
    experiences : Sequence[ExecutiveExperience]
    action_space : ActionSpace
    abstention_band : float
        Utility gap below which the example is treated as a tie
        (weight set to 0).
    weight_mode : str
        ``"uniform"`` (all weight 1.0), ``"utility"`` (weight by
        best_utility), or ``"gap"`` (weight by utility gap).
    """
    n = len(experiences)
    n_actions = len(action_space.action_ids)
    action_ids = action_space.action_ids

    utilities = np.zeros((n, n_actions), dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    best_idx = np.zeros(n, dtype=np.int64)

    for i, exp in enumerate(experiences):
        for j, aid in enumerate(action_ids):
            utilities[i, j] = exp.utilities[aid].utility
        best_idx[i] = action_ids.index(exp.best_action)

        # Compute gap (best - second best)
        sorted_u = np.sort(utilities[i])
        gap = sorted_u[-1] - sorted_u[-2] if n_actions > 1 else 1.0

        if gap <= abstention_band:
            weights[i] = 0.0  # tie → no training signal
        elif weight_mode == "uniform":
            weights[i] = 1.0
        elif weight_mode == "utility":
            weights[i] = max(exp.best_utility, 0.0)
        elif weight_mode == "gap":
            weights[i] = gap
        else:
            weights[i] = 1.0

    return ExecutiveTrainingTargets(
        utilities=utilities,
        weights=weights,
        best_action_idx=best_idx,
        action_ids=action_ids,
    )


def estimate_uncertainty(
    utilities: np.ndarray,
    n_replicates: int = 1,
) -> np.ndarray:
    """Estimate per-example uncertainty σ_i for N-action utilities.

    Generalizes ``estimate_sigma`` to N actions. For single replicate,
    σ_i is estimated from the utility gap (larger gaps → lower relative
    uncertainty).

    Parameters
    ----------
    utilities : np.ndarray  shape [N, n_actions] or [N, n_actions, R]
    n_replicates : int

    Returns
    -------
    np.ndarray  shape [N]
    """
    u = np.asarray(utilities, dtype=np.float64)
    if n_replicates > 1 and u.ndim == 3:
        # Variance across replicates
        var_per_action = u.var(axis=-1)  # [N, n_actions]
        return np.sqrt(var_per_action.sum(axis=-1) + 1e-8)

    # Single replicate: σ_i = gap * 0.1 + ε
    sorted_u = np.sort(u, axis=1)
    if sorted_u.shape[1] > 1:
        gap = sorted_u[:, -1] - sorted_u[:, -2]
    else:
        gap = np.ones(u.shape[0])
    return np.abs(gap) * 0.1 + 1e-6


__all__ = [
    "ExecutiveExperience",
    "build_executive_experiences",
    "experiences_to_training_arrays",
    "combine_action_confidences",
    "ExecutiveTrainingTargets",
    "build_executive_training_targets",
    "estimate_uncertainty",
]
