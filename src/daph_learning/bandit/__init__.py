"""v0.3.10 — contextual bandit-compatible logging (Section 20) and the
doubly-robust interface (Section 21).

During this release both backends can still be executed for training.
However, prepare AutoLearn for eventual partial feedback. Every
deployment routing decision must log the chosen action, the probability
of the chosen action, the full policy probabilities, the policy version,
and the available actions. This is essential for future inverse
propensity scoring (IPS) and doubly-robust (DR) estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..policy.types import PolicyDecision, Route


def log_policy_decision(
    task_id: str,
    action: Route,
    probabilities: dict[str, float],
    policy_version: str,
    available_actions: tuple[str, ...] = ("symbolic", "llm", "abstain"),
) -> PolicyDecision:
    """Create a :class:`PolicyDecision` log record (Section 20)."""
    return PolicyDecision(
        task_id=task_id,
        action=action,
        probabilities=dict(probabilities),
        policy_version=policy_version,
        available_actions=available_actions,
    )


def inverse_propensity_weight(
    observed_reward: float,
    propensity: float,
    *,
    clip: float | None = None,
) -> float:
    """Inverse propensity scoring (IPS) weight for a single observation.

    ``IPS = R / p(a|x)``

    Parameters
    ----------
    observed_reward : float
        The realized reward for the chosen action.
    propensity : float
        The probability with which the action was chosen (logged in
        :class:`PolicyDecision`). Must be > 0.
    clip : float | None
        Optional upper clip on ``1 / propensity`` to bound variance.
    """
    if propensity <= 0:
        raise ValueError("propensity must be > 0 for IPS")
    weight = 1.0 / propensity
    if clip is not None:
        weight = min(weight, clip)
    return observed_reward * weight


def doubly_robust_utility(
    *,
    u_hat: float,
    observed_action: str,
    observed_reward: float | None,
    propensity: float,
    target_action: str,
) -> float:
    """Doubly-robust utility estimate (Section 21).

    Conceptually::

        U_DR = U_hat(x, a)
               + I[a = observed_action] / p(a|x)
                 * (U_observed - U_hat(x, a))

    Parameters
    ----------
    u_hat : float
        The model's predicted utility for the target action.
    observed_action : str
        The action that was actually taken (logged).
    observed_reward : float | None
        The realized reward for the observed action. ``None`` if not
        observed (partial feedback).
    propensity : float
        The probability with which ``observed_action`` was chosen.
    target_action : str
        The action whose utility we want to estimate.

    Returns
    -------
    float
        The doubly-robust utility estimate.

    Notes
    -----
    Do NOT claim this is valid unless propensities are actually logged and
    overlap conditions hold (Section 21).
    """
    if propensity <= 0:
        raise ValueError("propensity must be > 0 for DR estimation")
    if observed_action == target_action and observed_reward is not None:
        correction = (observed_reward - u_hat) / propensity
        return u_hat + correction
    return u_hat


__all__ = [
    "doubly_robust_utility",
    "inverse_propensity_weight",
    "log_policy_decision",
]
