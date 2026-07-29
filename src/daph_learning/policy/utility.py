"""v0.3.10.3-alpha — canonical backend utility function (Section 13).

This is the ONE place where ``U_b`` is computed from a
:class:`~daph_learning.policy.types.BackendOutcome` and an
:class:`~daph_learning.policy.config.ExperimentConfig`.

All other code must call :func:`backend_utility` instead of
duplicating the formula. This prevents drift between the
experience-construction path and the evaluation path.
"""

from __future__ import annotations

from .config import ExperimentConfig
from .types import BackendOutcome


def backend_utility(outcome: BackendOutcome, cfg: ExperimentConfig) -> float:
    """Compute ``U_b`` from a :class:`BackendOutcome` and config.

    Formula (Section 13)::

        U_b = w_q * quality
              - λ_t * (latency_ms / t_ref)
              - λ_c * (cost / c_ref)
              - λ_r * risk

    This is the canonical implementation. Do not duplicate the formula
    elsewhere — import and call this function.
    """
    t = outcome.latency_sec * 1000.0  # back to ms
    time_term = cfg.lambda_time * (t / cfg.time_reference_ms)
    compute_term = cfg.lambda_compute * (
        outcome.normalized_cost / cfg.compute_reference)
    risk_term = cfg.lambda_risk * outcome.risk
    return (cfg.quality_weight * outcome.quality
            - time_term - compute_term - risk_term)


__all__ = ["backend_utility"]
