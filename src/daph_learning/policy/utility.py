"""v0.3.10.5-alpha — canonical backend utility function (Section 3, 8).

This is the ONE place where ``U_b`` is computed from a
:class:`~daph_learning.policy.types.BackendOutcome` and a
:class:`UtilityConfig` (Section 8) or a legacy
:class:`~daph_learning.policy.config.ExperimentConfig`.

Section 8 adds a frozen :class:`UtilityConfig` dataclass with an explicit
canonical formula, a ``utility_config_hash``, and two separated protocols:

* ``gate_a_accuracy_primary`` — correctness-dominant utility with a small,
  preregistered cost term. The AUTHORITATIVE primary endpoint.
* ``gate_a_cost_sensitive_secondary`` — operational utility with a larger
  compute penalty. Secondary endpoint only.

Gate A qualification identifies ONE authoritative primary endpoint before
the run. The report may include both, but it must not switch to whichever
one passes.

All other code must call :func:`compute_utility` (or the
:class:`ExperimentConfig`-typed :func:`backend_utility`) instead of
duplicating the formula. This prevents drift between the
experience-construction path and the evaluation path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import ExperimentConfig
from .types import BackendOutcome, CounterfactualExperience, Route


# ------------------------------------------------------------------
# Section 8: frozen UtilityConfig
# ------------------------------------------------------------------

@dataclass(frozen=True)
class UtilityConfig:
    """Section 8 — frozen utility configuration.

    The canonical utility is::

        U_a(x) = w_A * A_a(x)
                 - w_F * F_a(x)
                 - w_L * L_tilde_a(x)
                 - w_C * C_a
                 - w_U * V_a(x)
                 - w_T * T_a(x)

    where:

    * ``A_a`` is verified correctness (accuracy_weight);
    * ``F_a`` is execution failure (failure_penalty);
    * ``L_tilde_a`` is normalized latency (latency_weight / latency_normalizer_ms);
    * ``C_a`` is normalized compute cost (compute_weight, compute_cost_symbolic/llm);
    * ``V_a`` is unverifiable status (unverifiable_penalty);
    * ``T_a`` is timeout status (timeout_penalty).

    The config is frozen before development-model selection and final
    evaluation. Its hash is recorded in every artifact.
    """

    accuracy_weight: float = 1.0
    failure_penalty: float = 1.0
    latency_weight: float = 0.001
    compute_weight: float = 0.1
    unverifiable_penalty: float = 0.5
    timeout_penalty: float = 0.8
    latency_normalizer_ms: float = 1000.0
    compute_cost_symbolic: float = 0.05
    compute_cost_llm: float = 0.20
    utility_clip_min: float = -1.0
    utility_clip_max: float = 1.0

    def __post_init__(self) -> None:
        if self.latency_normalizer_ms <= 0:
            raise ValueError("latency_normalizer_ms must be > 0")
        if self.compute_cost_symbolic <= 0 or self.compute_cost_llm <= 0:
            raise ValueError("compute costs must be > 0")
        if self.utility_clip_min > self.utility_clip_max:
            raise ValueError("utility_clip_min must be <= utility_clip_max")
        for name in ("accuracy_weight", "failure_penalty", "latency_weight",
                     "compute_weight", "unverifiable_penalty",
                     "timeout_penalty"):
            val = getattr(self, name)
            if val < 0 or math.isnan(val) or math.isinf(val):
                raise ValueError(f"{name} must be a finite non-negative number")

    @property
    def utility_config_hash(self) -> str:
        """Section 8: SHA-256 of the canonical JSON serialization."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UtilityConfig":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# Section 8 — two explicitly separated protocols. The primary is
# AUTHORITATIVE; the secondary is reported alongside but must not be
# switched to whichever passes.
GATE_A_ACCURACY_PRIMARY = UtilityConfig(
    accuracy_weight=1.0,
    failure_penalty=1.0,
    latency_weight=0.0005,        # small, preregistered cost term
    compute_weight=0.02,
    unverifiable_penalty=0.5,
    timeout_penalty=0.8,
    latency_normalizer_ms=1000.0,
    compute_cost_symbolic=0.05,
    compute_cost_llm=0.20,
    utility_clip_min=-1.0,
    utility_clip_max=1.0,
)

GATE_A_COST_SENSITIVE_SECONDARY = UtilityConfig(
    accuracy_weight=1.0,
    failure_penalty=1.0,
    latency_weight=0.005,         # larger compute/latency penalty
    compute_weight=0.3,
    unverifiable_penalty=0.5,
    timeout_penalty=0.8,
    latency_normalizer_ms=1000.0,
    compute_cost_symbolic=0.05,
    compute_cost_llm=0.20,
    utility_clip_min=-1.0,
    utility_clip_max=1.0,
)

PROTOCOL_PRESETS: dict[str, UtilityConfig] = {
    "gate_a_accuracy_primary": GATE_A_ACCURACY_PRIMARY,
    "gate_a_cost_sensitive_secondary": GATE_A_COST_SENSITIVE_SECONDARY,
}


def get_protocol(name: str) -> UtilityConfig:
    """Look up a frozen protocol preset by name."""
    if name not in PROTOCOL_PRESETS:
        raise ValueError(
            f"unknown utility protocol {name!r}; expected one of "
            f"{sorted(PROTOCOL_PRESETS)}")
    return PROTOCOL_PRESETS[name]


# ------------------------------------------------------------------
# Section 8: compute_utility (the canonical BackendOutcome entry point)
# ------------------------------------------------------------------

def compute_utility(outcome: BackendOutcome, config: UtilityConfig) -> float:
    """Section 8 — canonical utility from a BackendOutcome + UtilityConfig.

    ``U_a(x) = w_A * A - w_F * F - w_L * L_tilde - w_C * C - w_U * V - w_T * T``

    * ``A`` = 1.0 if verified correct, else 0.0;
    * ``F`` = 1.0 if execution failed, else 0.0;
    * ``L_tilde`` = latency_ms / latency_normalizer_ms;
    * ``C`` = compute_cost_symbolic if backend == symbolic else compute_cost_llm;
    * ``V`` = 1.0 if UNVERIFIABLE, else 0.0;
    * ``T`` = 1.0 if TIMEOUT, else 0.0.

    The result is clipped to ``[utility_clip_min, utility_clip_max]``.
    """
    # Verified correctness.
    correct = 1.0 if outcome.correct else 0.0
    # Execution failure.
    failed = 1.0 if not outcome.execution_success else 0.0
    # Normalized latency.
    latency_ms = outcome.latency_sec * 1000.0
    latency_norm = latency_ms / config.latency_normalizer_ms
    # Normalized compute cost (backend-dependent).
    backend = getattr(outcome, "backend", "")
    if backend == "symbolic":
        compute = config.compute_cost_symbolic
    else:
        compute = config.compute_cost_llm
    # Unverifiable / timeout status from verifier_status.
    vstatus = str(getattr(outcome, "verifier_status", "")).lower()
    unverifiable = 1.0 if "unverifiable" in vstatus else 0.0
    timeout = 1.0 if "timeout" in vstatus else 0.0

    u = (
        config.accuracy_weight * correct
        - config.failure_penalty * failed
        - config.latency_weight * latency_norm
        - config.compute_weight * compute
        - config.unverifiable_penalty * unverifiable
        - config.timeout_penalty * timeout
    )
    # Clip.
    if u < config.utility_clip_min:
        u = config.utility_clip_min
    if u > config.utility_clip_max:
        u = config.utility_clip_max
    if math.isnan(u) or math.isinf(u):
        raise ValueError(f"compute_utility produced non-finite value: {u}")
    return u


# ------------------------------------------------------------------
# Legacy Section 3 formula (kept for backward compatibility; delegates
# to the same canonical definition so the formula cannot drift).
# ------------------------------------------------------------------

def utility_formula(
    *,
    quality: float,
    latency_ms: float,
    compute_cost: float,
    risk: float,
    quality_weight: float,
    lambda_time: float,
    lambda_compute: float,
    lambda_risk: float,
    time_reference_ms: float,
    compute_reference: float,
) -> float:
    """The single implementation of the canonical utility ``U_b`` (Section 3).

    ``U_b = w_q * quality - λ_t*(latency_ms/t_ref) - λ_c*(cost/c_ref) - λ_r*risk``

    Both :func:`backend_utility` (the ``BackendOutcome`` path) and
    ``daph_learning.autolearn.counterfactual.compute_backend_utility``
    (the ``BackendExecutionRecord`` path) delegate to this function so
    the formula cannot drift between the experience-construction and
    evaluation paths.
    """
    time_term = lambda_time * (latency_ms / time_reference_ms)
    compute_term = lambda_compute * (compute_cost / compute_reference)
    risk_term = lambda_risk * risk
    return quality_weight * quality - time_term - compute_term - risk_term


def backend_utility(outcome: BackendOutcome, cfg: ExperimentConfig) -> float:
    """Compute ``U_b`` from a :class:`BackendOutcome` and legacy config.

    This is the canonical ``BackendOutcome``-typed entry point for the
    legacy :class:`ExperimentConfig` path. New Gate A code should prefer
    :func:`compute_utility` with a :class:`UtilityConfig`.
    """
    return utility_formula(
        quality=outcome.quality,
        latency_ms=outcome.latency_sec * 1000.0,
        compute_cost=outcome.normalized_cost,
        risk=outcome.risk,
        quality_weight=cfg.quality_weight,
        lambda_time=cfg.lambda_time,
        lambda_compute=cfg.lambda_compute,
        lambda_risk=cfg.lambda_risk,
        time_reference_ms=cfg.time_reference_ms,
        compute_reference=cfg.compute_reference,
    )


def utility_for_route(
    experience: CounterfactualExperience,
    route: Route | str,
    config: ExperimentConfig,
) -> float:
    """Look up the canonical utility for a route on a given experience
    (Section 3).

    Centralizes route-utility lookup so that training, dev evaluation,
    calibration, final evaluation, promotion, steering experiments,
    random controls, and hand-router baselines all use the same
    :func:`backend_utility` definition rather than ad-hoc
    ``cfg.quality_weight * quality`` shortcuts.

    Parameters
    ----------
    experience : CounterfactualExperience
        The pre-computed counterfactual experience (both backend
        outcomes already captured + verified).
    route : Route | str
        ``"symbolic"``, ``"llm"``, or ``"abstain"``.
    config : ExperimentConfig
        Canonical config carrying the utility weights.

    Returns
    -------
    float
        ``U_b`` for the selected backend, or ``0.0`` for abstain
        (abstention earns no quality reward by definition).
    """
    if isinstance(route, Route):
        route = route.value
    if route == "abstain":
        return 0.0
    if route == "symbolic":
        return backend_utility(experience.symbolic, config)
    if route == "llm":
        return backend_utility(experience.llm, config)
    raise ValueError(f"unknown route {route!r}; expected symbolic|llm|abstain")


__all__ = [
    "GATE_A_ACCURACY_PRIMARY",
    "GATE_A_COST_SENSITIVE_SECONDARY",
    "PROTOCOL_PRESETS",
    "UtilityConfig",
    "backend_utility",
    "compute_utility",
    "get_protocol",
    "utility_for_route",
    "utility_formula",
]

