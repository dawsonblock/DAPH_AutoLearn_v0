"""v0.3.9 — routing policy contract for the counterfactual learning loop.

This module formalizes the explicit routing-policy interface required by the
corrected AutoLearn architecture (Section 2 of the v0.3.9 repair spec).

The trust-region optimizer must NOT be coupled directly to a specific model
implementation. Instead, the loop calls a ``RoutePolicyFn`` that maps
``(tasks, steering_vector, config) -> [PolicyRouteDecision]``. This keeps the
loop model-optional: a toy feature-based router can be injected for tests,
while a real deployment injects a steered-router adapter that wraps the
HuggingFace model path in :mod:`daph_learning.routing.policy`.

Key design rule (Section 1 of the repair spec):

    training:   Delta-U  ->  target
    evaluation: candidate vector  ->  candidate decision
               incumbent vector   ->  incumbent decision

The oracle / counterfactual Delta-U is allowed to provide the *training
target*. It is NOT allowed to provide the candidate's held-out routing
decision. That decision must come from the ``RoutePolicyFn`` applied to the
candidate or incumbent vector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# Reuse the canonical route targets from the counterfactual module.
from .counterfactual import ABSTAIN

RouteTarget = str  # "symbolic" | "llm" | "abstain"
_VALID_ROUTE_TARGETS: frozenset[str] = frozenset({"symbolic", "llm", "abstain"})


@dataclass(frozen=True)
class SteeringPolicyConfig:
    """Canonical, frozen runtime configuration for a steering-based route
    policy.

    This is the single source of truth for the lineage fields that were
    previously hard-coded (``layer=24``, ``token_location="anchor"``,
    ``alpha=1.0``). The lineage artifact is generated from this config so the
    recorded layer/alpha always reflects what was actually used.

    Attributes
    ----------
    layer : int
        Transformer layer index where the steering vector is applied.
    token_location : str
        Capture / intervention scope (e.g. ``"anchor"``, ``"last"``,
        ``"mean_anchor_span"``, ``"mean_sequence"``).
    alpha : float
        Steering scale applied at intervention time.
    decay_policy : str
        How the steering vector decays across tokens (e.g. ``"constant"``,
        ``"linear"``, ``"exponential"``).
    norm_clamp : float | None
        Maximum L2 norm the steering vector is clamped to before application.
        ``None`` means no clamp.
    cosine_clamp : float | None
        Maximum cosine shift between the incumbent and candidate vector
        permitted before the update is rejected. ``None`` means no clamp.
    model_revision : str | None
        HuggingFace model revision hash/branch used at runtime.
    tokenizer_revision : str | None
        HuggingFace tokenizer revision hash/branch used at runtime.
    vector_version : str
        Version label for the steering vector format/schema.
    """

    layer: int = 24
    token_location: str = "anchor"
    alpha: float = 1.0
    decay_policy: str = "constant"
    norm_clamp: float | None = None
    cosine_clamp: float | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    vector_version: str = "v0.3.9"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyClampTelemetry:
    """Telemetry from safety clamps applied during steering.

    Records whether the steering vector was clamped (norm or cosine) before
    being applied, and the pre/post clamp values.
    """

    norm_clamped: bool = False
    norm_before: float | None = None
    norm_after: float | None = None
    cosine_clamped: bool = False
    cosine_before: float | None = None
    cosine_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyRouteDecision:
    """A routing decision produced by a :data:`RoutePolicyFn`.

    Carries enough information for provenance and evaluation (Section 2 of the
    repair spec). Unlike :class:`daph_learning.routing.policy.RouteDecision`
    (the policy-level dispatcher record), this struct is the *learning-loop*
    decision record and includes the steering-vector identifier, model
    revision, layer, intervention location, alpha, and safety-clamp telemetry
    so the lineage artifact can be generated from actual runtime configuration.

    Attributes
    ----------
    task_id : str
        Identifier of the task this decision applies to.
    route : RouteTarget
        One of ``"symbolic"``, ``"llm"``, ``"abstain"``.
    confidence : float | None
        Policy confidence in ``[0, 1]``. ``None`` means not reported.
    abstained : bool
        ``True`` iff ``route == "abstain"``.
    vector_id : str | None
        Identifier of the steering vector that produced this decision.
    model_revision : str | None
        Model revision used (from :class:`SteeringPolicyConfig`).
    tokenizer_revision : str | None
        Tokenizer revision used.
    layer : int
        Layer where steering was applied.
    token_location : str
        Intervention / capture scope.
    alpha : float
        Steering scale.
    safety_clamp : SafetyClampTelemetry
        Telemetry from any safety clamps applied.
    raw_scores : dict[str, float]
        Per-backend raw scores that produced the decision, if available.
    """

    task_id: str
    route: RouteTarget
    confidence: float | None = None
    abstained: bool = False
    vector_id: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    layer: int = 24
    token_location: str = "anchor"
    alpha: float = 1.0
    safety_clamp: SafetyClampTelemetry = field(default_factory=SafetyClampTelemetry)
    raw_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.route not in _VALID_ROUTE_TARGETS:
            raise ValueError(
                f"route must be one of {sorted(_VALID_ROUTE_TARGETS)}, "
                f"got {self.route!r}"
            )
        # abstained must be consistent with route
        if self.route == ABSTAIN and not self.abstained:
            # Use object.__setattr__ because the dataclass is frozen.
            object.__setattr__(self, "abstained", True)
        if self.route != ABSTAIN and self.abstained:
            object.__setattr__(self, "abstained", False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["safety_clamp"] = self.safety_clamp.to_dict()
        return d


# The canonical routing-policy function type.
#
#   RoutePolicyFn(tasks, steering_vector, config) -> [PolicyRouteDecision]
#
# ``steering_vector`` is a 1-D numpy array. ``config`` is a
# :class:`SteeringPolicyConfig`. The function must return one decision per
# task, in input order, and MUST base the route on the supplied vector — not
# on oracle/counterfactual utility. This is the contract that makes the
# candidate-vs-incumbent evaluation real (Section 1/3 of the repair spec).
RoutePolicyFn = Callable[
    [Sequence[Mapping[str, Any]], np.ndarray, SteeringPolicyConfig],
    Sequence[PolicyRouteDecision],
]


def make_feature_route_policy(
    feature_fn: Callable[[Mapping[str, Any]], np.ndarray],
    *,
    abstain_threshold: float = 0.0,
) -> RoutePolicyFn:
    """Build a model-optional :data:`RoutePolicyFn` from a feature function.

    The route is determined by the sign of ``dot(steering_vector, features)``:

    - ``score > abstain_threshold``  -> ``"symbolic"``
    - ``score < -abstain_threshold`` -> ``"llm"``
    - otherwise                      -> ``"abstain"``

    This is the toy/deterministic router used by the synthetic tests and the
    vector-sensitivity / oracle-leakage tests. It guarantees that changing the
    steering vector changes the routing decision, which is the acceptance
    condition for the corrected loop.

    Parameters
    ----------
    feature_fn : callable
        ``feature_fn(task) -> np.ndarray`` (1-D, same dim as the steering
        vector).
    abstain_threshold : float
        Band around zero within which the policy abstains.
    """

    def _route(
        tasks: Sequence[Mapping[str, Any]],
        steering_vector: np.ndarray,
        config: SteeringPolicyConfig,
    ) -> list[PolicyRouteDecision]:
        vec = np.asarray(steering_vector, dtype=np.float32)
        decisions: list[PolicyRouteDecision] = []
        for task in tasks:
            task_id = str(task.get("task_id", ""))
            feats = np.asarray(feature_fn(task), dtype=np.float32)
            if feats.shape != vec.shape:
                raise ValueError(
                    f"feature dim {feats.shape} != steering vector dim {vec.shape} "
                    f"for task {task_id!r}"
                )
            score = float(np.dot(vec, feats))
            if score > abstain_threshold:
                route: RouteTarget = "symbolic"
            elif score < -abstain_threshold:
                route = "llm"
            else:
                route = ABSTAIN
            confidence = float(1.0 / (1.0 + np.exp(-abs(score)))) if score != 0.0 else 0.5
            decisions.append(
                PolicyRouteDecision(
                    task_id=task_id,
                    route=route,
                    confidence=confidence,
                    vector_id=config.vector_version,
                    model_revision=config.model_revision,
                    tokenizer_revision=config.tokenizer_revision,
                    layer=config.layer,
                    token_location=config.token_location,
                    alpha=config.alpha,
                    raw_scores={"dot_score": score},
                )
            )
        return decisions

    return _route


__all__ = [
    "PolicyRouteDecision",
    "RoutePolicyFn",
    "RouteTarget",
    "SafetyClampTelemetry",
    "SteeringPolicyConfig",
    "make_feature_route_policy",
]
