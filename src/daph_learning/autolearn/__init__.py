"""DAPH AutoLearn: iterative steering-vector learning from execution outcomes.

This module implements the learning loop that gives the project its name.
The loop:

1. Routes training tasks using the current steering vector.
2. Executes the chosen backend (symbolic or LLM).
3. Evaluates the outcome against the expected answer.
4. Identifies misrouted tasks (where the route produced a wrong outcome).
5. Re-captures activations from correctly-routed and misrouted tasks.
6. Updates the steering vector via contrastive mean difference.
7. Evaluates on the validation set.
8. Repeats for N iterations.

This is a **genuine learning loop** because it uses execution feedback
(did the answer match the expected value?) to improve the steering vector,
not just hand-coded labels. The steering vector evolves based on what the
system actually gets wrong.

See CLAIMS.md §18 (AutoLearn) for the distinction between this and the
v0.3.4 "extract once, apply forever" approach.
"""

from .loop import (
    AutoLearnConfig,
    AutoLearnResult,
    IterationMetrics,
    OutcomeLabel,
    StopReason,
    CaptureTokenScope,
    classify_outcome,
    run_autolearn_loop,
    _route_with_steering_generate,
    _route_without_steering_generate,
    _evaluate_routes_simple,
)

__all__ = [
    "AutoLearnConfig",
    "AutoLearnResult",
    "IterationMetrics",
    "OutcomeLabel",
    "StopReason",
    "CaptureTokenScope",
    "classify_outcome",
    "run_autolearn_loop",
]

# v0.3.9: the corrected counterfactual engine. Imported lazily to avoid
# circular imports and to keep the legacy engine importable on its own.
def __getattr__(name: str):  # noqa: D401
    if name in {
        "CounterfactualLearningLoop",
        "LoopConfig",
        "LoopIterationResult",
        "SteeringPolicyConfig",
        "PolicyRouteDecision",
        "RoutePolicyFn",
        "make_feature_route_policy",
    }:
        from . import trust_region, route_policy
        _exports = {
            "CounterfactualLearningLoop": trust_region.CounterfactualLearningLoop,
            "LoopConfig": trust_region.LoopConfig,
            "LoopIterationResult": trust_region.LoopIterationResult,
            "SteeringPolicyConfig": route_policy.SteeringPolicyConfig,
            "PolicyRouteDecision": route_policy.PolicyRouteDecision,
            "RoutePolicyFn": route_policy.RoutePolicyFn,
            "make_feature_route_policy": route_policy.make_feature_route_policy,
        }
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
