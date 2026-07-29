"""v0.3.10.1-alpha — calibrated abstention (Sections 10, 21).

ABSTAIN must not exist only because ``|ΔU| <= ε`` (that defines the
training target, not deployment uncertainty). At inference time the
controller abstains when it is insufficiently confident in its routing
probability, OR when the input is out-of-distribution.

For binary policy probability ``p = P(S | h)``::

    conf = max(p, 1 - p)
    if OOD(h) > tau_ood:        action = ABSTAIN  (reason: ood)
    elif conf < tau_conf:       action = ABSTAIN  (reason: low_confidence)
    elif p >= 0.5:              action = SYMBOLIC
    else:                       action = LLM

v0.3.10.1 — adds :func:`choose_route_with_reason` so abstention reasons
are recorded explicitly (Section 21). Reasons:

* ``low_confidence``   — conf < tau_conf
* ``ood``              — OOD(h) > tau_ood
* ``policy_tie``       — p exactly 0.5 and conf >= tau_conf (rare)
* ``execution_failure`` — set by the caller when the backend fails
* ``safety_gate``      — set by the caller when a safety gate trips

Do not collapse all abstentions into one opaque category.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Route


@dataclass(frozen=True)
class RouteDecision:
    """The result of choosing a route at inference time (Section 21).

    Attributes
    ----------
    route : Route
    reason : str
        Why this route was chosen. One of:
        ``"symbolic"``, ``"llm"``, ``"low_confidence"``, ``"ood"``,
        ``"policy_tie"``.
    confidence : float
        ``max(p, 1 - p)``.
    p_symbolic : float
        The input ``P(S | h)``.
    ood_score : float | None
        OOD score if an OOD check was performed, else None.
    """

    route: Route
    reason: str
    confidence: float
    p_symbolic: float
    ood_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "route": self.route.value,
            "reason": self.reason,
            "confidence": float(self.confidence),
            "p_symbolic": float(self.p_symbolic),
            "ood_score": None if self.ood_score is None else float(self.ood_score),
        }


def choose_route(
    p_symbolic: float,
    confidence_threshold: float = 0.70,
) -> Route:
    """Choose a route from ``P(S | h)`` with calibrated abstention.

    Parameters
    ----------
    p_symbolic : float
        ``P(S | h)`` in ``[0, 1]``.
    confidence_threshold : float
        ``τ_conf``. Abstain when ``max(p, 1-p) < τ_conf``. Must be in
        ``[0.5, 1.0]`` (below 0.5 every decision would abstain, which is
        degenerate).

    Returns
    -------
    Route
        ``SYMBOLIC``, ``LLM``, or ``ABSTAIN``.
    """
    if not 0.0 <= p_symbolic <= 1.0:
        raise ValueError("invalid probability")
    if not 0.5 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0.5, 1.0]")
    confidence = max(p_symbolic, 1.0 - p_symbolic)
    if confidence < confidence_threshold:
        return Route.ABSTAIN
    if p_symbolic >= 0.5:
        return Route.SYMBOLIC
    return Route.LLM


def choose_route_with_reason(
    p_symbolic: float,
    confidence_threshold: float = 0.70,
    *,
    ood_score: float | None = None,
    ood_threshold: float = float("inf"),
) -> RouteDecision:
    """Choose a route with an explicit abstention reason (Section 21).

    Parameters
    ----------
    p_symbolic : float
        ``P(S | h)`` in ``[0, 1]``.
    confidence_threshold : float
        ``τ_conf``. Must be in ``[0.5, 1.0]``.
    ood_score : float | None
        OOD score for this input. If ``None``, the OOD check is skipped.
    ood_threshold : float
        ``τ_ood``. Abstain when ``ood_score > ood_threshold``. Default
        ``+inf`` disables the OOD check (useful for tests; production
        runs should calibrate this — Section 19).

    Returns
    -------
    RouteDecision
        With ``route`` and an explicit ``reason``.
    """
    if not 0.0 <= p_symbolic <= 1.0:
        raise ValueError("invalid probability")
    if not 0.5 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0.5, 1.0]")
    if ood_threshold < 0:
        raise ValueError("ood_threshold must be >= 0")
    confidence = max(p_symbolic, 1.0 - p_symbolic)
    if ood_score is not None and ood_score > ood_threshold:
        return RouteDecision(
            route=Route.ABSTAIN, reason="ood",
            confidence=confidence, p_symbolic=p_symbolic,
            ood_score=ood_score,
        )
    if confidence < confidence_threshold:
        return RouteDecision(
            route=Route.ABSTAIN, reason="low_confidence",
            confidence=confidence, p_symbolic=p_symbolic,
            ood_score=ood_score,
        )
    # Confidence >= threshold. Handle exact tie at p=0.5.
    if p_symbolic == 0.5:
        return RouteDecision(
            route=Route.ABSTAIN, reason="policy_tie",
            confidence=confidence, p_symbolic=p_symbolic,
            ood_score=ood_score,
        )
    if p_symbolic > 0.5:
        return RouteDecision(
            route=Route.SYMBOLIC, reason="symbolic",
            confidence=confidence, p_symbolic=p_symbolic,
            ood_score=ood_score,
        )
    return RouteDecision(
        route=Route.LLM, reason="llm",
        confidence=confidence, p_symbolic=p_symbolic,
        ood_score=ood_score,
    )


__all__ = ["RouteDecision", "choose_route", "choose_route_with_reason"]
