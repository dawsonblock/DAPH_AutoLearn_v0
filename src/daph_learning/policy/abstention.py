"""v0.3.10 — calibrated abstention (Section 10).

ABSTAIN must not exist only because ``|ΔU| <= ε`` (that defines the
training target, not deployment uncertainty). At inference time the
controller abstains when it is insufficiently confident in its routing
probability.

For binary policy probability ``p = P(S | h)``::

    conf = max(p, 1 - p)
    if conf < τ_conf:  action = ABSTAIN
    elif p >= 0.5:     action = SYMBOLIC
    else:              action = LLM

``τ_conf`` is tuned on the calibration split (Section 34), never on final
test.
"""

from __future__ import annotations

from .types import Route


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


__all__ = ["choose_route"]
