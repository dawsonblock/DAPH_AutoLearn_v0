"""v0.3.10 — calibration metrics (Section 11).

Measure whether router probabilities mean what they claim.

Metrics
-------
Brier score          : ``(1/N) Σ (p_i - y_i)^2``
Expected Calibration : weighted average of per-bin |confidence - accuracy|
Error (ECE)
reliability bins     : per-bin confidence, accuracy, count
selective-risk curve : risk vs. coverage as the abstention threshold varies
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def brier_score(
    probabilities: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[float],
) -> float:
    """Brier score: ``(1/N) Σ (p_i - y_i)^2``.

    ``labels`` may be hard (0/1) or soft (the realized preferred-action
    probability). Lower is better.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must have the same shape")
    if len(p) == 0:
        raise ValueError("empty input")
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin in a reliability diagram."""

    lower: float
    upper: float
    confidence: float  # mean predicted confidence in this bin
    accuracy: float    # mean realized label in this bin
    count: int


def reliability_bins(
    probabilities: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[float],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Bin predictions into equal-width confidence bins.

    ``confidence = max(p, 1-p)``; ``accuracy`` is the mean of the
    realized binary label (1 if the argmax prediction was correct, 0
    otherwise) when labels are hard, or the mean soft label when labels
    are soft.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(p) == 0:
        raise ValueError("empty input")
    confidences = np.maximum(p, 1.0 - p)
    # For hard labels, accuracy = 1 if argmax(p) == y else 0.
    # For soft labels, use the soft label directly.
    if np.all((y == 0) | (y == 1)):
        predictions = (p >= 0.5).astype(np.float64)
        accuracies = (predictions == y).astype(np.float64)
    else:
        accuracies = y
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(ReliabilityBin(
            lower=float(lower), upper=float(upper),
            confidence=float(confidences[mask].mean()),
            accuracy=float(accuracies[mask].mean()),
            count=count,
        ))
    return bins


def expected_calibration_error(
    probabilities: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    ``ECE = Σ_b (n_b / N) |conf_b - acc_b|``
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(p) == 0:
        raise ValueError("empty input")
    bins = reliability_bins(p, y, n_bins=n_bins)
    n = len(p)
    ece = 0.0
    for b in bins:
        ece += (b.count / n) * abs(b.confidence - b.accuracy)
    return float(ece)


@dataclass(frozen=True)
class SelectiveRiskPoint:
    """One point on the selective-risk curve."""

    coverage: float
    risk: float
    threshold: float
    n_selected: int


def selective_risk_curve(
    probabilities: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[float],
    thresholds: np.ndarray | Sequence[float] | None = None,
) -> list[SelectiveRiskPoint]:
    """Selective-risk curve: risk vs. coverage as the abstention threshold
    varies.

    For each confidence threshold ``τ``, select tasks with
    ``max(p, 1-p) >= τ`` and compute the risk (error rate) on the
    selected set. Coverage = fraction selected.

    Parameters
    ----------
    probabilities : array-like
        ``P(S | h)``.
    labels : array-like
        Hard labels (0/1) or soft labels.
    thresholds : array-like | None
        Confidence thresholds to evaluate. Defaults to a linspace from
        0.5 to 1.0.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(p) == 0:
        raise ValueError("empty input")
    confidences = np.maximum(p, 1.0 - p)
    if np.all((y == 0) | (y == 1)):
        predictions = (p >= 0.5).astype(np.float64)
        errors = (predictions != y).astype(np.float64)
    else:
        # soft labels: risk = mean |p - y|
        errors = np.abs(p - y)
    if thresholds is None:
        thresholds = np.linspace(0.5, 1.0, 21)
    n = len(p)
    points: list[SelectiveRiskPoint] = []
    for tau in thresholds:
        selected = confidences >= tau
        n_sel = int(selected.sum())
        if n_sel == 0:
            points.append(SelectiveRiskPoint(
                coverage=0.0, risk=float("nan"), threshold=float(tau), n_selected=0))
            continue
        risk = float(errors[selected].mean())
        points.append(SelectiveRiskPoint(
            coverage=n_sel / n, risk=risk, threshold=float(tau), n_selected=n_sel))
    return points


__all__ = [
    "ReliabilityBin",
    "SelectiveRiskPoint",
    "action_confidence_ece",
    "brier_score",
    "expected_calibration_error",
    "preference_brier_soft",
    "predicted_action_correctness_target",
    "reliability_bins",
    "selective_risk_curve",
    "soft_calibration_error",
]


# ------------------------------------------------------------------
# v0.3.10.1 — Section 8: correct calibration math for soft/tie labels.
# ------------------------------------------------------------------

def predicted_action_correctness_target(
    p_symbolic: np.ndarray | Sequence[float],
    target_symbolic_prob: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Expected correctness probability of the predicted action under a
    soft target (Section 8B).

    Predicted action::

        a_hat_i = S  if p_i >= 0.5
                  L  otherwise

    Expected correctness probability under the soft target::

        c*_i = q_i      if a_hat_i = S
               1 - q_i  if a_hat_i = L

    For hard labels (``q_i ∈ {0, 1}``), ``c*_i`` reduces to ordinary
    0/1 correctness (1 if the predicted action matches the hard label,
    0 otherwise).

    Parameters
    ----------
    p_symbolic : array-like
        Predicted ``P(S | h)``. Shape ``[N]``.
    target_symbolic_prob : array-like
        Soft target ``q_i = P(S)`` (e.g. ``σ(ΔU / τ)``) or hard label
        (``0/1``). Shape ``[N]``.

    Returns
    -------
    np.ndarray
        Shape ``[N]`` expected correctness probabilities in ``[0, 1]``.
    """
    p = np.asarray(p_symbolic, dtype=np.float64)
    q = np.asarray(target_symbolic_prob, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(
            "p_symbolic and target_symbolic_prob must have the same shape")
    if len(p) == 0:
        raise ValueError("empty input")
    choose_symbolic = p >= 0.5
    return np.where(choose_symbolic, q, 1.0 - q)


def preference_brier_soft(
    p_symbolic: np.ndarray | Sequence[float],
    target_symbolic_prob: np.ndarray | Sequence[float],
) -> float:
    """Soft Brier score against the symbolic preference probability
    (Section 8A).

    ``B_soft = (1/N) Σ (p_i - q_i)^2``

    Compares the predicted ``P(S | h)`` directly against the soft
    target ``q = σ(ΔU / τ)``. Lower is better. This is the correct
    probability-calibration metric when targets are soft; the legacy
    :func:`brier_score` against hard 0/1 labels silently throws away
    the continuous preference information.
    """
    p = np.asarray(p_symbolic, dtype=np.float64)
    q = np.asarray(target_symbolic_prob, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(
            "p_symbolic and target_symbolic_prob must have the same shape")
    if len(p) == 0:
        raise ValueError("empty input")
    return float(np.mean((p - q) ** 2))


def soft_calibration_error(
    p_symbolic: np.ndarray | Sequence[float],
    target_symbolic_prob: np.ndarray | Sequence[float],
    n_bins: int = 10,
) -> float:
    """Soft calibration error (Section 8A): bin ``p_i`` against ``q_i``.

    Bins predictions by predicted probability ``p`` (NOT by confidence
    ``max(p, 1-p)``), and within each bin compares the mean predicted
    probability against the mean soft target. This is the
    probability-calibration analogue of ECE for soft targets::

        SCE = Σ_b (n_b / N) |mean(p_b) - mean(q_b)|
    """
    p = np.asarray(p_symbolic, dtype=np.float64)
    q = np.asarray(target_symbolic_prob, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(
            "p_symbolic and target_symbolic_prob must have the same shape")
    if len(p) == 0:
        raise ValueError("empty input")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(p)
    sce = 0.0
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        sce += (count / n) * abs(float(p[mask].mean() - q[mask].mean()))
    return float(sce)


def action_confidence_ece(
    p_symbolic: np.ndarray | Sequence[float],
    target_symbolic_prob: np.ndarray | Sequence[float],
    n_bins: int = 10,
) -> float:
    """Action-confidence ECE (Section 8B).

    For each example:
    * predicted action ``a_hat_i = S if p_i >= 0.5 else L``
    * predicted confidence ``c_hat_i = max(p_i, 1 - p_i)``
    * expected correctness under soft target ``c*_i`` (see
      :func:`predicted_action_correctness_target`)

    ECE = ``Σ_b (n_b / N) |mean(c_hat_b) - mean(c*_b)|`` where bins are
    over ``c_hat`` (confidence). For hard labels, ``c*_i`` reduces to
    ordinary 0/1 correctness, so this reduces to the standard ECE.
    """
    p = np.asarray(p_symbolic, dtype=np.float64)
    q = np.asarray(target_symbolic_prob, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(
            "p_symbolic and target_symbolic_prob must have the same shape")
    if len(p) == 0:
        raise ValueError("empty input")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    confidences = np.maximum(p, 1.0 - p)
    correctness = predicted_action_correctness_target(p, q)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    n = len(p)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / n) * abs(
            float(confidences[mask].mean() - correctness[mask].mean()))
    return float(ece)
