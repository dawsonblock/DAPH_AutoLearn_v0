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
    "brier_score",
    "expected_calibration_error",
    "reliability_bins",
    "selective_risk_curve",
]
