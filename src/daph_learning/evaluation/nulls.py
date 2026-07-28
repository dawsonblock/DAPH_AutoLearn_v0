"""Finite-sample empirical null statistics for steering controls."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EmpiricalNullResult:
    observed: float
    n_null: int
    exceedances: int
    empirical_p_value: float
    observed_percentile: float
    null_mean: float
    null_median: float
    null_interval_95: tuple[float, float]
    higher_is_better: bool
    protocol_eligible: bool
    minimum_protocol_nulls: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["null_interval_95"] = list(self.null_interval_95)
        return payload


def empirical_null_test(
    observed: float,
    null_scores: Sequence[float],
    *,
    higher_is_better: bool = True,
    minimum_protocol_nulls: int = 500,
) -> EmpiricalNullResult:
    """Compare one observed metric to an exchangeable Monte Carlo null.

    Uses the finite-sample correction

    ``p = (1 + exceedances) / (N + 1)``

    so zero observed exceedances never produces an impossible p-value of zero.
    """
    if minimum_protocol_nulls <= 0:
        raise ValueError("minimum_protocol_nulls must be positive")
    values = np.asarray(list(null_scores), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("null_scores must be a non-empty one-dimensional sequence")
    if not math.isfinite(float(observed)) or not np.isfinite(values).all():
        raise ValueError("observed and null scores must all be finite")

    if higher_is_better:
        exceedances = int(np.sum(values >= float(observed)))
        below = float(np.sum(values < float(observed)))
        equal = float(np.sum(values == float(observed)))
    else:
        exceedances = int(np.sum(values <= float(observed)))
        below = float(np.sum(values > float(observed)))
        equal = float(np.sum(values == float(observed)))
    p_value = (1.0 + exceedances) / (values.size + 1.0)
    percentile = 100.0 * (below + 0.5 * equal) / values.size
    interval = np.quantile(values, [0.025, 0.975])

    return EmpiricalNullResult(
        observed=float(observed),
        n_null=int(values.size),
        exceedances=exceedances,
        empirical_p_value=float(p_value),
        observed_percentile=float(percentile),
        null_mean=float(np.mean(values)),
        null_median=float(np.median(values)),
        null_interval_95=(float(interval[0]), float(interval[1])),
        higher_is_better=bool(higher_is_better),
        protocol_eligible=bool(values.size >= minimum_protocol_nulls),
        minimum_protocol_nulls=minimum_protocol_nulls,
    )


__all__ = ["EmpiricalNullResult", "empirical_null_test"]
