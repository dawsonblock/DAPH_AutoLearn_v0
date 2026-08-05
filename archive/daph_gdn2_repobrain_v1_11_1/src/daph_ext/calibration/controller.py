from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import numpy as np


@dataclass(frozen=True)
class CalibrationRecord:
    kl_drift: float
    generic_regression: float
    execution_regression: float
    repository_benefit: float
    repo_id: str | None = None

    def validate(self) -> None:
        for name in ("kl_drift", "generic_regression", "execution_regression", "repository_benefit"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.repo_id is not None and not self.repo_id.strip():
            raise ValueError("repo_id must be non-empty when provided")
        if self.kl_drift < 0:
            raise ValueError("kl_drift must be >= 0")


@dataclass(frozen=True)
class KLCalibrationResult:
    threshold: float
    safe_count: int
    total_count: int
    percentile: float
    ci_low: float
    ci_high: float
    repository_count: int
    min_repository_benefit: float

    @property
    def safe_fraction(self) -> float:
        return self.safe_count / self.total_count if self.total_count else 0.0


def calibrate_kl_envelope(
    records: Sequence[CalibrationRecord],
    *,
    percentile: float = 95.0,
    max_generic_regression: float = 0.01,
    max_execution_regression: float = 0.01,
    min_repository_benefit: float = 0.0,
    min_safe_count: int = 1,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> KLCalibrationResult:
    """Calibrate KL only from adapters that are both safe and useful.

    A percentile without uncertainty is unstable at small n, so the result
    includes a bootstrap confidence interval. Production promotion should use
    substantially more than the pilot minimum (recommended >=30, preferably >=50).
    """
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0,100]")
    if min_safe_count <= 0 or bootstrap_samples <= 0:
        raise ValueError("min_safe_count and bootstrap_samples must be > 0")
    if max_generic_regression < 0 or max_execution_regression < 0 or min_repository_benefit < 0:
        raise ValueError("regression/benefit thresholds must be >= 0")
    safe_records: list[CalibrationRecord] = []
    for r in records:
        r.validate()
        if (r.generic_regression <= max_generic_regression
                and r.execution_regression <= max_execution_regression
                and r.repository_benefit > min_repository_benefit):
            safe_records.append(r)
    if len(safe_records) < min_safe_count:
        raise ValueError(f"KL calibration requires at least {min_safe_count} safe+useful validation adapters; got {len(safe_records)}")
    safe = np.asarray([r.kl_drift for r in safe_records], dtype=np.float64)
    threshold = float(np.percentile(safe, percentile))
    rng = np.random.default_rng(seed)
    boots = np.empty(bootstrap_samples, dtype=np.float64)
    for i in range(bootstrap_samples):
        sample = rng.choice(safe, size=safe.size, replace=True)
        boots[i] = np.percentile(sample, percentile)
    ci_low, ci_high = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    repo_ids = {r.repo_id for r in safe_records if r.repo_id is not None}
    repository_count = len(repo_ids) if repo_ids else len(safe_records)
    return KLCalibrationResult(
        threshold=threshold, safe_count=len(safe_records), total_count=len(records),
        percentile=percentile, ci_low=ci_low, ci_high=ci_high,
        repository_count=repository_count, min_repository_benefit=min_repository_benefit,
    )


@dataclass(frozen=True)
class CandidatePoint:
    name: str
    repo_metric: float
    generic_regression: float
    execution_regression: float
    kl_drift: float

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("candidate name must be non-empty")
        for field in ("repo_metric", "generic_regression", "execution_regression", "kl_drift"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.kl_drift < 0:
            raise ValueError("kl_drift must be >= 0")


def pareto_frontier(points: Iterable[CandidatePoint]) -> list[CandidatePoint]:
    """Return nondominated points: higher repo metric, lower regression/KL."""
    pts = list(points)
    for p in pts:
        p.validate()
    out: list[CandidatePoint] = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            no_worse = (q.repo_metric >= p.repo_metric
                        and q.generic_regression <= p.generic_regression
                        and q.execution_regression <= p.execution_regression
                        and q.kl_drift <= p.kl_drift)
            strictly_better = (q.repo_metric > p.repo_metric
                               or q.generic_regression < p.generic_regression
                               or q.execution_regression < p.execution_regression
                               or q.kl_drift < p.kl_drift)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda x: (-x.repo_metric, x.generic_regression, x.execution_regression, x.kl_drift, x.name))


@dataclass(frozen=True)
class CorrectionCandidate:
    kind: str
    value: float


def build_correction_grid(cfg) -> list[CorrectionCandidate]:
    return ([CorrectionCandidate("scale", float(v)) for v in cfg.scale_grid]
            + [CorrectionCandidate("hard_fisher", float(v)) for v in cfg.hard_fisher_grid]
            + [CorrectionCandidate("soft_fisher", float(v)) for v in cfg.soft_fisher_grid])
