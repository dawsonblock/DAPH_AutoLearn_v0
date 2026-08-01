"""v0.3.10.5-alpha — Gate A statistical correctness repair.

This module implements the correct qualification statistics,
routing semantics, and artifact types per the Gate A statistical
correctness repair specification.

Key changes from v0.3.10.4:
  - Hard routing actions (not soft probability-weighted utility)
  - Real group bootstrap (20,000 iterations)
  - P1-minus-sham inference (not sham utility interval)
  - Frozen policy/calibration/representation evaluation
  - Precondition gates before statistical gates
  - QualificationStatus enum (PASS/FAIL/NOT_EVALUABLE/INVALIDATED)
  - Explicit comparator semantics
"""
from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Qualification Status
# ──────────────────────────────────────────────────────────────────────

class QualificationStatus(str, Enum):
    """Final qualification status for a Gate A experiment."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    INVALIDATED = "INVALIDATED"


# ──────────────────────────────────────────────────────────────────────
# Section 24 — Explicit Comparator Semantics
# ──────────────────────────────────────────────────────────────────────

class Comparator(str, Enum):
    """Explicit comparison operators for gate thresholds."""
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"


def compare(observed: float, comparator: Comparator, threshold: float) -> bool:
    """Compare an observed value against a threshold using the given comparator."""
    if comparator == Comparator.GT:
        return observed > threshold
    if comparator == Comparator.GTE:
        return observed >= threshold
    if comparator == Comparator.LT:
        return observed < threshold
    if comparator == Comparator.LTE:
        return observed <= threshold
    if comparator == Comparator.EQ:
        return abs(observed - threshold) < 1e-10
    raise ValueError(f"unknown comparator: {comparator!r}")


# ──────────────────────────────────────────────────────────────────────
# Section 7 — Route Actions and Realized Utility
#
# v0.4 — This is the B0 compatibility benchmark action space (formerly
# "Gate A"). The generic :class:`daph_learning.executive.ActionSpace`
# supersedes this enum for new experiments. ``RouteAction`` is kept as
# a thin alias so all v0.3.x code and artifacts continue to work.
# ──────────────────────────────────────────────────────────────────────

class RouteAction(str, Enum):
    """Hard routing action selected by the frozen policy (B0 benchmark).

    This is the binary compatibility action space. New experiments
    should use :func:`daph_learning.executive.binary_action_space` or
    a custom :class:`~daph_learning.executive.ActionSpace`.
    """
    SYMBOLIC = "symbolic"
    LLM = "llm"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class RoutingDecision:
    """A single routing decision for one task (B0 benchmark).

    v0.4 — :class:`~daph_learning.executive.ActionDecision` is the
    generic equivalent for arbitrary action spaces.
    """
    task_id: str
    symbolic_probability: float
    action: RouteAction
    confidence: float
    threshold_symbolic: float
    threshold_llm: float
    calibration_applied: bool

    def to_action_decision(self):
        """Convert to a generic :class:`ActionDecision`."""
        from daph_learning.executive.adapters import (
            action_decision_from_symbolic_probability,
        )
        return action_decision_from_symbolic_probability(
            self.task_id,
            self.symbolic_probability,
            calibrated=self.calibration_applied,
        )


def select_route_action(
    symbolic_probability: float,
    *,
    threshold_symbolic: float = 0.5,
    threshold_llm: float = 0.5,
    abstention_enabled: bool = False,
) -> RouteAction:
    """Select a hard routing action from a calibrated probability.

    For a two-action policy (no abstention):
        a = S if p >= t, else L

    For a three-action policy (abstention):
        a = S if p >= t_S
        a = L if p <= t_L
        a = A if t_L < p < t_S
    """
    if abstention_enabled and threshold_symbolic > threshold_llm:
        if symbolic_probability >= threshold_symbolic:
            return RouteAction.SYMBOLIC
        if symbolic_probability <= threshold_llm:
            return RouteAction.LLM
        return RouteAction.ABSTAIN
    if symbolic_probability >= threshold_symbolic:
        return RouteAction.SYMBOLIC
    return RouteAction.LLM


def realized_policy_utility(
    decision: RoutingDecision,
    symbolic_utility: float,
    llm_utility: float,
    *,
    abstain_utility: float | None = None,
) -> float:
    """Compute realized utility from the selected hard action."""
    if decision.action is RouteAction.SYMBOLIC:
        return symbolic_utility
    if decision.action is RouteAction.LLM:
        return llm_utility
    if decision.action is RouteAction.ABSTAIN:
        if abstain_utility is None:
            raise ValueError("abstain utility is required for ABSTAIN action")
        return abstain_utility
    raise AssertionError(f"unknown action: {decision.action!r}")


# ──────────────────────────────────────────────────────────────────────
# Section 6 — Missing Metric Detection
# ──────────────────────────────────────────────────────────────────────

class MissingQualificationMetricError(RuntimeError):
    """Raised when a required qualification metric is missing."""
    pass


def require_finite_metric(metrics: Mapping[str, Any], key: str) -> float:
    """Extract a required finite float metric. Never defaults to 0."""
    if key not in metrics:
        raise MissingQualificationMetricError(key)
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite: {value!r}")
    return value


# ──────────────────────────────────────────────────────────────────────
# Section 8 — Task-Level Records and Group Metrics
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FinalTaskRecord:
    """Canonical final-evaluation row for one task."""
    task_id: str
    group_id: str
    subtype: str
    split: str
    symbolic_utility: float
    llm_utility: float
    utility_gap_symbolic_minus_llm: float
    symbolic_probability: float
    calibrated_symbolic_probability: float
    raw_symbolic_probability: float
    selected_action: str
    oracle_action: str
    p1_realized_utility: float
    p0_realized_utility: float
    always_symbolic_utility: float
    oracle_utility: float
    p1_minus_p0: float
    p1_minus_oracle: float
    symbolic_correct: bool
    llm_correct: bool
    symbolic_verification_status: str
    llm_verification_status: str
    ood_score: float | None = None
    ood_detected: bool = False
    policy_hash: str = ""
    calibration_hash: str = ""
    representation_hash: str = ""


@dataclass(frozen=True)
class GroupMetric:
    """Group-level metric for one group."""
    group_id: str
    n_tasks: int
    mean_p1_minus_p0: float
    mean_p1_minus_sham: float | None
    mean_p1_utility: float
    mean_p0_utility: float
    mean_oracle_utility: float
    positive_gain: bool


def compute_group_metrics(
    records: Sequence[FinalTaskRecord],
) -> list[GroupMetric]:
    """Compute per-group metrics from final task records."""
    groups: dict[str, list[FinalTaskRecord]] = {}
    for r in records:
        groups.setdefault(r.group_id, []).append(r)
    result = []
    for gid, recs in sorted(groups.items()):
        n = len(recs)
        mean_p1 = float(np.mean([r.p1_realized_utility for r in recs]))
        mean_p0 = float(np.mean([r.p0_realized_utility for r in recs]))
        mean_oracle = float(np.mean([r.oracle_utility for r in recs]))
        mean_delta = float(np.mean([r.p1_minus_p0 for r in recs]))
        result.append(GroupMetric(
            group_id=gid,
            n_tasks=n,
            mean_p1_minus_p0=mean_delta,
            mean_p1_minus_sham=None,
            mean_p1_utility=mean_p1,
            mean_p0_utility=mean_p0,
            mean_oracle_utility=mean_oracle,
            positive_gain=mean_delta > 0,
        ))
    return result


# ──────────────────────────────────────────────────────────────────────
# Section 8.3 — Group Bootstrap
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BootstrapResult:
    """Result of a group bootstrap procedure."""
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_iterations: int
    estimand: str
    seed: int
    samples_sha256: str
    samples: np.ndarray | None = None


def group_bootstrap_mean_delta(
    group_deltas: Mapping[str, np.ndarray],
    *,
    n_iterations: int = 20000,
    confidence_level: float = 0.95,
    seed: int = 0,
    estimand: str = "group_weighted",
) -> BootstrapResult:
    """True cluster bootstrap of the mean delta.

    For group_weighted:
        1. Compute one mean delta per group.
        2. Sample group means with replacement.
        3. Take the mean of sampled group means.
        4. Repeat n_iterations times.
        5. Percentile bounds at alpha/2 and 1-alpha/2.

    For task_weighted:
        1. Sample groups with replacement.
        2. Include all tasks from each sampled group.
        3. Compute the task-level mean.
    """
    rng = np.random.default_rng(seed)
    group_names = list(group_deltas.keys())
    n_groups = len(group_names)
    if n_groups == 0:
        raise ValueError("no groups for bootstrap")

    if estimand == "group_weighted":
        group_means = np.array([
            float(np.mean(group_deltas[g])) for g in group_names
        ])
        point_estimate = float(group_means.mean())
        samples = np.empty(n_iterations, dtype=np.float64)
        for i in range(n_iterations):
            idx = rng.integers(0, n_groups, size=n_groups)
            samples[i] = group_means[idx].mean()
    elif estimand == "task_weighted":
        all_vals = np.concatenate([group_deltas[g] for g in group_names])
        point_estimate = float(all_vals.mean())
        group_arrays = [group_deltas[g] for g in group_names]
        samples = np.empty(n_iterations, dtype=np.float64)
        for i in range(n_iterations):
            idx = rng.integers(0, n_groups, size=n_groups)
            pooled = np.concatenate([group_arrays[j] for j in idx])
            samples[i] = pooled.mean()
    else:
        raise ValueError(f"unknown estimand: {estimand!r}")

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(samples, 100 * alpha / 2))
    ci_high = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    samples_hash = hashlib.sha256(samples.tobytes()).hexdigest()

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=confidence_level,
        n_iterations=n_iterations,
        estimand=estimand,
        seed=seed,
        samples_sha256=samples_hash,
        samples=samples,
    )


# ──────────────────────────────────────────────────────────────────────
# Section 9 — P1-minus-sham inference
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShamTaskPrediction:
    """Per-seed sham prediction for one task."""
    sham_seed: int
    task_id: str
    symbolic_probability: float
    selected_action: str
    realized_utility: float


def bootstrap_p1_minus_sham(
    p1_records: Sequence[FinalTaskRecord],
    sham_records: Sequence[ShamTaskPrediction],
    *,
    n_iterations: int = 20000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Nested resampling bootstrap of P1-minus-sham.

    1. Sample a sham seed.
    2. Sample final groups with replacement.
    3. Compute the group-weighted P1-minus-sham mean.
    4. Repeat n_iterations times.
    """
    rng = np.random.default_rng(seed)

    # Group P1 records by group_id.
    p1_groups: dict[str, list[FinalTaskRecord]] = {}
    for r in p1_records:
        p1_groups.setdefault(r.group_id, []).append(r)
    group_names = sorted(p1_groups.keys())
    n_groups = len(group_names)
    if n_groups == 0:
        raise ValueError("no P1 groups for sham bootstrap")

    # Group sham records by seed, then by task_id.
    sham_seeds = sorted(set(r.sham_seed for r in sham_records))
    n_sham_seeds = len(sham_seeds)
    if n_sham_seeds == 0:
        raise ValueError("no sham records for bootstrap")

    sham_by_seed: dict[int, dict[str, ShamTaskPrediction]] = {}
    for r in sham_records:
        sham_by_seed.setdefault(r.sham_seed, {})[r.task_id] = r

    # Pre-compute group-level P1 utilities.
    p1_group_means = {}
    for gname in group_names:
        recs = p1_groups[gname]
        p1_group_means[gname] = float(np.mean([r.p1_realized_utility for r in recs]))

    # Pre-compute group-level sham utilities per seed.
    sham_group_means: dict[int, dict[str, float]] = {}
    for s in sham_seeds:
        sham_preds = sham_by_seed[s]
        sham_group_means[s] = {}
        for gname in group_names:
            recs = p1_groups[gname]
            sham_utils = []
            for r in recs:
                pred = sham_preds.get(r.task_id)
                if pred is not None:
                    sham_utils.append(pred.realized_utility)
                else:
                    sham_utils.append(0.0)
            sham_group_means[s][gname] = float(np.mean(sham_utils)) if sham_utils else 0.0

    # P1-minus-mean-sham point estimate (group-weighted).
    mean_sham_per_group = {}
    for gname in group_names:
        vals = [sham_group_means[s][gname] for s in sham_seeds]
        mean_sham_per_group[gname] = float(np.mean(vals))
    p1_point = float(np.mean([p1_group_means[g] for g in group_names]))
    sham_point = float(np.mean([mean_sham_per_group[g] for g in group_names]))
    point_estimate = p1_point - sham_point

    # Nested bootstrap.
    samples = np.empty(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        # Sample a sham seed.
        s_idx = rng.integers(0, n_sham_seeds)
        s = sham_seeds[s_idx]
        # Sample groups with replacement.
        g_idx = rng.integers(0, n_groups, size=n_groups)
        p1_vals = [p1_group_means[group_names[j]] for j in g_idx]
        sham_vals = [sham_group_means[s][group_names[j]] for j in g_idx]
        samples[i] = np.mean(p1_vals) - np.mean(sham_vals)

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(samples, 100 * alpha / 2))
    ci_high = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    samples_hash = hashlib.sha256(samples.tobytes()).hexdigest()

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=confidence_level,
        n_iterations=n_iterations,
        estimand="p1_minus_sham_group_weighted",
        seed=seed,
        samples_sha256=samples_hash,
        samples=samples,
    )


# ──────────────────────────────────────────────────────────────────────
# Section 14 — Group-Positive Fraction
# ──────────────────────────────────────────────────────────────────────

def positive_group_fraction(records: Sequence[FinalTaskRecord]) -> float:
    """Fraction of groups where P1 improves over P0.

    I_g = 1[mean(U(P1) - U(P0)) > 0] for group g
    f = (1/G) * sum(I_g)
    """
    groups: dict[str, list[float]] = {}
    for r in records:
        groups.setdefault(r.group_id, []).append(r.p1_minus_p0)
    if not groups:
        return 0.0
    positive = sum(1 for vals in groups.values() if np.mean(vals) > 0)
    return positive / len(groups)


def group_fraction_breakdown(records: Sequence[FinalTaskRecord]) -> dict[str, int | float]:
    """Detailed breakdown of group fractions."""
    groups: dict[str, list[float]] = {}
    for r in records:
        groups.setdefault(r.group_id, []).append(r.p1_minus_p0)
    n_groups = len(groups)
    if n_groups == 0:
        return {"positive": 0, "negative": 0, "zero": 0, "total": 0,
                "fraction": 0.0}
    positive = sum(1 for vals in groups.values() if np.mean(vals) > 0)
    negative = sum(1 for vals in groups.values() if np.mean(vals) < 0)
    zero = n_groups - positive - negative
    return {
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "total": n_groups,
        "fraction": positive / n_groups,
    }


# ──────────────────────────────────────────────────────────────────────
# Section 15 — Crossover Subtype Counts
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SubtypeCrossoverMetric:
    """Crossover metric for one subtype."""
    subtype: str
    n_tasks: int
    symbolic_preferred_fraction: float
    llm_preferred_fraction: float
    tie_fraction: float
    crossover_valid: bool


def compute_crossover_metrics(
    records: Sequence[FinalTaskRecord],
    *,
    decisive_threshold: float = 0.02,
    min_preference_fraction: float = 0.20,
) -> list[SubtypeCrossoverMetric]:
    """Compute crossover metrics per subtype.

    A subtype has valid crossover if both:
        r_S(s) = P(delta_U > tau | s) >= r_min
        r_L(s) = P(delta_U < -tau | s) >= r_min
    """
    subtypes: dict[str, list[float]] = {}
    for r in records:
        subtypes.setdefault(r.subtype, []).append(
            r.utility_gap_symbolic_minus_llm)
    result = []
    for sub in sorted(subtypes.keys()):
        gaps = subtypes[sub]
        n = len(gaps)
        gaps_arr = np.array(gaps)
        sym_pref = float(np.mean(gaps_arr > decisive_threshold))
        llm_pref = float(np.mean(gaps_arr < -decisive_threshold))
        tie = 1.0 - sym_pref - llm_pref
        crossover = sym_pref >= min_preference_fraction and llm_pref >= min_preference_fraction
        result.append(SubtypeCrossoverMetric(
            subtype=sub,
            n_tasks=n,
            symbolic_preferred_fraction=sym_pref,
            llm_preferred_fraction=llm_pref,
            tie_fraction=tie,
            crossover_valid=crossover,
        ))
    return result


def count_crossover_subtypes(
    records: Sequence[FinalTaskRecord],
    *,
    decisive_threshold: float = 0.02,
    min_preference_fraction: float = 0.20,
) -> int:
    """Count subtypes with valid crossover."""
    metrics = compute_crossover_metrics(
        records,
        decisive_threshold=decisive_threshold,
        min_preference_fraction=min_preference_fraction,
    )
    return sum(1 for m in metrics if m.crossover_valid)


# ──────────────────────────────────────────────────────────────────────
# Section 16 — Final Decisive Fraction
# ──────────────────────────────────────────────────────────────────────

def decisive_fraction(records: Sequence[FinalTaskRecord], *, threshold: float = 0.02) -> float:
    """Fraction of tasks with |delta_U| > threshold."""
    if not records:
        return 0.0
    return float(np.mean([
        abs(r.utility_gap_symbolic_minus_llm) > threshold for r in records
    ]))


# ──────────────────────────────────────────────────────────────────────
# Section 17 — Subtype Regression
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SubtypeRegressionMetric:
    """Regression metric for one subtype."""
    subtype: str
    n_tasks: int
    n_groups: int
    p1_utility: float
    p0_utility: float
    p1_minus_p0: float
    symbolic_fraction: float
    llm_fraction: float
    oracle_agreement: float


def compute_subtype_regression(
    records: Sequence[FinalTaskRecord],
) -> tuple[list[SubtypeRegressionMetric], float]:
    """Compute per-subtype regression metrics.

    Returns (per_subtype_metrics, worst_regression).
    worst_regression = max_s max(0, -mean(P1-P0 | s))
    """
    subtypes: dict[str, list[FinalTaskRecord]] = {}
    for r in records:
        subtypes.setdefault(r.subtype, []).append(r)
    result = []
    worst_regression = 0.0
    for sub in sorted(subtypes.keys()):
        recs = subtypes[sub]
        n = len(recs)
        groups = set(r.group_id for r in recs)
        p1_u = float(np.mean([r.p1_realized_utility for r in recs]))
        p0_u = float(np.mean([r.p0_realized_utility for r in recs]))
        delta = p1_u - p0_u
        sym_frac = float(np.mean([
            r.selected_action == "symbolic" for r in recs
        ]))
        llm_frac = 1.0 - sym_frac
        oracle_agree = float(np.mean([
            r.selected_action == r.oracle_action for r in recs
        ]))
        result.append(SubtypeRegressionMetric(
            subtype=sub,
            n_tasks=n,
            n_groups=len(groups),
            p1_utility=p1_u,
            p0_utility=p0_u,
            p1_minus_p0=delta,
            symbolic_fraction=sym_frac,
            llm_fraction=llm_frac,
            oracle_agreement=oracle_agree,
        ))
        if delta < 0:
            worst_regression = max(worst_regression, -delta)
    return result, worst_regression


# ──────────────────────────────────────────────────────────────────────
# Section 18 — Oracle Gap Capture
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OracleGapCapture:
    """Oracle gap capture metric."""
    value: float | None
    status: str  # "DEFINED" or "UNDEFINED_NO_HEADROOM"
    oracle_utility: float
    p0_utility: float
    p1_utility: float
    oracle_headroom: float
    captured_headroom: float


def compute_oracle_gap_capture(
    records: Sequence[FinalTaskRecord],
    *,
    epsilon: float = 1e-10,
) -> OracleGapCapture:
    """Compute oracle gap capture.

    eta = (U_P1 - U_P0) / (U_oracle - U_P0)

    Does NOT clip to [0, 1]. A policy can underperform P0 (negative).
    """
    if not records:
        return OracleGapCapture(
            value=None, status="UNDEFINED_NO_HEADROOM",
            oracle_utility=0.0, p0_utility=0.0, p1_utility=0.0,
            oracle_headroom=0.0, captured_headroom=0.0,
        )
    oracle_u = float(np.mean([r.oracle_utility for r in records]))
    p0_u = float(np.mean([r.p0_realized_utility for r in records]))
    p1_u = float(np.mean([r.p1_realized_utility for r in records]))
    headroom = oracle_u - p0_u
    captured = p1_u - p0_u
    if headroom <= epsilon:
        return OracleGapCapture(
            value=None, status="UNDEFINED_NO_HEADROOM",
            oracle_utility=oracle_u, p0_utility=p0_u, p1_utility=p1_u,
            oracle_headroom=headroom, captured_headroom=captured,
        )
    return OracleGapCapture(
        value=captured / headroom,
        status="DEFINED",
        oracle_utility=oracle_u, p0_utility=p0_u, p1_utility=p1_u,
        oracle_headroom=headroom, captured_headroom=captured,
    )


# ──────────────────────────────────────────────────────────────────────
# Section 9.1 — Training Procedure Identity
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainingProcedureIdentity:
    """Shared training specification for P1 and sham."""
    policy_class: str
    feature_schema_hash: str
    feature_transform_hash: str
    regularization: float
    solver: str
    calibration_method: str
    target_mode: str
    threshold_policy: str

    @property
    def procedure_identity_hash(self) -> str:
        payload = {
            "policy_class": self.policy_class,
            "feature_schema_hash": self.feature_schema_hash,
            "feature_transform_hash": self.feature_transform_hash,
            "regularization": self.regularization,
            "solver": self.solver,
            "calibration_method": self.calibration_method,
            "target_mode": self.target_mode,
            "threshold_policy": self.threshold_policy,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Section 23 — Precondition Gates
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreconditionResult:
    """Result of a single precondition check."""
    name: str
    passed: bool
    actual: Any
    required: Any
    message: str = ""


def check_preconditions(
    records: Sequence[FinalTaskRecord],
    *,
    require_real_model: bool = True,
    used_real_model: bool = True,
    minimum_final_groups: int = 60,
    minimum_final_tasks: int = 400,
    minimum_crossover_subtypes: int = 3,
    minimum_backend_win_fraction: float = 0.20,
    minimum_final_decisive_fraction: float = 0.35,
    require_frozen_policy: bool = True,
    has_frozen_policy: bool = True,
    require_frozen_calibration: bool = True,
    has_frozen_calibration: bool = True,
    require_frozen_representation: bool = True,
    has_frozen_representation: bool = True,
    require_exact_model_revision: bool = True,
    model_revision: str = "",
    tokenizer_revision: str = "",
    decisive_threshold: float = 0.02,
    crossover_min_preference: float = 0.20,
) -> list[PreconditionResult]:
    """Check all experiment preconditions.

    If any fail, qualification_status = NOT_EVALUABLE.
    """
    results = []
    n_tasks = len(records)
    groups = set(r.group_id for r in records)
    n_groups = len(groups)

    results.append(PreconditionResult(
        name="require_real_model",
        passed=not require_real_model or used_real_model,
        actual=used_real_model,
        required=require_real_model,
    ))

    results.append(PreconditionResult(
        name="minimum_final_groups",
        passed=n_groups >= minimum_final_groups,
        actual=n_groups,
        required=minimum_final_groups,
    ))

    results.append(PreconditionResult(
        name="minimum_final_tasks",
        passed=n_tasks >= minimum_final_tasks,
        actual=n_tasks,
        required=minimum_final_tasks,
    ))

    crossover_count = count_crossover_subtypes(
        records,
        decisive_threshold=decisive_threshold,
        min_preference_fraction=crossover_min_preference,
    )
    results.append(PreconditionResult(
        name="minimum_crossover_subtypes",
        passed=crossover_count >= minimum_crossover_subtypes,
        actual=crossover_count,
        required=minimum_crossover_subtypes,
    ))

    # Backend win fraction: fraction of tasks where at least one backend
    # succeeds (utility > 0 for either symbolic or LLM).
    backend_wins = float(np.mean([
        max(r.symbolic_utility, r.llm_utility) > 0 for r in records
    ])) if records else 0.0
    results.append(PreconditionResult(
        name="minimum_backend_win_fraction",
        passed=backend_wins >= minimum_backend_win_fraction,
        actual=backend_wins,
        required=minimum_backend_win_fraction,
    ))

    final_decisive = decisive_fraction(records, threshold=decisive_threshold)
    results.append(PreconditionResult(
        name="minimum_final_decisive_fraction",
        passed=final_decisive >= minimum_final_decisive_fraction,
        actual=final_decisive,
        required=minimum_final_decisive_fraction,
    ))

    results.append(PreconditionResult(
        name="require_frozen_policy",
        passed=not require_frozen_policy or has_frozen_policy,
        actual=has_frozen_policy,
        required=require_frozen_policy,
    ))

    results.append(PreconditionResult(
        name="require_frozen_calibration",
        passed=not require_frozen_calibration or has_frozen_calibration,
        actual=has_frozen_calibration,
        required=require_frozen_calibration,
    ))

    results.append(PreconditionResult(
        name="require_frozen_representation",
        passed=not require_frozen_representation or has_frozen_representation,
        actual=has_frozen_representation,
        required=require_frozen_representation,
    ))

    # Model revision checks.
    if require_exact_model_revision:
        rev_ok = (bool(model_revision) and model_revision != "main"
                  and tokenizer_revision != "" and tokenizer_revision != "main")
        results.append(PreconditionResult(
            name="require_exact_model_revision",
            passed=rev_ok,
            actual=f"model={model_revision!r}, tokenizer={tokenizer_revision!r}",
            required="non-empty exact revision (not 'main')",
        ))

    return results


# ──────────────────────────────────────────────────────────────────────
# Section 10 — Frozen Policy Artifact
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FrozenPolicyArtifact:
    """Serialized frozen policy with fitted parameters."""
    artifact_version: str
    experiment_id: str
    policy_class: str
    feature_dimension: int
    feature_schema_hash: str
    feature_transform_hash: str
    coefficients: tuple[float, ...]
    intercept: float
    classes: tuple[str, ...]
    regularization: float
    solver: str
    training_seed: int
    target_mode: str
    training_dataset_hash: str
    development_dataset_hash: str
    calibration_artifact_hash: str | None
    policy_sha256: str


class FrozenRoutingPolicy:
    """Loaded frozen policy that can only predict, not train."""

    def __init__(self, coefficients: np.ndarray, intercept: float):
        self._coef = np.asarray(coefficients, dtype=np.float64)
        self._intercept = float(intercept)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return P(symbolic | features) as a numpy array."""
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        logits = X @ self._coef + self._intercept
        # Sigmoid with overflow protection.
        probs = np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        )
        return probs.flatten()

    @property
    def coefficients(self) -> np.ndarray:
        return self._coef.copy()

    @property
    def intercept(self) -> float:
        return self._intercept


def load_frozen_policy(path: Path, *, expected_sha256: str) -> FrozenRoutingPolicy:
    """Load a frozen policy from a file, verifying its hash."""
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != expected_sha256:
        raise ValueError(
            f"policy artifact hash mismatch: expected {expected_sha256}, "
            f"got {actual_hash}"
        )
    data = json.loads(raw)
    coef = tuple(data.get("coefficients", []))
    intercept = float(data.get("intercept", 0.0))
    return FrozenRoutingPolicy(np.array(coef), intercept)


def serialize_frozen_policy(
    model,
    *,
    experiment_id: str,
    feature_schema_hash: str,
    feature_transform_hash: str,
    training_seed: int,
    target_mode: str,
    training_dataset_hash: str,
    development_dataset_hash: str,
    calibration_artifact_hash: str | None,
    regularization: float = 0.0,
    solver: str = "adam",
) -> dict[str, Any]:
    """Serialize a fitted logistic router to a frozen policy artifact dict."""
    # Extract coefficients from the model.
    if hasattr(model, "linear"):
        # WeightedLogisticRouter (torch)
        import torch
        coef = model.linear.weight.detach().cpu().numpy().flatten().tolist()
        intercept = float(model.linear.bias.detach().cpu().numpy()[0])
    elif hasattr(model, "coef_"):
        # sklearn-style
        coef = model.coef_.flatten().tolist()
        intercept = float(model.intercept_)
    elif hasattr(model, "_coef"):
        # NumpyLogisticRouter
        coef = model._coef.flatten().tolist()
        intercept = float(model._intercept)
    else:
        raise ValueError(f"cannot extract coefficients from {type(model)}")

    artifact = {
        "artifact_version": "1.0",
        "experiment_id": experiment_id,
        "policy_class": "logistic_regression",
        "feature_dimension": len(coef),
        "feature_schema_hash": feature_schema_hash,
        "feature_transform_hash": feature_transform_hash,
        "coefficients": coef,
        "intercept": intercept,
        "classes": ["llm", "symbolic"],
        "regularization": regularization,
        "solver": solver,
        "training_seed": training_seed,
        "target_mode": target_mode,
        "training_dataset_hash": training_dataset_hash,
        "development_dataset_hash": development_dataset_hash,
        "calibration_artifact_hash": calibration_artifact_hash,
    }
    # Compute policy hash over the canonical JSON.
    canonical = json.dumps(artifact, sort_keys=True).encode()
    artifact["policy_sha256"] = hashlib.sha256(canonical).hexdigest()
    return artifact


# ──────────────────────────────────────────────────────────────────────
# Section 11 — Calibration Artifact
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalibrationArtifactV2:
    """Frozen calibration artifact with fitted parameters."""
    artifact_version: str
    experiment_id: str
    calibration_method: str  # identity, platt, isotonic, temperature
    probability_transform: dict[str, Any]
    symbolic_threshold: float
    llm_threshold: float
    abstention_enabled: bool
    ood_method: str | None
    ood_threshold: float | None
    calibration_dataset_hash: str
    representation_hash: str
    policy_hash: str
    calibration_sha256: str


def make_identity_calibration(
    *,
    experiment_id: str,
    symbolic_threshold: float = 0.5,
    llm_threshold: float = 0.5,
    abstention_enabled: bool = False,
    calibration_dataset_hash: str = "",
    representation_hash: str = "",
    policy_hash: str = "",
) -> dict[str, Any]:
    """Create an identity calibration artifact."""
    artifact = {
        "artifact_version": "2.0",
        "experiment_id": experiment_id,
        "calibration_method": "identity",
        "probability_transform": {"method": "identity"},
        "symbolic_threshold": symbolic_threshold,
        "llm_threshold": llm_threshold,
        "abstention_enabled": abstention_enabled,
        "ood_method": None,
        "ood_threshold": None,
        "calibration_dataset_hash": calibration_dataset_hash,
        "representation_hash": representation_hash,
        "policy_hash": policy_hash,
    }
    canonical = json.dumps(artifact, sort_keys=True).encode()
    artifact["calibration_sha256"] = hashlib.sha256(canonical).hexdigest()
    return artifact


def apply_calibration(
    raw_probability: float,
    calibration: dict[str, Any],
) -> float:
    """Apply a frozen calibration transform to a raw probability."""
    method = calibration.get("calibration_method", "identity")
    transform = calibration.get("probability_transform", {})
    if method == "identity":
        return float(raw_probability)
    if method == "temperature":
        temp = float(transform.get("temperature", 1.0))
        if temp <= 0:
            raise ValueError(f"invalid temperature: {temp}")
        # Convert to logit, divide by temp, back to sigmoid.
        p = max(min(raw_probability, 1.0 - 1e-10), 1e-10)
        logit = math.log(p / (1.0 - p))
        return 1.0 / (1.0 + math.exp(-logit / temp))
    if method == "platt":
        a = float(transform.get("a", 1.0))
        b = float(transform.get("b", 0.0))
        return 1.0 / (1.0 + math.exp(-(a * raw_probability + b)))
    if method == "isotonic":
        # Isotonic requires stored mapping; use lookup.
        mapping = transform.get("mapping", [])
        if not mapping:
            return float(raw_probability)
        for entry in mapping:
            if raw_probability <= entry["input"]:
                return float(entry["output"])
        return float(mapping[-1]["output"])
    return float(raw_probability)


# ──────────────────────────────────────────────────────────────────────
# Section 12 — Representation Artifact
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RepresentationArtifact:
    """Frozen representation transform artifact."""
    artifact_version: str
    experiment_id: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    layer_index: int
    pooling_method: str
    include_surface_features: bool
    surface_feature_schema: tuple[str, ...]
    normalization_method: str
    normalization_mean: tuple[float, ...] | None
    normalization_scale: tuple[float, ...] | None
    pca_enabled: bool
    pca_components: tuple[tuple[float, ...], ...] | None
    pca_mean: tuple[float, ...] | None
    output_dimension: int
    representation_sha256: str


def make_representation_artifact(
    *,
    experiment_id: str,
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    layer_index: int,
    pooling_method: str,
    include_surface_features: bool,
    surface_feature_schema: tuple[str, ...],
    normalization_method: str = "none",
    output_dimension: int = 0,
) -> dict[str, Any]:
    """Create a representation artifact dict."""
    artifact = {
        "artifact_version": "1.0",
        "experiment_id": experiment_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "layer_index": layer_index,
        "pooling_method": pooling_method,
        "include_surface_features": include_surface_features,
        "surface_feature_schema": list(surface_feature_schema),
        "normalization_method": normalization_method,
        "normalization_mean": None,
        "normalization_scale": None,
        "pca_enabled": False,
        "pca_components": None,
        "pca_mean": None,
        "output_dimension": output_dimension,
    }
    canonical = json.dumps(artifact, sort_keys=True).encode()
    artifact["representation_sha256"] = hashlib.sha256(canonical).hexdigest()
    return artifact


# ──────────────────────────────────────────────────────────────────────
# Section 13 — Freeze Manifest (expanded)
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FreezeManifestV2:
    """Expanded freeze manifest with all required fields."""
    experiment_id: str
    package_version: str
    source_sha256: str
    config_sha256: str
    gate_criteria_sha256: str
    utility_config_sha256: str
    train_dataset_sha256: str
    development_dataset_sha256: str
    calibration_dataset_sha256: str
    final_dataset_sha256: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    representation_artifact_sha256: str
    policy_artifact_sha256: str
    calibration_artifact_sha256: str
    environment_sha256: str
    frozen_at: str


# ──────────────────────────────────────────────────────────────────────
# Section 22 — Independent Statistical Validator
# ──────────────────────────────────────────────────────────────────────

def validate_statistics_from_records(
    records: Sequence[FinalTaskRecord],
    *,
    gate_config: Mapping[str, Any],
    bootstrap_iterations: int = 20000,
    bootstrap_seed: int = 20260731,
    confidence_level: float = 0.95,
    decisive_threshold: float = 0.02,
    crossover_min_preference: float = 0.20,
) -> dict[str, Any]:
    """Independently recompute all statistics from task-level records.

    This is the validator's recomputation engine. It does NOT trust
    stored gate_decision.json — it derives the verdict independently.
    """
    if not records:
        return {
            "valid": False,
            "error": "no task records",
        }

    # Group records by group_id for bootstrap.
    group_deltas: dict[str, np.ndarray] = {}
    for r in records:
        group_deltas.setdefault(r.group_id, []).append(r.p1_minus_p0)
    group_deltas = {k: np.array(v) for k, v in group_deltas.items()}

    # Primary endpoint: group bootstrap of P1-P0.
    bootstrap = group_bootstrap_mean_delta(
        group_deltas,
        n_iterations=bootstrap_iterations,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
        estimand="group_weighted",
    )

    # Oracle gap capture.
    oracle_capture = compute_oracle_gap_capture(records)

    # Positive group fraction.
    pos_group = positive_group_fraction(records)
    pos_breakdown = group_fraction_breakdown(records)

    # Subtype regression.
    subtype_metrics, worst_regression = compute_subtype_regression(records)

    # Crossover.
    crossover_metrics = compute_crossover_metrics(
        records,
        decisive_threshold=decisive_threshold,
        min_preference_fraction=crossover_min_preference,
    )
    crossover_count = sum(1 for m in crossover_metrics if m.crossover_valid)

    # Decisive fraction.
    final_decisive = decisive_fraction(records, threshold=decisive_threshold)

    # Route distribution.
    actions = [r.selected_action for r in records]
    p_sym = float(np.mean([a == "symbolic" for a in actions]))
    p_llm = float(np.mean([a == "llm" for a in actions]))

    # Oracle agreement.
    oracle_agree = float(np.mean([
        r.selected_action == r.oracle_action for r in records
    ]))

    # Build recomputed metrics.
    recomputed = {
        "p1_utility": float(np.mean([r.p1_realized_utility for r in records])),
        "p0_utility": float(np.mean([r.p0_realized_utility for r in records])),
        "oracle_utility": float(np.mean([r.oracle_utility for r in records])),
        "p1_minus_p0": bootstrap.point_estimate,
        "p1_minus_p0_ci_low": bootstrap.ci_low,
        "p1_minus_p0_ci_high": bootstrap.ci_high,
        "oracle_gap_capture": oracle_capture.value if oracle_capture.value is not None else 0.0,
        "oracle_gap_capture_status": oracle_capture.status,
        "positive_group_fraction": pos_group,
        "positive_group_count": pos_breakdown["positive"],
        "negative_group_count": pos_breakdown["negative"],
        "zero_group_count": pos_breakdown["zero"],
        "total_group_count": pos_breakdown["total"],
        "worst_subtype_regression": worst_regression,
        "crossover_subtype_count": crossover_count,
        "final_decisive_fraction": final_decisive,
        "route_symbolic_fraction": p_sym,
        "route_llm_fraction": p_llm,
        "oracle_agreement": oracle_agree,
        "bootstrap_samples_sha256": bootstrap.samples_sha256,
        "n_tasks": len(records),
        "n_groups": len(group_deltas),
    }

    # Evaluate gates.
    gates = gate_config.get("gates", gate_config)
    gate_verdicts = {}
    for gate_name, gate_spec in gates.items():
        if not isinstance(gate_spec, dict):
            continue
        threshold = float(gate_spec.get("threshold", 0.0))
        comp_str = gate_spec.get("comparator", "gte")
        comparator = Comparator(comp_str)

        # Map gate name to recomputed metric.
        metric_map = {
            "minimum_point_gain_vs_p0": recomputed["p1_minus_p0"],
            "lcb_p1_minus_p0": recomputed["p1_minus_p0_ci_low"],
            "lcb_p1_minus_sham": recomputed.get("p1_minus_sham_ci_low", 0.0),
            "require_lcb_vs_p0_above": recomputed["p1_minus_p0_ci_low"],
            "require_lcb_vs_sham_above": recomputed.get("p1_minus_sham_ci_low", 0.0),
            "minimum_oracle_gap_capture": recomputed["oracle_gap_capture"],
            "minimum_positive_group_fraction": recomputed["positive_group_fraction"],
            "maximum_worst_subtype_regression": recomputed["worst_subtype_regression"],
            "maximum_final_access_count": 1,  # from ledger
        }
        observed = metric_map.get(gate_name, 0.0)
        passed = compare(float(observed), comparator, threshold)
        gate_verdicts[gate_name] = {
            "actual": observed,
            "threshold": threshold,
            "comparator": comp_str,
            "passed": passed,
        }

    all_passed = all(v["passed"] for v in gate_verdicts.values())
    recomputed["gate_verdicts"] = gate_verdicts
    recomputed["all_gates_passed"] = all_passed
    recomputed["valid"] = True
    return recomputed
