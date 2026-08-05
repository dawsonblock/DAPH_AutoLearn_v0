"""v0.3.9 (V039-008/009) — validation promotion gate and vector lineage.

Item 4 of the recommended repair sequence:

* Evaluate the **candidate** and the **incumbent** on the **same** held-out
  examples.
* Promote only when **all** of the following hold:
    - **adequate coverage** — enough held-out tasks produced a real decision;
    - **paired improvement** — the candidate beats the incumbent on
      discordant pairs (exact paired binomial p-value at the configured
      confidence);
    - **bounded regressions** — the per-task regression rate does not exceed
      the configured ceiling;
    - **confidence** — the Clopper-Pearson lower bound on the
      "improved-given-discordant" probability clears the configured floor.
* Otherwise **roll back** to the incumbent.

Vector lineage (V039-009) records every candidate/incumbent pair and the
promotion decision so a prior vector can be restored for auditability.

The module reuses the existing exact paired-binomial and Clopper-Pearson
implementations in :mod:`daph_learning.evaluation.paired` so the statistics
stay consistent with the rest of the protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

from daph_learning.evaluation.paired import (
    clopper_pearson_interval,
    exact_paired_binomial,
)


PromoteAction = Literal["promote", "rollback"]


@dataclass(frozen=True)
class PromotionGateConfig:
    """Configuration for the promotion gate. Frozen for reproducibility.

    v0.3.9 additions (Section 18/19):
    - ``min_sample_count``: minimum number of decided held-out tasks.
    - ``capability_regression_thresholds``: per-capability max regression rate.
      A candidate fails if any protected capability exceeds its threshold.
    - ``protected_capabilities``: capabilities that cannot regress beyond their
      threshold even if global utility improves.
    """

    min_coverage: float = 0.8
    min_discordant_n: int = 5
    max_p_value: float = 0.05
    max_regression_rate: float = 0.2
    min_improved_probability_lower: float = 0.55
    confidence: float = 0.95
    min_utility_delta: float = 0.0
    min_sample_count: int = 0
    capability_regression_thresholds: Mapping[str, float] = field(default_factory=dict)
    protected_capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1]")
        if not 0.0 < self.max_p_value < 1.0:
            raise ValueError("max_p_value must be in (0, 1)")
        if not 0.0 <= self.max_regression_rate <= 1.0:
            raise ValueError("max_regression_rate must be in [0, 1]")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if self.min_sample_count < 0:
            raise ValueError("min_sample_count must be >= 0")


@dataclass(frozen=True)
class HeldOutTaskResult:
    """One held-out task evaluated under both candidate and incumbent.

    ``candidate_correct`` / ``incumbent_correct`` are the verified-correctness
    of the backend each policy *routed to* on this task. ``decided`` is True
    when both policies produced a real (non-abstain, non-failed) decision;
    only decided tasks count toward coverage and the paired comparison.

    v0.3.9 additions (Section 3/18/19):
    - ``candidate_route`` / ``incumbent_route``: the actual route each policy
      selected (not the oracle's preferred route).
    - ``capability_id`` / ``task_family``: for capability regression gates.
    - ``utility_delta``: ``U_candidate - U_incumbent``.
    """

    task_id: str
    candidate_correct: bool
    incumbent_correct: bool
    candidate_utility: float = 0.0
    incumbent_utility: float = 0.0
    decided: bool = True
    candidate_route: str = ""
    incumbent_route: str = ""
    capability_id: str | None = None
    task_family: str | None = None

    @property
    def utility_delta(self) -> float:
        return self.candidate_utility - self.incumbent_utility


@dataclass(frozen=True)
class PromotionDecision:
    """The gate's verdict on a candidate vs incumbent comparison."""

    action: PromoteAction
    reason: str
    improved: int
    regressed: int
    tied: int
    n_decided: int
    n_held_out: int
    coverage: float
    regression_rate: float
    p_value: float
    improved_probability: float
    improved_probability_ci: tuple[float, float]
    mean_utility_delta: float
    config: PromotionGateConfig

    @property
    def promoted(self) -> bool:
        return self.action == "promote"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["improved_probability_ci"] = list(self.improved_probability_ci)
        return d


def evaluate_promotion_gate(
    results: Sequence[HeldOutTaskResult],
    config: PromotionGateConfig | None = None,
) -> PromotionDecision:
    """Run the promotion gate on held-out paired results.

    Returns a :class:`PromotionDecision`. The decision is ``"promote"`` only
    when every gate criterion passes; otherwise ``"rollback"`` with the
    failing reason recorded.
    """
    cfg = config or PromotionGateConfig()
    n_held_out = len(results)
    decided = [r for r in results if r.decided]
    n_decided = len(decided)
    coverage = n_decided / n_held_out if n_held_out else 0.0

    improved = sum(1 for r in decided if r.candidate_correct and not r.incumbent_correct)
    regressed = sum(1 for r in decided if r.incumbent_correct and not r.candidate_correct)
    tied = n_decided - improved - regressed
    regression_rate = regressed / n_decided if n_decided else 0.0
    p_value = exact_paired_binomial(improved, regressed)
    improved_probability = improved / (improved + regressed) if (improved + regressed) else 0.5
    ci = clopper_pearson_interval(
        improved, improved + regressed, confidence=cfg.confidence
    )
    utility_deltas = [r.candidate_utility - r.incumbent_utility for r in decided]
    mean_utility_delta = sum(utility_deltas) / n_decided if n_decided else 0.0

    def _reject(reason: str) -> PromotionDecision:
        return PromotionDecision(
            action="rollback", reason=reason, improved=improved, regressed=regressed,
            tied=tied, n_decided=n_decided, n_held_out=n_held_out, coverage=coverage,
            regression_rate=regression_rate, p_value=p_value,
            improved_probability=improved_probability,
            improved_probability_ci=ci, mean_utility_delta=mean_utility_delta,
            config=cfg,
        )

    # 1. adequate coverage
    if coverage < cfg.min_coverage:
        return _reject(
            f"coverage {coverage:.3f} < min_coverage {cfg.min_coverage:.3f}"
        )
    # 2. enough discordant pairs to form a paired test
    if improved + regressed < cfg.min_discordant_n:
        return _reject(
            f"discordant_n {improved + regressed} < min_discordant_n "
            f"{cfg.min_discordant_n}"
        )
    # 3. paired improvement (more improved than regressed AND significant)
    if improved <= regressed:
        return _reject(
            f"no paired improvement: improved={improved} <= regressed={regressed}"
        )
    if p_value > cfg.max_p_value:
        return _reject(
            f"p_value {p_value:.4f} > max_p_value {cfg.max_p_value}"
        )
    # 4. bounded regressions
    if regression_rate > cfg.max_regression_rate:
        return _reject(
            f"regression_rate {regression_rate:.3f} > max_regression_rate "
            f"{cfg.max_regression_rate:.3f}"
        )
    # 5. confidence (Clopper-Pearson lower bound on improved-given-discordant)
    if ci[0] < cfg.min_improved_probability_lower:
        return _reject(
            f"improved_probability lower bound {ci[0]:.3f} < "
            f"min_improved_probability_lower {cfg.min_improved_probability_lower:.3f}"
        )
    # 6. utility delta floor
    if mean_utility_delta < cfg.min_utility_delta:
        return _reject(
            f"mean_utility_delta {mean_utility_delta:.4f} < "
            f"min_utility_delta {cfg.min_utility_delta}"
        )
    # 7. minimum sample count (Section 18)
    if n_decided < cfg.min_sample_count:
        return _reject(
            f"n_decided {n_decided} < min_sample_count {cfg.min_sample_count}"
        )
    # 8. capability regression gates (Section 19)
    cap_regressions = _check_capability_regressions(decided, cfg)
    if cap_regressions:
        cap_str = "; ".join(
            f"{cap}: {rate:.3f} > {thresh:.3f}"
            for cap, rate, thresh in cap_regressions
        )
        return _reject(f"capability regression: {cap_str}")

    return PromotionDecision(
        action="promote", reason="all gate criteria satisfied", improved=improved,
        regressed=regressed, tied=tied, n_decided=n_decided, n_held_out=n_held_out,
        coverage=coverage, regression_rate=regression_rate, p_value=p_value,
        improved_probability=improved_probability, improved_probability_ci=ci,
        mean_utility_delta=mean_utility_delta, config=cfg,
    )


def _check_capability_regressions(
    decided: Sequence[HeldOutTaskResult],
    cfg: PromotionGateConfig,
) -> list[tuple[str, float, float]]:
    """Check per-capability regression rates against configured thresholds.

    Returns a list of ``(capability_id, actual_rate, threshold)`` tuples for
    capabilities that exceeded their regression threshold. A capability is
    checked if it is in ``protected_capabilities`` or has an entry in
    ``capability_regression_thresholds``.
    """
    if not cfg.capability_regression_thresholds and not cfg.protected_capabilities:
        return []
    # Group decided results by capability_id.
    by_cap: dict[str, list[HeldOutTaskResult]] = {}
    for r in decided:
        cap = r.capability_id
        if cap is None:
            continue
        by_cap.setdefault(cap, []).append(r)
    violations: list[tuple[str, float, float]] = []
    for cap, group in by_cap.items():
        threshold = cfg.capability_regression_thresholds.get(cap)
        if threshold is None and cap in cfg.protected_capabilities:
            threshold = cfg.max_regression_rate
        if threshold is None:
            continue
        n = len(group)
        cap_reg = sum(1 for r in group if r.incumbent_correct and not r.candidate_correct)
        rate = cap_reg / n if n > 0 else 0.0
        if rate > threshold:
            violations.append((cap, rate, threshold))
    return violations


# --- Vector lineage (V039-009) ---

@dataclass(frozen=True)
class VectorLineageRecord:
    """Immutable lineage record for one candidate/incumbent promotion step.

    Supports rollback: the ``parent_vector_id`` chain lets any prior vector
    be restored. The ``promotion_decision`` is the full gate verdict.
    """

    vector_id: str
    parent_vector_id: str | None
    iteration: int
    training_dataset_sha256: str
    development_dataset_sha256: str
    optimizer_parameters: dict[str, Any]
    utility_config_hash: str
    layer: int
    token_location: str
    norm: float
    alpha: float
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    promotion_decision: dict[str, Any]
    lineage_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_lineage_hash(record_without_hash: dict[str, Any]) -> str:
    import hashlib
    import json
    payload = {k: v for k, v in record_without_hash.items() if k != "lineage_hash"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def build_lineage_record(
    *,
    vector_id: str,
    parent_vector_id: str | None,
    iteration: int,
    training_dataset_sha256: str,
    development_dataset_sha256: str,
    optimizer_parameters: dict[str, Any],
    utility_config_hash: str,
    layer: int,
    token_location: str,
    norm: float,
    alpha: float,
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
    promotion_decision: PromotionDecision,
) -> VectorLineageRecord:
    fields_dict = {
        "vector_id": vector_id,
        "parent_vector_id": parent_vector_id,
        "iteration": iteration,
        "training_dataset_sha256": training_dataset_sha256,
        "development_dataset_sha256": development_dataset_sha256,
        "optimizer_parameters": dict(optimizer_parameters),
        "utility_config_hash": utility_config_hash,
        "layer": layer,
        "token_location": token_location,
        "norm": norm,
        "alpha": alpha,
        "metrics_before": dict(metrics_before),
        "metrics_after": dict(metrics_after),
        "promotion_decision": promotion_decision.to_dict(),
    }
    return VectorLineageRecord(
        lineage_hash=compute_lineage_hash(fields_dict), **fields_dict
    )


@dataclass
class VectorLineage:
    """Append-only lineage chain supporting rollback to any ancestor.

    ``current`` is the incumbent vector id (the last promoted candidate, or
    the seed). ``rollback_to(target_id)`` returns the lineage record for
    ``target_id`` so the caller can restore that vector; the chain itself
    is never mutated, only the ``current`` pointer.
    """

    records: list[VectorLineageRecord] = field(default_factory=list)
    current: str | None = None

    def append(self, record: VectorLineageRecord, *, promote: bool) -> None:
        self.records.append(record)
        if promote:
            self.current = record.vector_id
        # on rollback, current stays at the incumbent (parent)

    def get(self, vector_id: str) -> VectorLineageRecord:
        for r in self.records:
            if r.vector_id == vector_id:
                return r
        raise KeyError(f"no lineage record for vector_id {vector_id!r}")

    def rollback_to(self, target_id: str) -> VectorLineageRecord:
        """Restore a prior vector: returns its lineage record and moves the
        current pointer back to it. The rolled-back candidate's record
        remains in the chain for auditability.

        Raises ``ValueError`` if ``target_id`` is not an ancestor of the
        current incumbent (or the current incumbent itself)."""
        record = self.get(target_id)
        # Verify target_id is an ancestor of (or equal to) the current incumbent
        if self.current is not None:
            by_id = {r.vector_id: r for r in self.records}
            cur = self.current
            found = False
            seen: set[str] = set()
            while cur is not None and cur in by_id and cur not in seen:
                if cur == target_id:
                    found = True
                    break
                seen.add(cur)
                cur = by_id[cur].parent_vector_id
            if not found:
                raise ValueError(
                    f"cannot rollback to {target_id!r}: it is not an ancestor "
                    f"of the current incumbent {self.current!r}"
                )
        self.current = target_id
        return record

    def chain(self) -> list[VectorLineageRecord]:
        """Return the records from the seed to the current incumbent."""
        if self.current is None:
            return []
        # walk parent pointers from current back to the seed
        by_id = {r.vector_id: r for r in self.records}
        ordered: list[VectorLineageRecord] = []
        cur = self.current
        seen: set[str] = set()
        while cur is not None and cur in by_id and cur not in seen:
            rec = by_id[cur]
            ordered.append(rec)
            seen.add(cur)
            cur = rec.parent_vector_id
        ordered.reverse()
        return ordered


__all__ = [
    "HeldOutTaskResult",
    "PromoteAction",
    "PromotionDecision",
    "PromotionGateConfig",
    "VectorLineage",
    "VectorLineageRecord",
    "build_lineage_record",
    "compute_lineage_hash",
    "evaluate_promotion_gate",
]
