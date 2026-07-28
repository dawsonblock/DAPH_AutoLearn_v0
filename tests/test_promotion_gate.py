"""V039-008/009 — promotion gate and vector rollback lineage.

Acceptance tests for Item 4:
* promote only with adequate coverage, paired improvement, bounded
  regressions, and confidence
* candidate regression causes rollback
* vector lineage supports rollback to a prior incumbent
"""

from __future__ import annotations

import pytest

from daph_learning.autolearn.promotion import (
    HeldOutTaskResult,
    PromotionGateConfig,
    VectorLineage,
    build_lineage_record,
    evaluate_promotion_gate,
)


def _res(tid, c, i, decided=True, cu=0.0, iu=0.0):
    return HeldOutTaskResult(task_id=tid, candidate_correct=c,
                             incumbent_correct=i, candidate_utility=cu,
                             incumbent_utility=iu, decided=decided)


# --- promote when all criteria satisfied ---

def test_promote_on_clear_paired_improvement():
    # 10 improved, 2 regressed, 8 tied -> p~0.039, regression_rate 0.167,
    # improved_probability 0.833 (CP95 lower ~0.519).
    results = (
        [_res(f"imp{i}", True, False, cu=1.0, iu=0.0) for i in range(10)]
        + [_res(f"reg{i}", False, True, cu=0.0, iu=1.0) for i in range(2)]
        + [_res(f"tie{i}", True, True, cu=1.0, iu=1.0) for i in range(8)]
    )
    cfg = PromotionGateConfig(min_coverage=0.9, min_discordant_n=5,
                              max_p_value=0.05, max_regression_rate=0.2,
                              min_improved_probability_lower=0.5)
    decision = evaluate_promotion_gate(results, cfg)
    assert decision.promoted is True
    assert decision.improved == 10
    assert decision.regressed == 2
    assert decision.coverage == 1.0


# --- rollback paths ---

def test_rollback_on_low_coverage():
    results = [_res(f"t{i}", True, False) for i in range(5)] + [
        _res(f"u{i}", True, True, decided=False) for i in range(20)
    ]
    cfg = PromotionGateConfig(min_coverage=0.8)
    decision = evaluate_promotion_gate(results, cfg)
    assert decision.promoted is False
    assert "coverage" in decision.reason


def test_rollback_on_no_paired_improvement():
    # improved == regressed
    results = (
        [_res(f"imp{i}", True, False) for i in range(5)]
        + [_res(f"reg{i}", False, True) for i in range(5)]
    )
    cfg = PromotionGateConfig(min_discordant_n=5, max_regression_rate=0.6)
    decision = evaluate_promotion_gate(results, cfg)
    assert decision.promoted is False
    assert "no paired improvement" in decision.reason


def test_rollback_on_too_few_discordant():
    results = [_res("imp", True, False)] + [_res(f"tie{i}", True, True) for i in range(20)]
    cfg = PromotionGateConfig(min_discordant_n=5)
    decision = evaluate_promotion_gate(results, cfg)
    assert decision.promoted is False
    assert "discordant_n" in decision.reason


def test_rollback_on_excessive_regressions():
    # 6 improved, 4 regressed -> regression_rate 0.4 > 0.2. p-value for 6v4
    # is ~0.754, so set max_p_value=0.9 to let the regression-rate gate fire.
    results = (
        [_res(f"imp{i}", True, False) for i in range(6)]
        + [_res(f"reg{i}", False, True) for i in range(4)]
    )
    cfg = PromotionGateConfig(min_discordant_n=5, max_p_value=0.9,
                              max_regression_rate=0.2,
                              min_improved_probability_lower=0.0)
    decision = evaluate_promotion_gate(results, cfg)
    assert decision.promoted is False
    assert "regression_rate" in decision.reason


def test_candidate_regression_causes_rollback():
    """Acceptance test: a candidate that regresses on more tasks than it
    improves is rolled back, never promoted."""
    results = (
        [_res(f"imp{i}", True, False) for i in range(3)]
        + [_res(f"reg{i}", False, True) for i in range(8)]
    )
    cfg = PromotionGateConfig(min_discordant_n=5, max_regression_rate=0.5)
    decision = evaluate_promotion_gate(results, cfg)
    assert decision.promoted is False
    assert decision.action == "rollback"
    assert decision.regressed > decision.improved


def test_rollback_on_low_confidence():
    # 7 improved, 3 regressed -> p~0.344 (passes max_p_value=0.5),
    # regression_rate 0.3 (passes max_regression_rate=0.4),
    # improved_probability 0.7 but CP95 lower bound ~0.348 < 0.5 -> confidence
    # gate rejects.
    results = (
        [_res(f"imp{i}", True, False) for i in range(7)]
        + [_res(f"reg{i}", False, True) for i in range(3)]
    )
    cfg = PromotionGateConfig(min_discordant_n=5, max_p_value=0.5,
                              max_regression_rate=0.4,
                              min_improved_probability_lower=0.5)
    decision = evaluate_promotion_gate(results, cfg)
    assert decision.promoted is False
    assert "improved_probability lower bound" in decision.reason


def test_rollback_on_negative_utility_delta():
    # 8 improved, 2 regressed -> p~0.109 (passes 0.5), regression_rate 0.2
    # (passes 0.2 boundary), improved_probability 0.8 CP95 lower ~0.493
    # (passes 0.4). mean utility delta is negative -> utility gate rejects.
    results = (
        [_res(f"imp{i}", True, False, cu=0.1, iu=0.0) for i in range(8)]
        + [_res(f"reg{i}", False, True, cu=-5.0, iu=1.0) for i in range(2)]
    )
    cfg = PromotionGateConfig(min_discordant_n=5, max_p_value=0.5,
                              max_regression_rate=0.2,
                              min_improved_probability_lower=0.4,
                              min_utility_delta=0.0)
    decision = evaluate_promotion_gate(results, cfg)
    assert decision.promoted is False
    assert "mean_utility_delta" in decision.reason


# --- vector lineage + rollback ---

def _lineage(vid, parent, iteration, decision, promote=True):
    return build_lineage_record(
        vector_id=vid, parent_vector_id=parent, iteration=iteration,
        training_dataset_sha256="train-sha", development_dataset_sha256="dev-sha",
        optimizer_parameters={"eta": 0.25, "method": "trust_region"},
        utility_config_hash="cfg-sha", layer=24, token_location="anchor",
        norm=1.0, alpha=1.0, metrics_before={"f1": 0.5},
        metrics_after={"f1": 0.6}, promotion_decision=decision,
    )


def test_lineage_chain_and_rollback():
    # seed -> v1 (promoted) -> v2 (rolled back) -> rollback to v1
    d_promote = evaluate_promotion_gate(
        [_res(f"t{i}", True, False) for i in range(10)], PromotionGateConfig())
    d_reject = evaluate_promotion_gate(
        [_res(f"t{i}", False, True) for i in range(10)], PromotionGateConfig())

    lineage = VectorLineage()
    seed = _lineage("seed", None, 0, d_promote)
    v1 = _lineage("v1", "seed", 1, d_promote)
    v2 = _lineage("v2", "v1", 2, d_reject)
    lineage.append(seed, promote=True)
    lineage.append(v1, promote=True)
    lineage.append(v2, promote=False)  # candidate v2 rejected
    # current stays at v1 (the incumbent)
    assert lineage.current == "v1"
    # rollback to v1 returns its record
    restored = lineage.rollback_to("v1")
    assert restored.vector_id == "v1"
    assert lineage.current == "v1"
    # the chain from seed to current is seed -> v1
    chain = lineage.chain()
    assert [r.vector_id for r in chain] == ["seed", "v1"]


def test_lineage_records_are_immutable_and_hashed():
    d = evaluate_promotion_gate(
        [_res(f"t{i}", True, False) for i in range(10)], PromotionGateConfig())
    rec = _lineage("v1", "seed", 1, d)
    assert rec.lineage_hash
    with pytest.raises(Exception):
        rec.vector_id = "tampered"  # type: ignore[misc]


def test_lineage_hash_changes_with_decision():
    d_promote = evaluate_promotion_gate(
        [_res(f"t{i}", True, False) for i in range(10)], PromotionGateConfig())
    d_reject = evaluate_promotion_gate(
        [_res(f"t{i}", False, True) for i in range(10)], PromotionGateConfig())
    r1 = _lineage("v1", "seed", 1, d_promote)
    r2 = _lineage("v1", "seed", 1, d_reject)
    assert r1.lineage_hash != r2.lineage_hash
