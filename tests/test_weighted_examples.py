"""V039-006 — utility-weighted learning examples.

``w_i = g(|ΔU_i|)`` (default ``min(w_max, |ΔU|)``). High-confidence backend
preference -> more weight. Near ties / abstain / both-fail -> zero weight.
"""

from __future__ import annotations

import pytest

from daph_learning.autolearn.counterfactual import (
    ABSTAIN,
    UtilityConfig,
    derive_learning_target,
    execute_both_backends,
)


def _task(expected=42, oracle="symbolic"):
    return {
        "task_id": "t1", "expected": expected,
        "capability_ids": ["integer_arithmetic"], "utility_oracle": oracle,
    }


def _runners(sym_ans, llm_ans):
    def _sym(t):
        return {"output": None if sym_ans is None else f"FINAL: {sym_ans}",
                "execution_succeeded": sym_ans is not None,
                "latency_ms": 1.0, "compute_cost": 0.0,
                "failure_type": None if sym_ans is not None else "execution_error"}
    def _llm(t):
        return {"output": None if llm_ans is None else f"FINAL: {llm_ans}",
                "execution_succeeded": llm_ans is not None,
                "latency_ms": 50.0, "compute_cost": 0.01,
                "failure_type": None if llm_ans is not None else "execution_error"}
    return _sym, _llm


def test_weight_grows_with_abs_delta():
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.1)
    # small preference: symbolic correct, llm wrong but only risk diff = 1
    sym, llm = _runners(42, 99)
    t_small = derive_learning_target(
        execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=sym, llm_runner=llm), cfg, w_max=1000.0)
    # large preference: add a big latency penalty to llm via custom runners
    def llm_slow(t):
        return {"output": "FINAL: 99", "execution_succeeded": True,
                "latency_ms": 100000.0, "compute_cost": 1.0, "failure_type": None}
    t_large = derive_learning_target(
        execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=sym, llm_runner=llm_slow),
        UtilityConfig(lambda_time=1.0, lambda_compute=1.0, lambda_risk=1.0,
                      abstention_band=0.1, time_reference_ms=1000.0,
                      compute_reference=1.0), w_max=1000.0)
    assert t_small.weight > 0
    assert t_large.weight > t_small.weight


def test_weight_zero_for_abstain():
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.5)
    sym, llm = _runners(42, 42)  # both correct -> tie -> abstain
    t = derive_learning_target(
        execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=sym, llm_runner=llm), cfg)
    assert t.target == ABSTAIN
    assert t.weight == 0.0


def test_weight_zero_for_both_fail():
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.5)
    sym, llm = _runners(None, None)  # both fail
    t = derive_learning_target(
        execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=sym, llm_runner=llm), cfg)
    assert t.target == ABSTAIN
    assert t.weight == 0.0


def test_weight_capped_at_w_max():
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.1)
    sym, llm = _runners(42, 99)
    t = derive_learning_target(
        execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=sym, llm_runner=llm), cfg, w_max=0.3)
    assert t.weight == pytest.approx(0.3)


def test_custom_weight_fn_used():
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.1)
    sym, llm = _runners(42, 99)
    t = derive_learning_target(
        execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=sym, llm_runner=llm), cfg,
        weight_fn=lambda d, wmax: 999.0)
    assert t.weight == 999.0
