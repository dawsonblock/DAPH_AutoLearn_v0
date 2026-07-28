"""V039-004 — both backends fail: no fabricated preference.

When both symbolic and LLM fail, the learning target must be ABSTAIN with
zero weight, never a fabricated symbolic or llm preference.
"""

from __future__ import annotations

from daph_learning.autolearn.counterfactual import (
    ABSTAIN,
    UtilityConfig,
    derive_learning_target,
    execute_both_backends,
)


def _task(expected=42):
    return {"task_id": "t1", "expected": expected,
            "capability_ids": ["integer_arithmetic"], "utility_oracle": "symbolic"}


def _fail_runner(t):
    return {"output": None, "execution_succeeded": False,
            "latency_ms": None, "compute_cost": None, "failure_type": "execution_error"}


def test_both_fail_target_is_abstain():
    o = execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=_fail_runner, llm_runner=_fail_runner)
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.5)
    t = derive_learning_target(o, cfg)
    assert t.target == ABSTAIN
    assert t.weight == 0.0


def test_both_fail_delta_zero():
    o = execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=_fail_runner, llm_runner=_fail_runner)
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.0)
    t = derive_learning_target(o, cfg)
    # both failed with equal risk -> ΔU = 0
    assert t.delta_u == 0.0


def test_both_fail_no_correct_backend():
    o = execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=_fail_runner, llm_runner=_fail_runner)
    assert o.backends["symbolic"].is_correct is False
    assert o.backends["llm"].is_correct is False
