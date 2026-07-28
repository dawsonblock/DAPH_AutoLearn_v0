"""V039-004 — replace categorical accuracy oracle.

Record ``symbolic_correct``, ``llm_correct``, ``accuracy_delta`` where
``+1 = symbolic only correct``, ``0 = tie``, ``-1 = LLM only correct``.
Utility is kept separate from accuracy. Both-fail produces no fabricated
preference.
"""

from __future__ import annotations

from daph_learning.autolearn.counterfactual import (
    ABSTAIN,
    UtilityConfig,
    derive_learning_target,
    execute_both_backends,
)


def _task(expected=42, oracle="symbolic"):
    return {"task_id": "t1", "expected": expected,
            "capability_ids": ["integer_arithmetic"], "utility_oracle": oracle}


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


def _accuracy_delta(outcome):
    s = outcome.backends["symbolic"].is_correct
    l = outcome.backends["llm"].is_correct
    if s and not l:
        return 1
    if l and not s:
        return -1
    return 0  # tie (both correct or both wrong)


def test_accuracy_delta_symbolic_only_correct():
    sym, llm = _runners(42, 99)
    o = execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=sym, llm_runner=llm)
    assert _accuracy_delta(o) == 1


def test_accuracy_delta_llm_only_correct():
    sym, llm = _runners(99, 42)
    o = execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=sym, llm_runner=llm)
    assert _accuracy_delta(o) == -1


def test_accuracy_delta_tie_both_correct():
    sym, llm = _runners(42, 42)
    o = execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=sym, llm_runner=llm)
    assert _accuracy_delta(o) == 0


def test_accuracy_delta_tie_both_wrong():
    sym, llm = _runners(99, 99)
    o = execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=sym, llm_runner=llm)
    assert _accuracy_delta(o) == 0


def test_utility_separate_from_accuracy():
    """Both backends correct (accuracy tie) but symbolic much slower.
    Accuracy delta is 0 but utility argmax flips to LLM under lambda_time."""
    sym, llm = _runners(42, 42)
    # override symbolic to be slow
    def sym_slow(t):
        return {"output": "FINAL: 42", "execution_succeeded": True,
                "latency_ms": 100000.0, "compute_cost": 0.0, "failure_type": None}
    o = execute_both_backends(_task(), backend_selected="symbolic",
                              symbolic_runner=sym_slow, llm_runner=llm)
    assert _accuracy_delta(o) == 0  # accuracy tie
    cfg = UtilityConfig(lambda_time=10.0, lambda_risk=1.0, abstention_band=0.1,
                        time_reference_ms=1000.0)
    t = derive_learning_target(o, cfg)
    # utility separates: symbolic penalized for latency -> llm preferred
    assert t.target == "llm"
