"""V039-001 — counterfactual backend execution.

Both backends are executed on every feasible task and independently verified.
Acceptance: paired outcomes for a small task set; both backends executed
exactly once per task; an unexecuted backend is never correct.
"""

from __future__ import annotations

from daph_learning.autolearn.counterfactual import (
    UtilityConfig,
    derive_learning_target,
    execute_both_backends,
)
from daph_learning.evaluation.verification import VerificationStatus


def _task(expected=42, oracle="symbolic"):
    return {
        "task_id": "t1",
        "expected": expected,
        "capability_ids": ["integer_arithmetic"],
        "utility_oracle": oracle,
    }


def _sym_runner(answer):
    def _run(task):
        return {
            "output": None if answer is None else f"FINAL: {answer}",
            "execution_succeeded": answer is not None,
            "latency_ms": 1.0,
            "compute_cost": 0.0,
            "failure_type": None if answer is not None else "execution_error",
        }
    return _run


def _llm_runner(answer):
    def _run(task):
        return {
            "output": f"FINAL: {answer}",
            "execution_succeeded": True,
            "latency_ms": 50.0,
            "compute_cost": 0.01,
            "failure_type": None,
        }
    return _run


def test_both_backends_executed_exactly_once():
    task = _task()
    outcome = execute_both_backends(
        task,
        backend_selected="symbolic",
        symbolic_runner=_sym_runner(42),
        llm_runner=_llm_runner(99),
    )
    assert outcome.both_backends_executed is True
    # exactly two backend records, one per backend
    assert set(outcome.backends) == {"symbolic", "llm"}
    assert len(outcome.backends) == 2


def test_paired_outcomes_symbolic_correct_llm_wrong():
    task = _task()
    outcome = execute_both_backends(
        task,
        backend_selected="symbolic",
        symbolic_runner=_sym_runner(42),
        llm_runner=_llm_runner(99),
    )
    assert outcome.backends["symbolic"].is_correct is True
    assert outcome.backends["llm"].is_correct is False
    assert outcome.backends["llm"].verification.status is VerificationStatus.VERIFIED_INCORRECT


def test_paired_outcomes_both_correct():
    task = _task()
    outcome = execute_both_backends(
        task,
        backend_selected="symbolic",
        symbolic_runner=_sym_runner(42),
        llm_runner=_llm_runner(42),
    )
    assert outcome.backends["symbolic"].is_correct is True
    assert outcome.backends["llm"].is_correct is True


def test_skip_llm_keeps_invariant_unexecuted_never_correct():
    task = _task()
    outcome = execute_both_backends(
        task,
        backend_selected="symbolic",
        symbolic_runner=_sym_runner(42),
        llm_runner=None,
        skip_llm=True,
    )
    assert outcome.backends["llm"].executed is False
    assert outcome.backends["llm"].is_correct is False
    assert outcome.backends["llm"].verification.status is VerificationStatus.NOT_EXECUTED


def test_runner_exception_recorded_as_failure_not_crash():
    task = _task()

    def bad_llm(task):
        raise RuntimeError("boom")

    outcome = execute_both_backends(
        task,
        backend_selected="symbolic",
        symbolic_runner=_sym_runner(42),
        llm_runner=bad_llm,
    )
    llm = outcome.backends["llm"]
    assert llm.executed is True
    assert llm.is_correct is False
    assert llm.failure_type is not None
    assert "runner_error" in llm.failure_type


def test_utility_recorded_for_both_backends():
    task = _task()
    outcome = execute_both_backends(
        task,
        backend_selected="symbolic",
        symbolic_runner=_sym_runner(42),
        llm_runner=_llm_runner(99),
    )
    cfg = UtilityConfig(lambda_time=0.01, lambda_compute=1.0, lambda_risk=1.0)
    target = derive_learning_target(outcome, cfg)
    # symbolic correct -> Q=1, R=0; llm wrong-but-valid -> Q=0, R=0
    # (a wrong answer is not a failure; it just earns no quality reward)
    assert target.u_symbolic.quality == 1.0
    assert target.u_llm.quality == 0.0
    assert target.u_symbolic.risk_term == 0.0
    assert target.u_llm.risk_term == 0.0
    assert target.delta_u > 0
    assert target.target == "symbolic"
