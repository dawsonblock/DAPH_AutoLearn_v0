"""V039-003 — three-action routing with ABSTAIN.

``symbolic wrong ∧ llm wrong`` must not produce a fabricated preference.
The ΔU abstention band around zero yields the ABSTAIN target for ties and
both-fail cases.
"""

from __future__ import annotations

from daph_learning.autolearn.counterfactual import (
    ABSTAIN,
    UtilityConfig,
    derive_learning_target,
    execute_both_backends,
)


def _task(expected=42, oracle="symbolic"):
    return {
        "task_id": "t1",
        "expected": expected,
        "capability_ids": ["integer_arithmetic"],
        "utility_oracle": oracle,
    }


def _llm(answer):
    def _run(task):
        return {
            "output": None if answer is None else f"FINAL: {answer}",
            "execution_succeeded": answer is not None,
            "latency_ms": 50.0,
            "compute_cost": 0.01,
            "failure_type": None if answer is not None else "execution_error",
        }
    return _run


def _sym(answer):
    def _run(task):
        return {
            "output": None if answer is None else f"FINAL: {answer}",
            "execution_succeeded": answer is not None,
            "latency_ms": 1.0,
            "compute_cost": 0.0,
            "failure_type": None if answer is not None else "execution_error",
        }
    return _run


def test_both_fail_yields_abstain():
    task = _task()
    outcome = execute_both_backends(
        task,
        backend_selected="symbolic",
        symbolic_runner=_sym(None),
        llm_runner=_llm(None),
    )
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.5)
    target = derive_learning_target(outcome, cfg)
    # both failed: U_sym = -λ_r, U_llm = -λ_r -> ΔU = 0 -> abstain
    assert target.delta_u == 0.0
    assert target.target == ABSTAIN
    assert target.weight == 0.0


def test_both_correct_tie_yields_abstain():
    task = _task()
    outcome = execute_both_backends(
        task,
        backend_selected="symbolic",
        symbolic_runner=_sym(42),
        llm_runner=_llm(42),
    )
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.5)
    target = derive_learning_target(outcome, cfg)
    # both correct, no latency/compute terms -> ΔU = 0 -> abstain
    assert target.target == ABSTAIN
    assert target.weight == 0.0


def test_clear_symbolic_preference_outside_band():
    task = _task()
    outcome = execute_both_backends(
        task,
        backend_selected="symbolic",
        symbolic_runner=_sym(42),
        llm_runner=_llm(99),
    )
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.5)
    target = derive_learning_target(outcome, cfg)
    assert target.target == "symbolic"
    assert target.weight > 0.0


def test_clear_llm_preference_outside_band():
    task = _task()
    outcome = execute_both_backends(
        task,
        backend_selected="llm",
        symbolic_runner=_sym(99),
        llm_runner=_llm(42),
    )
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.5)
    target = derive_learning_target(outcome, cfg)
    assert target.target == "llm"
    assert target.weight > 0.0


def test_within_band_abstains_even_if_nonzero_delta():
    """A small ΔU inside the band abstains despite a nonzero preference."""
    task = _task()
    # symbolic correct with small latency cost; llm correct with no cost.
    # ΔU is small positive but inside the band.
    outcome = execute_both_backends(
        task,
        backend_selected="symbolic",
        symbolic_runner=_sym(42),
        llm_runner=_llm(42),
    )
    # add a tiny latency term to symbolic so ΔU is small nonzero
    from daph_learning.autolearn.counterfactual import compute_backend_utility
    cfg = UtilityConfig(lambda_time=0.001, lambda_risk=1.0, abstention_band=1.0,
                        time_reference_ms=1000.0)
    target = derive_learning_target(outcome, cfg)
    # symbolic latency 1ms -> time_term = 0.001*(1/1000) = 1e-6; ΔU = -1e-6
    assert abs(target.delta_u) <= cfg.abstention_band
    assert target.target == ABSTAIN
