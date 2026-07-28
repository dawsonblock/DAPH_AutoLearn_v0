"""V039-005 — explicit utility model and argmax selection.

``U_b = Q_b − λ_t·T_b − λ_c·C_b − λ_r·R_b``. ``y*(x) = argmax_a U(a, x)``
outside the abstention band. Verifies argmax behaviour across tie /
latency-dominant / failure cases, and that the utility config is frozen
and hashed.
"""

from __future__ import annotations

import pytest

from daph_learning.autolearn.counterfactual import (
    ABSTAIN,
    UtilityConfig,
    compute_backend_utility,
    dataclass_replace,
    derive_learning_target,
    execute_both_backends,
)
from daph_learning.autolearn.outcome import BackendExecutionRecord
from daph_learning.evaluation.verification import (
    VerificationResult,
    VerificationStatus,
)


def _rec(backend, correct, latency=None, compute=None, failed=False):
    if failed:
        status = VerificationStatus.EXECUTION_FAILED
        failure_type = "execution_error"
    elif correct:
        status = VerificationStatus.VERIFIED_CORRECT
        failure_type = None
    else:
        status = VerificationStatus.VERIFIED_INCORRECT
        failure_type = None
    return BackendExecutionRecord(
        backend=backend,
        selected=(backend == "symbolic"),
        executed=True,
        output="FINAL: 42" if correct else "FINAL: 99",
        verification=VerificationResult(status=status, verifier="NumericVerifier"),
        latency_ms=latency,
        compute_cost=compute,
        failure_type=failure_type,
    )


def test_quality_reward_only_for_correct():
    cfg = UtilityConfig().freeze()
    u_correct = compute_backend_utility(_rec("symbolic", correct=True), cfg)
    u_wrong = compute_backend_utility(_rec("symbolic", correct=False), cfg)
    assert u_correct.utility == 1.0
    assert u_wrong.utility == 0.0


def test_latency_term_reduces_utility():
    cfg = UtilityConfig(lambda_time=1.0, time_reference_ms=1000.0).freeze()
    u = compute_backend_utility(_rec("symbolic", correct=True, latency=500.0), cfg)
    # U = 1 - 1.0*(500/1000) = 0.5
    assert u.utility == pytest.approx(0.5)
    assert u.time_term == pytest.approx(0.5)


def test_compute_term_reduces_utility():
    cfg = UtilityConfig(lambda_compute=2.0, compute_reference=0.05).freeze()
    u = compute_backend_utility(_rec("symbolic", correct=True, compute=0.025), cfg)
    # U = 1 - 2.0*(0.025/0.05) = 1 - 1.0 = 0.0
    assert u.utility == pytest.approx(0.0)


def test_risk_penalty_for_failure():
    cfg = UtilityConfig(lambda_risk=5.0).freeze()
    u = compute_backend_utility(_rec("symbolic", correct=False, failed=True), cfg)
    # U = 0 - 0 - 0 - 5*1 = -5
    assert u.utility == pytest.approx(-5.0)
    assert u.risk_term == pytest.approx(5.0)


def test_argmax_symbolic_when_symbolic_better():
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.1)
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"],
            "utility_oracle": "symbolic"}
    outcome = execute_both_backends(
        task, backend_selected="symbolic",
        symbolic_runner=lambda t: {"output": "FINAL: 42", "execution_succeeded": True,
                                   "latency_ms": 1.0, "compute_cost": 0.0, "failure_type": None},
        llm_runner=lambda t: {"output": "FINAL: 99", "execution_succeeded": True,
                              "latency_ms": 50.0, "compute_cost": 0.01, "failure_type": None},
    )
    target = derive_learning_target(outcome, cfg)
    assert target.target == "symbolic"


def test_argmax_llm_when_llm_better():
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.1)
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"],
            "utility_oracle": "llm"}
    outcome = execute_both_backends(
        task, backend_selected="llm",
        symbolic_runner=lambda t: {"output": "FINAL: 99", "execution_succeeded": True,
                                   "latency_ms": 1.0, "compute_cost": 0.0, "failure_type": None},
        llm_runner=lambda t: {"output": "FINAL: 42", "execution_succeeded": True,
                              "latency_ms": 50.0, "compute_cost": 0.01, "failure_type": None},
    )
    target = derive_learning_target(outcome, cfg)
    assert target.target == "llm"


def test_latency_dominant_can_flip_preference():
    """Symbolic correct but very slow; LLM correct and fast. With a high
    lambda_time the utility argmax flips to LLM."""
    cfg = UtilityConfig(lambda_time=10.0, lambda_risk=1.0, abstention_band=0.1,
                        time_reference_ms=1000.0)
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"],
            "utility_oracle": "symbolic"}
    outcome = execute_both_backends(
        task, backend_selected="symbolic",
        symbolic_runner=lambda t: {"output": "FINAL: 42", "execution_succeeded": True,
                                   "latency_ms": 2000.0, "compute_cost": 0.0, "failure_type": None},
        llm_runner=lambda t: {"output": "FINAL: 42", "execution_succeeded": True,
                              "latency_ms": 10.0, "compute_cost": 0.0, "failure_type": None},
    )
    target = derive_learning_target(outcome, cfg)
    # U_sym = 1 - 10*(2000/1000) = 1 - 20 = -19; U_llm = 1 - 10*(10/1000) = 1 - 0.1 = 0.9
    # ΔU = -19 - 0.9 = -19.9 -> llm
    assert target.target == "llm"
    assert target.delta_u < 0


def test_utility_config_is_frozen_and_hashed():
    cfg = UtilityConfig(lambda_time=0.5, lambda_risk=2.0)
    assert cfg.config_sha256 is None
    frozen = cfg.freeze()
    assert frozen.config_sha256 is not None
    # re-freezing is idempotent and yields the same hash
    assert frozen.freeze().config_sha256 == frozen.config_sha256
    # a different config yields a different hash
    other = dataclass_replace(frozen, lambda_time=0.6).freeze()
    assert other.config_sha256 != frozen.config_sha256


def test_utility_breakdown_records_config_hash():
    cfg = UtilityConfig(lambda_risk=1.0).freeze()
    u = compute_backend_utility(_rec("symbolic", correct=True), cfg)
    assert u.config_sha256 == cfg.config_sha256


def test_abstention_band_zero_means_exact_tie_only():
    """With band=0, only an exact ΔU=0 abstains (both-correct-no-cost or
    both-fail-same-risk). A tiny nonzero preference is acted on."""
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.0)
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"],
            "utility_oracle": "symbolic"}
    # both correct, symbolic slightly slower -> ΔU tiny negative
    outcome = execute_both_backends(
        task, backend_selected="symbolic",
        symbolic_runner=lambda t: {"output": "FINAL: 42", "execution_succeeded": True,
                                   "latency_ms": 2.0, "compute_cost": 0.0, "failure_type": None},
        llm_runner=lambda t: {"output": "FINAL: 42", "execution_succeeded": True,
                              "latency_ms": 1.0, "compute_cost": 0.0, "failure_type": None},
    )
    target = derive_learning_target(outcome, cfg)
    # with lambda_time=0, ΔU = 0 -> abstain
    assert target.target == ABSTAIN
