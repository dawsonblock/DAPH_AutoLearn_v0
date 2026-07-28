"""Acceptance test: nonnumeric tasks use appropriate output schemas.

A nonnumeric (free-form / knowledge / structured) task must be verified by a
schema-appropriate verifier, not the integer NumericVerifier. This guards
against the historical failure mode where every task was scored as an
integer and nonnumeric tasks were silently marked unverifiable or wrongly
credited.
"""

from __future__ import annotations

from daph_learning.autolearn.counterfactual import (
    UtilityConfig,
    derive_learning_target,
    execute_both_backends,
)
from daph_learning.evaluation.verifiers import (
    NumericVerifier,
    ReferenceAnswerVerifier,
    StructuredSchemaVerifier,
    select_verifier_for_task,
)
from daph_learning.evaluation.verification import VerificationStatus


def test_nonnumeric_task_dispatches_to_reference_verifier():
    task = {
        "task_id": "k1",
        "output_schema": "reference",
        "reference_answer": "Paris",
        "capability_ids": [],
        "utility_oracle": "llm",
    }
    v = select_verifier_for_task(task)
    assert isinstance(v, ReferenceAnswerVerifier)
    assert not isinstance(v, NumericVerifier)


def test_nonnumeric_task_verified_correctly_via_counterfactual_execution():
    """A knowledge task answered correctly by the LLM is credited via the
    reference verifier, not the numeric one (which would mark it
    unverifiable)."""
    task = {
        "task_id": "k2",
        "output_schema": "reference",
        "reference_answer": "Paris",
        "capability_ids": [],
        "utility_oracle": "llm",
    }
    verifier = select_verifier_for_task(task)
    outcome = execute_both_backends(
        task,
        backend_selected="llm",
        symbolic_runner=lambda t: {"output": None, "execution_succeeded": False,
                                   "latency_ms": None, "compute_cost": None,
                                   "failure_type": "unsupported"},
        llm_runner=lambda t: {"output": "The capital of France is Paris.",
                              "execution_succeeded": True, "latency_ms": 80.0,
                              "compute_cost": 0.02, "failure_type": None},
        verifier=verifier,
    )
    # LLM verified correct via reference containment; symbolic unsupported.
    assert outcome.backends["llm"].verification.status is VerificationStatus.VERIFIED_CORRECT
    assert outcome.backends["symbolic"].verification.status is VerificationStatus.EXECUTION_FAILED
    assert outcome.backends["llm"].is_correct is True
    # ΔU target favours LLM
    cfg = UtilityConfig(lambda_risk=1.0, abstention_band=0.1)
    target = derive_learning_target(outcome, cfg)
    assert target.target == "llm"


def test_structured_task_uses_structured_verifier():
    task = {
        "task_id": "s1",
        "output_schema": "structured",
        "output_schema_spec": {"answer": int, "confidence": float},
        "capability_ids": [],
        "utility_oracle": "llm",
    }
    v = select_verifier_for_task(task)
    assert isinstance(v, StructuredSchemaVerifier)
    good = '{"answer": 42, "confidence": 0.9}'
    assert v.verify(task, good).status is VerificationStatus.VERIFIED_CORRECT
    bad = '{"answer": "x", "confidence": 0.9}'
    assert v.verify(task, bad).status is VerificationStatus.INVALID_OUTPUT


def test_integer_task_still_uses_numeric_verifier():
    task = {"task_id": "a1", "output_schema": "integer", "expected": 42,
            "capability_ids": ["integer_arithmetic"], "utility_oracle": "symbolic"}
    v = select_verifier_for_task(task)
    assert isinstance(v, NumericVerifier)
    assert v.verify(task, "FINAL: 42").status is VerificationStatus.VERIFIED_CORRECT


def test_free_form_task_not_credited_by_numeric_verifier():
    """A free-form essay task must NOT be routed to the numeric verifier,
    which would either mark it unverifiable or, worse, parse a stray integer
    and credit it."""
    task = {"task_id": "f1", "output_schema": "free_form",
            "capability_ids": [], "utility_oracle": "llm"}
    v = select_verifier_for_task(task)
    assert not isinstance(v, NumericVerifier)
    # an essay that happens to contain "42" must not be credited as integer 42
    result = v.verify(task, "In 42 years the climate will change.")
    assert result.status is not VerificationStatus.VERIFIED_CORRECT
