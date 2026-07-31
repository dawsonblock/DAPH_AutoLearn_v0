"""V039-002 — verifier abstraction with task-appropriate output schemas.

Acceptance tests for Item 5:
* each verifier type behaves correctly
* the router is never its own verifier
* composite verifier precedence rules
* nonnumeric tasks use the appropriate output schema (dispatch via
  ``select_verifier_for_task``)
* free-form tasks are never credited by guessing (UNVERIFIABLE without a judge)
"""

from __future__ import annotations

import pytest

from daph_learning.evaluation.verifiers import (
    CompositeVerifier,
    ExactMatchVerifier,
    FreeFormVerifier,
    KNOWN_SCHEMAS,
    NumericVerifier,
    ReferenceAnswerVerifier,
    SCHEMA_FREE_FORM,
    SCHEMA_INTEGER,
    SCHEMA_REFERENCE,
    SCHEMA_STRUCTURED,
    StructuredSchemaVerifier,
    select_verifier_for_task,
)
from daph_learning.evaluation.verification import VerificationStatus


# --- NumericVerifier (integer schema) ---

def test_integer_schema_uses_numeric_verifier():
    task = {"task_id": "t1", "expected": 42, "output_schema": SCHEMA_INTEGER}
    v = select_verifier_for_task(task)
    assert isinstance(v, NumericVerifier)
    assert v.verify(task, "FINAL: 42").status is VerificationStatus.VERIFIED_CORRECT
    assert v.verify(task, "FINAL: 99").status is VerificationStatus.VERIFIED_INCORRECT


def test_integer_schema_accepts_final_answer_format():
    """The canonical FINAL_ANSWER: format must also be verified."""
    task = {"task_id": "t1", "expected": 42, "output_schema": SCHEMA_INTEGER}
    v = select_verifier_for_task(task)
    assert v.verify(task, "FINAL_ANSWER: 42").status is VerificationStatus.VERIFIED_CORRECT
    assert v.verify(task, "FINAL_ANSWER: 99").status is VerificationStatus.VERIFIED_INCORRECT


def test_integer_schema_rejects_substring_match():
    task = {"task_id": "t1", "expected": 12, "output_schema": SCHEMA_INTEGER}
    v = select_verifier_for_task(task)
    assert v.verify(task, "312").status is VerificationStatus.VERIFIED_INCORRECT


# --- ExactMatchVerifier ---

def test_exact_schema_exact_match():
    task = {"task_id": "t1", "expected": "yes", "output_schema": "exact"}
    v = select_verifier_for_task(task)
    assert isinstance(v, ExactMatchVerifier)
    assert v.verify(task, "yes").status is VerificationStatus.VERIFIED_CORRECT
    assert v.verify(task, "no").status is VerificationStatus.VERIFIED_INCORRECT


def test_exact_match_case_insensitive_option():
    v = ExactMatchVerifier(case_sensitive=False)
    task = {"expected": "Yes"}
    assert v.verify(task, "yes").status is VerificationStatus.VERIFIED_CORRECT


# --- ReferenceAnswerVerifier (free-form / knowledge) ---

def test_reference_schema_uses_reference_verifier():
    task = {"task_id": "t1", "reference_answer": "Paris",
            "output_schema": SCHEMA_REFERENCE}
    v = select_verifier_for_task(task)
    assert isinstance(v, ReferenceAnswerVerifier)
    assert v.verify(task, "The capital is Paris.").status is VerificationStatus.VERIFIED_CORRECT
    assert v.verify(task, "The capital is London.").status is VerificationStatus.VERIFIED_INCORRECT


def test_reference_verifier_normalizes_whitespace_and_case():
    v = ReferenceAnswerVerifier()
    task = {"reference_answer": "  the  Sky  is  Blue "}
    assert v.verify(task, "the sky is blue").status is VerificationStatus.VERIFIED_CORRECT


# --- StructuredSchemaVerifier ---

def test_structured_schema_validates_fields_and_types():
    spec = {"answer": int, "confidence": float}
    task = {"task_id": "t1", "output_schema": SCHEMA_STRUCTURED,
            "output_schema_spec": spec}
    v = select_verifier_for_task(task)
    assert isinstance(v, StructuredSchemaVerifier)
    good = '{"answer": 42, "confidence": 0.9}'
    assert v.verify(task, good).status is VerificationStatus.VERIFIED_CORRECT
    missing = '{"answer": 42}'
    assert v.verify(task, missing).status is VerificationStatus.INVALID_OUTPUT
    bad_type = '{"answer": 42, "confidence": "high"}'
    assert v.verify(task, bad_type).status is VerificationStatus.INVALID_OUTPUT
    not_json = "not json"
    assert v.verify(task, not_json).status is VerificationStatus.INVALID_OUTPUT


def test_structured_schema_rejects_bool_for_int():
    spec = {"answer": int}
    task = {"output_schema_spec": spec}
    v = StructuredSchemaVerifier()
    assert v.verify(task, '{"answer": true}').status is VerificationStatus.INVALID_OUTPUT


# --- FreeFormVerifier ---

def test_free_form_schema_unverifiable_without_judge():
    task = {"task_id": "t1", "output_schema": SCHEMA_FREE_FORM}
    v = select_verifier_for_task(task)
    assert isinstance(v, FreeFormVerifier)
    result = v.verify(task, "a long essay about the sky")
    assert result.status is VerificationStatus.UNVERIFIABLE


def test_free_form_with_judge_uses_judge_verdict():
    judge = lambda task, output: "blue" in str(output).lower()  # noqa: E731
    v = FreeFormVerifier(judge=judge)  # type: ignore[arg-type]
    task = {"task_id": "t1"}
    assert v.verify(task, "the sky is blue").status is VerificationStatus.VERIFIED_CORRECT
    assert v.verify(task, "the sky is green").status is VerificationStatus.VERIFIED_INCORRECT


def test_free_form_judge_abstention_yields_unverifiable():
    judge = lambda task, output: None  # noqa: E731
    v = FreeFormVerifier(judge=judge)  # type: ignore[arg-type]
    assert v.verify(task={"task_id": "t1"}, output="x").status is VerificationStatus.UNVERIFIABLE


def test_free_form_judge_exception_yields_unverifiable_not_crash():
    def bad_judge(task, output):
        raise RuntimeError("judge broken")
    v = FreeFormVerifier(judge=bad_judge)  # type: ignore[arg-type]
    result = v.verify({"task_id": "t1"}, "x")
    assert result.status is VerificationStatus.UNVERIFIABLE
    assert "judge raised" in (result.reason or "")


# --- CompositeVerifier precedence ---

def test_composite_first_definitive_wins():
    composite = CompositeVerifier((
        StructuredSchemaVerifier(),  # INVALID_OUTPUT (not definitive correct/incorrect)
        NumericVerifier(),
    ))
    task = {"output_schema_spec": {"answer": int}, "expected": 42}
    # output is not JSON -> structured returns INVALID_OUTPUT (non-definitive);
    # numeric parses FINAL: 42 -> CORRECT (definitive) wins.
    result = composite.verify(task, "FINAL: 42")
    assert result.status is VerificationStatus.VERIFIED_CORRECT
    assert result.verifier.startswith("CompositeVerifier")


def test_composite_all_unverifiable_yields_unverifiable():
    composite = CompositeVerifier((
        ReferenceAnswerVerifier(),  # no reference_answer -> UNVERIFIABLE
        FreeFormVerifier(),         # no judge -> UNVERIFIABLE
    ))
    result = composite.verify({"task_id": "t1"}, "some output")
    assert result.status is VerificationStatus.UNVERIFIABLE


def test_composite_execution_failed_short_circuits():
    composite = CompositeVerifier((NumericVerifier(),))
    result = composite.verify({"expected": 42}, None, execution_succeeded=False)
    assert result.status is VerificationStatus.EXECUTION_FAILED


# --- Router is never its own verifier ---

def test_select_verifier_never_returns_router():
    """The dispatch returns an independent verifier, never a router object."""
    for schema in KNOWN_SCHEMAS:
        task = {"task_id": "t1", "output_schema": schema, "expected": 1,
                "reference_answer": "x", "output_schema_spec": {"a": int}}
        v = select_verifier_for_task(task)
        # every returned verifier implements the OutcomeVerifier protocol and
        # is independent of any router.
        assert hasattr(v, "verify")
        assert not type(v).__name__.endswith("Router")


def test_unknown_schema_fails_closed_toward_check_not_credit():
    task = {"task_id": "t1", "output_schema": "mystery", "expected": "yes"}
    v = select_verifier_for_task(task)
    # unknown schema with non-int expected -> ExactMatchVerifier (a real check)
    assert isinstance(v, ExactMatchVerifier)
    assert v.verify(task, "no").status is VerificationStatus.VERIFIED_INCORRECT
