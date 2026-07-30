"""Section 7 / 20.4 — canonical FINAL_ANSWER verifier tests.

Required tests (Section 20.4):

* test_exact_single_final_answer
* test_missing_final_answer_is_unverifiable
* test_multiple_final_answers_are_unverifiable
* test_conflicting_final_answers_are_unverifiable
* test_incidental_expected_number_does_not_pass
* test_negative_integer_answer
* test_unicode_minus_normalization
"""

from __future__ import annotations

import pytest

from daph_learning.evaluation.canonical_verifier import (
    CanonicalIntegerVerifier,
    ParsedFinalAnswer,
    VerificationStatus,
    parse_canonical_integer_answer,
)


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------

def test_exact_single_final_answer():
    r = parse_canonical_integer_answer("FINAL_ANSWER: 42")
    assert r.status == "VALID"
    assert r.value == 42
    assert r.raw_matches == ("42",)


def test_incidental_expected_number_does_not_pass():
    """A response that mentions the expected number in prose but does not
    emit the canonical field must be UNVERIFIABLE."""
    r = parse_canonical_integer_answer("The answer is not 42. I think 41.")
    assert r.status == "MISSING"
    assert r.value is None


def test_incidental_number_with_canonical_field_uses_field_only():
    """'The answer is not 42.\\nFINAL_ANSWER: 43' verifies as 43 only."""
    r = parse_canonical_integer_answer("The answer is not 42.\nFINAL_ANSWER: 43")
    assert r.status == "VALID"
    assert r.value == 43


def test_missing_final_answer_is_unverifiable():
    r = parse_canonical_integer_answer("the answer is 42")
    assert r.status == "MISSING"
    assert r.value is None
    assert not r.is_verifiable


def test_multiple_final_answers_are_unverifiable():
    r = parse_canonical_integer_answer("FINAL_ANSWER: 42\nFINAL_ANSWER: 43")
    assert r.status in {"CONTRADICTORY", "MULTIPLE"}
    assert r.value is None
    assert not r.is_verifiable


def test_conflicting_final_answers_are_unverifiable():
    r = parse_canonical_integer_answer("FINAL_ANSWER: 42\nFINAL_ANSWER: 43")
    assert r.status == "CONTRADICTORY"
    assert r.value is None


def test_redundant_agreeing_final_answers_are_valid():
    r = parse_canonical_integer_answer("FINAL_ANSWER: 42\nFINAL_ANSWER: 42")
    assert r.status == "VALID"
    assert r.value == 42


def test_malformed_range_is_unverifiable():
    r = parse_canonical_integer_answer("FINAL_ANSWER: 1..10")
    assert r.status == "MALFORMED"
    assert r.value is None


def test_malformed_list_is_unverifiable():
    r = parse_canonical_integer_answer("FINAL_ANSWER: 1, 2, 3")
    assert r.status == "MALFORMED"
    assert r.value is None


def test_malformed_arithmetic_is_unverifiable():
    r = parse_canonical_integer_answer("FINAL_ANSWER: 2+3")
    assert r.status == "MALFORMED"
    assert r.value is None


def test_negative_integer_answer():
    r = parse_canonical_integer_answer("FINAL_ANSWER: -7")
    assert r.status == "VALID"
    assert r.value == -7


def test_unicode_minus_normalization():
    # U+2212 MINUS SIGN and fullwidth hyphen normalize to ASCII '-'.
    r = parse_canonical_integer_answer("FINAL_ANSWER: \u22125")
    assert r.status == "VALID"
    assert r.value == -5


def test_trailing_period_is_stripped():
    r = parse_canonical_integer_answer("FINAL_ANSWER: 42.")
    assert r.status == "VALID"
    assert r.value == 42


def test_case_insensitive_marker():
    r = parse_canonical_integer_answer("final_answer: 99")
    assert r.status == "VALID"
    assert r.value == 99


def test_none_input_is_missing():
    r = parse_canonical_integer_answer(None)  # type: ignore[arg-type]
    assert r.status == "MISSING"


# ------------------------------------------------------------------
# Verifier (closed enum)
# ------------------------------------------------------------------

def test_verifier_correct():
    v = CanonicalIntegerVerifier()
    assert v.verify({"expected": 42}, "FINAL_ANSWER: 42") == VerificationStatus.CORRECT


def test_verifier_incorrect():
    v = CanonicalIntegerVerifier()
    assert v.verify({"expected": 42}, "FINAL_ANSWER: 41") == VerificationStatus.INCORRECT


def test_verifier_missing_field_is_unverifiable():
    v = CanonicalIntegerVerifier()
    assert v.verify({"expected": 42}, "the answer is 42") == VerificationStatus.UNVERIFIABLE


def test_verifier_multiple_is_unverifiable():
    v = CanonicalIntegerVerifier()
    assert v.verify(
        {"expected": 42}, "FINAL_ANSWER: 42\nFINAL_ANSWER: 43"
    ) == VerificationStatus.UNVERIFIABLE


def test_verifier_incidental_number_does_not_credit():
    v = CanonicalIntegerVerifier()
    # The expected number appears in prose but no canonical field.
    assert v.verify(
        {"expected": 42}, "The answer is not 42."
    ) == VerificationStatus.UNVERIFIABLE


def test_verifier_execution_error():
    v = CanonicalIntegerVerifier()
    assert v.verify({"expected": 42}, None,
                    execution_succeeded=False) == VerificationStatus.EXECUTION_ERROR


def test_verifier_timeout():
    v = CanonicalIntegerVerifier()
    assert v.verify({"expected": 42}, "FINAL_ANSWER: 42",
                    timed_out=True) == VerificationStatus.TIMEOUT


def test_verifier_non_integer_expected_is_unverifiable():
    v = CanonicalIntegerVerifier()
    assert v.verify({"expected": "42"}, "FINAL_ANSWER: 42") == VerificationStatus.UNVERIFIABLE


def test_verification_status_enum_is_closed():
    """The enum must contain exactly the five closed statuses."""
    members = {s.name for s in VerificationStatus}
    assert members == {"CORRECT", "INCORRECT", "UNVERIFIABLE",
                       "EXECUTION_ERROR", "TIMEOUT"}


def test_legacy_first_int_does_not_credit_qualification():
    """The legacy permissive parser must not be wired into qualification.
    This test documents that the canonical verifier ignores it: a response
    whose first integer equals the expected value but lacks the canonical
    field is UNVERIFIABLE."""
    from daph_learning.evaluation.scoring import parse_legacy_first_int
    # Legacy parser would extract 42 from prose.
    assert parse_legacy_first_int("The answer is not 42.") == 42
    # Canonical verifier refuses to credit it.
    v = CanonicalIntegerVerifier()
    assert v.verify({"expected": 42}, "The answer is not 42.") == \
        VerificationStatus.UNVERIFIABLE
