"""Typed, fail-closed outcome verification.

Routing decisions and answer verification are deliberately separate.  A route
label is not evidence that the selected backend answered correctly, and raw
route-generation text is never reused as an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from .scoring import parse_final_or_exact


class VerificationStatus(str, Enum):
    """Canonical verification states used by learning and evaluation."""

    VERIFIED_CORRECT = "verified_correct"
    VERIFIED_INCORRECT = "verified_incorrect"
    UNVERIFIABLE = "unverifiable"
    EXECUTION_FAILED = "execution_failed"
    INVALID_OUTPUT = "invalid_output"


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    verifier: str
    expected: Any | None = None
    observed: Any | None = None
    reason: str | None = None

    @property
    def correct(self) -> bool | None:
        if self.status is VerificationStatus.VERIFIED_CORRECT:
            return True
        if self.status is VerificationStatus.VERIFIED_INCORRECT:
            return False
        return None


@runtime_checkable
class OutcomeVerifier(Protocol):
    """Independent verifier contract for backend outputs."""

    def verify(
        self,
        task: Mapping[str, Any],
        output: Any | None,
        *,
        execution_succeeded: bool = True,
    ) -> VerificationResult:
        ...


@dataclass(frozen=True)
class NumericVerifier:
    """Verify integer answers using only ``FINAL: n`` or an exact integer.

    This intentionally rejects substring matching.  For example, expected
    ``12`` does not match ``312`` or prose containing the number 12.
    """

    expected_field: str = "expected"

    def verify(
        self,
        task: Mapping[str, Any],
        output: Any | None,
        *,
        execution_succeeded: bool = True,
    ) -> VerificationResult:
        expected = task.get(self.expected_field)
        if not execution_succeeded:
            return VerificationResult(
                status=VerificationStatus.EXECUTION_FAILED,
                verifier=type(self).__name__,
                expected=expected,
                reason="backend execution failed",
            )
        if expected is None:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__,
                reason=f"task has no {self.expected_field!r} value",
            )
        if type(expected) is not int:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__,
                expected=expected,
                reason="expected value is not an integer",
            )
        if output is None:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__,
                expected=expected,
                reason="backend produced no output",
            )

        observed = parse_final_or_exact(str(output))
        if observed is None:
            return VerificationResult(
                status=VerificationStatus.INVALID_OUTPUT,
                verifier=type(self).__name__,
                expected=expected,
                reason="output is neither an exact integer nor a FINAL integer",
            )
        return VerificationResult(
            status=(
                VerificationStatus.VERIFIED_CORRECT
                if observed == expected
                else VerificationStatus.VERIFIED_INCORRECT
            ),
            verifier=type(self).__name__,
            expected=expected,
            observed=observed,
        )


@dataclass(frozen=True)
class RouteSuitabilityVerifier:
    """Verify route suitability from an explicit oracle field.

    This is a bootstrap policy-label verifier, not an answer verifier.  Its
    result must never be presented as measured backend correctness.
    """

    oracle_field: str = "utility_oracle"

    def verify_route(
        self,
        task: Mapping[str, Any],
        route: str | None,
    ) -> VerificationResult:
        expected = task.get(self.oracle_field)
        if expected not in {"symbolic", "llm"}:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__,
                expected=expected,
                observed=route,
                reason=f"missing or invalid {self.oracle_field!r}",
            )
        if route not in {"symbolic", "llm"}:
            return VerificationResult(
                status=VerificationStatus.INVALID_OUTPUT,
                verifier=type(self).__name__,
                expected=expected,
                observed=route,
                reason="route is missing or malformed",
            )
        return VerificationResult(
            status=(
                VerificationStatus.VERIFIED_CORRECT
                if route == expected
                else VerificationStatus.VERIFIED_INCORRECT
            ),
            verifier=type(self).__name__,
            expected=expected,
            observed=route,
        )


__all__ = [
    "NumericVerifier",
    "OutcomeVerifier",
    "RouteSuitabilityVerifier",
    "VerificationResult",
    "VerificationStatus",
]
