"""v0.3.9 (V039-002) — verifier abstraction with task-appropriate output schemas.

Item 5 of the recommended repair sequence: start with exact SAT/SMT/arithmetic
verification, then add separately judged free-form tasks with task-appropriate
output schemas. The router must not be its own verifier.

Each task declares an ``output_schema`` (defaulting to ``"integer"`` for the
existing arithmetic protocol). :func:`select_verifier_for_task` dispatches to
the schema-appropriate verifier:

- ``"integer"``   -> :class:`NumericVerifier` (exact integer / ``FINAL: n``)
- ``"exact"``     -> :class:`ExactMatchVerifier` (exact string equality)
- ``"reference"`` -> :class:`ReferenceAnswerVerifier` (normalized reference match)
- ``"structured"``-> :class:`StructuredSchemaVerifier` (JSON field/type schema)
- ``"free_form"`` -> :class:`FreeFormVerifier` (external judge; UNVERIFIABLE
  until a judge callable is supplied, so free-form tasks are never credited
  by guessing)

A :class:`CompositeVerifier` applies a sequence of verifiers and lets the
first definitive (correct/incorrect) result win, so a task can require both a
schema check and a reference-answer check.

This module reuses the :class:`OutcomeVerifier` protocol and
:class:`VerificationResult` / :class:`VerificationStatus` from
:mod:`daph_learning.evaluation.verification`; it adds new verifier
implementations rather than mutating the existing ones.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .verification import (
    NumericVerifier,
    OutcomeVerifier,
    VerificationResult,
    VerificationStatus,
)

# Canonical schema labels a task may declare via ``output_schema``.
SCHEMA_INTEGER = "integer"
SCHEMA_EXACT = "exact"
SCHEMA_REFERENCE = "reference"
SCHEMA_STRUCTURED = "structured"
SCHEMA_FREE_FORM = "free_form"
KNOWN_SCHEMAS = frozenset(
    {SCHEMA_INTEGER, SCHEMA_EXACT, SCHEMA_REFERENCE, SCHEMA_STRUCTURED, SCHEMA_FREE_FORM}
)


@dataclass(frozen=True)
class ExactMatchVerifier:
    """Exact string equality against ``expected`` (after whitespace trim)."""

    expected_field: str = "expected"
    case_sensitive: bool = True

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
                verifier=type(self).__name__, expected=expected,
                reason="backend execution failed",
            )
        if expected is None:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__,
                reason=f"task has no {self.expected_field!r} value",
            )
        if output is None:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__, expected=expected,
                reason="backend produced no output",
            )
        exp = str(expected)
        obs = str(output)
        if not self.case_sensitive:
            exp, obs = exp.casefold(), obs.casefold()
        observed = obs.strip()
        expected_str = exp.strip()
        status = (
            VerificationStatus.VERIFIED_CORRECT
            if observed == expected_str
            else VerificationStatus.VERIFIED_INCORRECT
        )
        return VerificationResult(
            status=status, verifier=type(self).__name__,
            expected=expected, observed=observed,
        )


_WS_RE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    return _WS_RE.sub(" ", value.casefold()).strip()


def _token_subsequence_match(needle: list[str], haystack: list[str]) -> bool:
    """Return True if ``needle`` appears as a contiguous subsequence in ``haystack``.

    Trailing punctuation is stripped from each token so that ``"paris"``
    matches ``"paris."`` while ``"yes"`` does not match ``"yesterday"``.
    """
    _STRIP = ".,;:!?\"'()[]{}"
    needle_s = [t.rstrip(_STRIP) for t in needle]
    haystack_s = [t.rstrip(_STRIP) for t in haystack]
    if not needle_s:
        return True
    if len(needle_s) > len(haystack_s):
        return False
    for i in range(len(haystack_s) - len(needle_s) + 1):
        if haystack_s[i:i + len(needle_s)] == needle_s:
            return True
    return False


@dataclass(frozen=True)
class ReferenceAnswerVerifier:
    """Reference-answer match for free-form / knowledge tasks.

    Verifies by normalized containment: the reference answer must appear as a
    whole normalized token sequence in the output, OR the normalized output
    equals the normalized reference. This is a conservative judge; a real
    LLM-judge verifier can be supplied via :class:`FreeFormVerifier` for
    higher-fidelity free-form scoring.
    """

    reference_field: str = "reference_answer"

    def verify(
        self,
        task: Mapping[str, Any],
        output: Any | None,
        *,
        execution_succeeded: bool = True,
    ) -> VerificationResult:
        reference = task.get(self.reference_field)
        if not execution_succeeded:
            return VerificationResult(
                status=VerificationStatus.EXECUTION_FAILED,
                verifier=type(self).__name__, expected=reference,
                reason="backend execution failed",
            )
        if reference is None:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__,
                reason=f"task has no {self.reference_field!r} value",
            )
        if output is None:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__, expected=reference,
                reason="backend produced no output",
            )
        ref = _normalize_text(str(reference))
        obs = _normalize_text(str(output))
        if not ref:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__, expected=reference,
                reason="empty reference answer",
            )
        ref_tokens = ref.split()
        obs_tokens = obs.split()
        correct = (obs == ref) or _token_subsequence_match(ref_tokens, obs_tokens)
        return VerificationResult(
            status=(
                VerificationStatus.VERIFIED_CORRECT
                if correct
                else VerificationStatus.VERIFIED_INCORRECT
            ),
            verifier=type(self).__name__, expected=reference, observed=str(output),
        )


@dataclass(frozen=True)
class StructuredSchemaVerifier:
    """Validate a structured (JSON) output against a declared field schema.

    The task carries ``output_schema_spec``: a mapping of field name ->
    Python type (e.g. ``{"answer": int, "confidence": float}``). The output
    must parse as JSON and contain every required field with the declared
    type. Correctness of *values* is not judged here — pair with a
    :class:`CompositeVerifier` including a value verifier for that.
    """

    schema_field: str = "output_schema_spec"

    def verify(
        self,
        task: Mapping[str, Any],
        output: Any | None,
        *,
        execution_succeeded: bool = True,
    ) -> VerificationResult:
        spec = task.get(self.schema_field)
        if not execution_succeeded:
            return VerificationResult(
                status=VerificationStatus.EXECUTION_FAILED,
                verifier=type(self).__name__, expected=spec,
                reason="backend execution failed",
            )
        if not isinstance(spec, Mapping) or not spec:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__,
                reason=f"task has no {self.schema_field!r} mapping",
            )
        if output is None:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__, expected=spec,
                reason="backend produced no output",
            )
        try:
            parsed = json.loads(str(output))
        except (json.JSONDecodeError, TypeError) as exc:
            return VerificationResult(
                status=VerificationStatus.INVALID_OUTPUT,
                verifier=type(self).__name__, expected=spec, observed=output,
                reason=f"output is not valid JSON: {exc}",
            )
        if not isinstance(parsed, dict):
            return VerificationResult(
                status=VerificationStatus.INVALID_OUTPUT,
                verifier=type(self).__name__, expected=spec, observed=output,
                reason="output JSON is not an object",
            )
        missing = [k for k in spec if k not in parsed]
        if missing:
            return VerificationResult(
                status=VerificationStatus.INVALID_OUTPUT,
                verifier=type(self).__name__, expected=spec, observed=output,
                reason=f"missing required fields: {missing}",
            )
        type_errors = []
        for field_name, expected_type in spec.items():
            value = parsed[field_name]
            # bool is a subclass of int; reject bool where int is expected.
            if expected_type is int and isinstance(value, bool):
                type_errors.append(f"{field_name}: expected int, got bool")
            elif not isinstance(value, expected_type):
                type_errors.append(
                    f"{field_name}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
        if type_errors:
            return VerificationResult(
                status=VerificationStatus.INVALID_OUTPUT,
                verifier=type(self).__name__, expected=spec, observed=output,
                reason=f"type errors: {type_errors}",
            )
        return VerificationResult(
            status=VerificationStatus.VERIFIED_CORRECT,
            verifier=type(self).__name__, expected=spec, observed=parsed,
        )


@dataclass(frozen=True)
class FreeFormVerifier:
    """Free-form verifier delegating to an external judge callable.

    Free-form tasks (essays, explanations) require a separate judge; the
    router must not be its own verifier. Without a judge callable the result
    is ``UNVERIFIABLE`` — a free-form task is **never** credited by guessing.
    Supply ``judge`` as ``judge(task, output) -> bool | None``.
    """

    judge: Callable[[Mapping[str, Any], Any | None], bool | None] | None = None

    def verify(
        self,
        task: Mapping[str, Any],
        output: Any | None,
        *,
        execution_succeeded: bool = True,
    ) -> VerificationResult:
        if not execution_succeeded:
            return VerificationResult(
                status=VerificationStatus.EXECUTION_FAILED,
                verifier=type(self).__name__,
                reason="backend execution failed",
            )
        if self.judge is None:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__,
                reason="no external judge supplied; free-form tasks are not "
                       "credited by guessing",
            )
        if output is None:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__,
                reason="backend produced no output",
            )
        try:
            verdict = self.judge(task, output)
        except Exception as exc:  # noqa: BLE001 - judge contract failure
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__, observed=output,
                reason=f"judge raised {type(exc).__name__}: {exc}",
            )
        if verdict is None:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__, observed=output,
                reason="judge abstained",
            )
        return VerificationResult(
            status=(
                VerificationStatus.VERIFIED_CORRECT
                if verdict
                else VerificationStatus.VERIFIED_INCORRECT
            ),
            verifier=type(self).__name__, observed=output,
        )


@dataclass(frozen=True)
class CompositeVerifier:
    """Apply verifiers in order; the first definitive result wins.

    A "definitive" result is ``VERIFIED_CORRECT`` or ``VERIFIED_INCORRECT``.
    If every verifier returns a non-definitive status, the composite returns
    ``UNVERIFIABLE``. This lets a task require both a schema check and a
    value check.
    """

    verifiers: Sequence[OutcomeVerifier] = field(default_factory=tuple)

    def verify(
        self,
        task: Mapping[str, Any],
        output: Any | None,
        *,
        execution_succeeded: bool = True,
    ) -> VerificationResult:
        if not execution_succeeded:
            return VerificationResult(
                status=VerificationStatus.EXECUTION_FAILED,
                verifier=type(self).__name__,
                reason="backend execution failed",
            )
        last_unverifiable: VerificationResult | None = None
        for v in self.verifiers:
            result = v.verify(task, output, execution_succeeded=True)
            if result.status in {
                VerificationStatus.VERIFIED_CORRECT,
                VerificationStatus.VERIFIED_INCORRECT,
            }:
                # Replace the verifier name so the composite is auditable,
                # but preserve the underlying verdict.
                return VerificationResult(
                    status=result.status,
                    verifier=f"CompositeVerifier({result.verifier})",
                    expected=result.expected, observed=result.observed,
                    reason=result.reason,
                )
            if result.status is VerificationStatus.UNVERIFIABLE:
                last_unverifiable = result
        if last_unverifiable is not None:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier=type(self).__name__,
                reason=f"all verifiers unverifiable; last={last_unverifiable.reason}",
            )
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            verifier=type(self).__name__,
            reason="no verifiers configured",
        )


def select_verifier_for_task(task: Mapping[str, Any]) -> OutcomeVerifier:
    """Dispatch to the schema-appropriate verifier for a task.

    The task's ``output_schema`` field selects the verifier. Unknown schemas
    fall back to :class:`NumericVerifier` only when the task has an integer
    ``expected``; otherwise an :class:`ExactMatchVerifier` is used so a
    non-integer expected value is still checked rather than silently passed.
    The router is never selected as the verifier.
    """
    schema = task.get("output_schema", SCHEMA_INTEGER)
    if schema == SCHEMA_INTEGER:
        return NumericVerifier()
    if schema == SCHEMA_EXACT:
        return ExactMatchVerifier()
    if schema == SCHEMA_REFERENCE:
        return ReferenceAnswerVerifier()
    if schema == SCHEMA_STRUCTURED:
        return StructuredSchemaVerifier()
    if schema == SCHEMA_FREE_FORM:
        return FreeFormVerifier()
    # Unknown schema: fail closed toward a real check, never toward credit.
    expected = task.get("expected")
    if isinstance(expected, int):
        return NumericVerifier()
    return ExactMatchVerifier()


__all__ = [
    "CompositeVerifier",
    "ExactMatchVerifier",
    "FreeFormVerifier",
    "KNOWN_SCHEMAS",
    "NumericVerifier",
    "ReferenceAnswerVerifier",
    "SCHEMA_EXACT",
    "SCHEMA_FREE_FORM",
    "SCHEMA_INTEGER",
    "SCHEMA_REFERENCE",
    "SCHEMA_STRUCTURED",
    "StructuredSchemaVerifier",
    "select_verifier_for_task",
]
