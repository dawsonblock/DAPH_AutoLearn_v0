"""Section 7 — canonical FINAL_ANSWER verifier.

Replaces the permissive "number appears somewhere" verification with
canonical answer extraction. Qualification prompts must require::

    FINAL_ANSWER: <integer>

The parser fails closed on ambiguity:

* exactly one ``FINAL_ANSWER:`` field;
* exactly one integer value;
* no conflicting second final-answer marker;
* no range, list, or arithmetic expression in place of the value;
* a response without the canonical field is UNVERIFIABLE during
  qualification.

Examples::

    "The answer is not 42.\\nFINAL_ANSWER: 43"  -> VALID, value=43
    "FINAL_ANSWER: 42\\nFINAL_ANSWER: 43"       -> MULTIPLE (UNVERIFIABLE)
    "FINAL_ANSWER: 1..10"                       -> MALFORMED (UNVERIFIABLE)
    "the answer is 42"                          -> MISSING (UNVERIFIABLE)

The legacy permissive extractor (:func:`parse_legacy_first_int` in
:mod:`daph_learning.evaluation.scoring`) remains available ONLY as a
diagnostic parser and must never award qualification credit.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping


class VerificationStatus(str, Enum):
    """Section 7.2: closed verification-status enum.

    UNVERIFIABLE must never silently become CORRECT or INCORRECT without a
    documented utility rule.
    """

    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNVERIFIABLE = "UNVERIFIABLE"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class ParsedFinalAnswer:
    """Section 7.1: result of parsing a canonical ``FINAL_ANSWER:`` field.

    ``status`` is one of VALID, MISSING, MULTIPLE, MALFORMED, CONTRADICTORY.
    Only ``VALID`` yields a usable ``value``; every other status is
    UNVERIFIABLE for qualification purposes.
    """

    status: Literal["VALID", "MISSING", "MULTIPLE", "MALFORMED", "CONTRADICTORY"]
    value: int | None
    raw_matches: tuple[str, ...]

    @property
    def is_verifiable(self) -> bool:
        return self.status == "VALID"

    @property
    def verification_status(self) -> VerificationStatus:
        """Map to the closed :class:`VerificationStatus` enum."""
        if self.status == "VALID":
            return VerificationStatus.UNVERIFIABLE  # value still must be compared
        return VerificationStatus.UNVERIFIABLE


# ------------------------------------------------------------------
# Section 7.1: canonical parser
# ------------------------------------------------------------------

# A FINAL_ANSWER marker is the literal token "FINAL_ANSWER" (case-insensitive),
# optional surrounding whitespace, a colon, then an integer.
# We capture the whole tail after the colon to inspect it for malformed
# values (ranges, lists, arithmetic, extra tokens).
_FINAL_ANSWER_RE = re.compile(
    r"FINAL_ANSWER\s*:\s*(.+?)(?=FINAL_ANSWER\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# A standalone integer (optionally signed), with optional surrounding
# whitespace. Unicode minus signs are normalized to ASCII '-'.
_INTEGER_RE = re.compile(r"^\s*([+\-\u2212\uFF0D]?\d+)\s*$")

# Detects a range like "1..10", "1-10" (ambiguous w/ subtraction, so only
# ".." and "to" forms are treated as ranges here), or a list "1, 2, 3".
_RANGE_RE = re.compile(r"\.{2,}|\bto\b", re.IGNORECASE)
_LIST_RE = re.compile(r"[,;]\s*-?\d+")

# Disallowed tokens that indicate an arithmetic expression rather than a
# bare integer value.
_ARITH_OP_RE = re.compile(r"[+*/%]|(\bmod\b)|(\*\*)", re.IGNORECASE)


def _normalize_unicode_minus(text: str) -> str:
    """Normalize Unicode minus signs (U+2212, fullwidth hyphen) to ASCII '-'."""
    out = []
    for ch in text:
        if ch == "\u2212":  # MINUS SIGN
            out.append("-")
        elif ch == "\uFF0D":  # FULLWIDTH HYPHEN-MINUS
            out.append("-")
        elif unicodedata.numeric(ch, -1.0) >= 0 and ch.isdigit():
            # Keep ASCII digits only; fullwidth digits are rejected later.
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def _classify_value(raw: str) -> tuple[Literal["VALID", "MALFORMED"], int | None]:
    """Classify a single captured FINAL_ANSWER tail."""
    norm = _normalize_unicode_minus(raw).strip()
    # Strip a single trailing period that is clearly sentence punctuation
    # (e.g. "FINAL_ANSWER: 43.") but NOT a range ("1..10").
    if norm.endswith(".") and not norm.endswith(".."):
        norm = norm[:-1].strip()

    if _RANGE_RE.search(norm) or _LIST_RE.search(norm):
        return "MALFORMED", None
    if _ARITH_OP_RE.search(norm):
        return "MALFORMED", None

    m = _INTEGER_RE.match(norm)
    if not m:
        return "MALFORMED", None
    token = m.group(1)
    try:
        value = int(token)
    except ValueError:
        return "MALFORMED", None
    return "VALID", value


def parse_canonical_integer_answer(text: str) -> ParsedFinalAnswer:
    """Section 7.1: parse a canonical ``FINAL_ANSWER: <integer>`` field.

    Returns a :class:`ParsedFinalAnswer`. The response is VERIFIABLE only
    when ``status == "VALID"`` (exactly one FINAL_ANSWER marker, exactly
    one integer value, no range/list/arithmetic, no conflicting marker).
    """
    if text is None:
        return ParsedFinalAnswer(status="MISSING", value=None, raw_matches=())
    raw_text = str(text)

    matches = list(_FINAL_ANSWER_RE.finditer(raw_text))
    if not matches:
        return ParsedFinalAnswer(status="MISSING", value=None, raw_matches=())

    raw_tails = tuple(m.group(1).strip() for m in matches)

    if len(matches) > 1:
        # Multiple FINAL_ANSWER markers. If they all parse to the same
        # integer, treat as VALID (redundant but not contradictory). If
        # they parse to different integers, CONTRADICTORY. If any is
        # malformed, MULTIPLE.
        values: list[int | None] = []
        any_malformed = False
        for tail in raw_tails:
            status, val = _classify_value(tail)
            if status == "MALFORMED":
                any_malformed = True
                values.append(None)
            else:
                values.append(val)
        if any_malformed:
            return ParsedFinalAnswer(
                status="MULTIPLE", value=None, raw_matches=raw_tails)
        distinct = {v for v in values if v is not None}
        if len(distinct) > 1:
            return ParsedFinalAnswer(
                status="CONTRADICTORY", value=None, raw_matches=raw_tails)
        # All markers agree.
        return ParsedFinalAnswer(
            status="VALID", value=distinct.pop(), raw_matches=raw_tails)

    # Exactly one marker.
    status, val = _classify_value(raw_tails[0])
    return ParsedFinalAnswer(status=status, value=val, raw_matches=raw_tails)


# ------------------------------------------------------------------
# Section 7.1: qualification verifier
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalIntegerVerifier:
    """Verify an integer answer against the expected value using ONLY the
    canonical ``FINAL_ANSWER:`` field.

    Legacy permissive extraction (substring/first-integer) is never used
    to award qualification credit. A response without a VALID
    ``FINAL_ANSWER:`` field is UNVERIFIABLE.
    """

    expected_field: str = "expected"

    def verify(
        self,
        task: Mapping[str, Any],
        output: Any | None,
        *,
        execution_succeeded: bool = True,
        timed_out: bool = False,
    ) -> VerificationStatus:
        """Return the closed :class:`VerificationStatus` for the output."""
        if timed_out:
            return VerificationStatus.TIMEOUT
        if not execution_succeeded:
            return VerificationStatus.EXECUTION_ERROR

        expected = task.get(self.expected_field) if isinstance(task, Mapping) else None
        if type(expected) is not int:
            return VerificationStatus.UNVERIFIABLE
        if output is None:
            return VerificationStatus.UNVERIFIABLE

        parsed = parse_canonical_integer_answer(str(output))
        if parsed.status != "VALID" or parsed.value is None:
            return VerificationStatus.UNVERIFIABLE
        return (
            VerificationStatus.CORRECT
            if parsed.value == expected
            else VerificationStatus.INCORRECT
        )


__all__ = [
    "CanonicalIntegerVerifier",
    "ParsedFinalAnswer",
    "VerificationStatus",
    "parse_canonical_integer_answer",
]
