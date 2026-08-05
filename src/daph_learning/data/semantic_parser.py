"""v0.3.10.3.2-alpha — bounded semantic parser for NL arithmetic
(Section 14).

This module implements a deterministic template-based semantic parser
that extracts structured variables and builds a symbolic expression
from natural-language arithmetic prompts. This allows the symbolic
backend to compete on semantic tasks (subtypes B, C, E, F), enabling
true within-subtype crossover.

The parser does NOT use the expected answer — it only uses the prompt
text. If parsing fails, it returns ``None`` (low confidence), and the
LLM backend is preferred.

Supported patterns:
  - "A warehouse has {a} {noun} with {b} {suffix}" → a * b
  - "What is {x} minus twice {y}?" → x - 2*y
  - "Take {x} and subtract three times {y}." → x - 3*y
  - "Add {x} and {y}, then double the result." → 2*(x+y)
  - "What is half of {x} plus {y}?" → x//2 + y
  - "Which is larger: {a1}*{b1} or {a2}*{b2}?" → max(a1*b1, a2*b2)
  - "A tank has {total} L, loses {p}%, then gains {g} L." → total - total*p//100 + g
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParseResult:
    """Result of a semantic parse attempt."""
    expression: str  # symbolic expression string
    confidence: float  # 0.0 to 1.0
    variables: dict[str, int]  # extracted variables
    parse_method: str  # which pattern matched

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "confidence": self.confidence,
            "variables": self.variables,
            "parse_method": self.parse_method,
        }


def parse_semantic_arithmetic(prompt: str) -> ParseResult | None:
    """Section 14: parse a natural-language arithmetic prompt into a
    symbolic expression.

    Returns ``None`` if no pattern matches (parse failure). The
    expression is a string that can be evaluated by the bounded
    symbolic executor.

    The parser does NOT use the expected answer — only the prompt text.
    """
    # Strip wrapper prefixes (e.g., "Question: ", "Word problem: ", etc.)
    # to find the core arithmetic pattern.
    cleaned = re.sub(
        r'^(Question|Problem|Solve|Arithmetic puzzle|'
        r'Held-out question|Reasoning task|Calibration problem|'
        r'Final question|Comparison|Relation check|Size question|'
        r'Held-out comparison|Magnitude check|Calibration comparison|'
        r'Final comparison|Word problem|Scenario|Applied math|'
        r'Held-out scenario|Practical problem|Calibration scenario|'
        r'Final scenario'
        r'):\s*', '', prompt).strip()

    # Pattern B: "... {a} {noun} with {b} ... total?"
    m = re.search(r'(\d+)\s+\w+\s+with\s+(\d+)\s+\w+', cleaned)
    if m and ('total' in cleaned.lower() or 'how many' in cleaned.lower()):
        a, b = int(m.group(1)), int(m.group(2))
        return ParseResult(
            expression=f"{a} * {b}",
            confidence=0.95,
            variables={"a": a, "b": b},
            parse_method="semantic_extraction_multiplication",
        )

    # Pattern C: "What is {x} minus twice {y}?"
    m = re.search(r'What is\s+(\d+)\s+minus twice\s+(\d+)', cleaned)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return ParseResult(
            expression=f"{x} - 2 * {y}",
            confidence=0.90,
            variables={"x": x, "y": y},
            parse_method="semantic_minus_twice",
        )

    # Pattern C: "Take {x} and subtract three times {y}."
    m = re.search(r'Take\s+(\d+)\s+and subtract three times\s+(\d+)', cleaned)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return ParseResult(
            expression=f"{x} - 3 * {y}",
            confidence=0.90,
            variables={"x": x, "y": y},
            parse_method="semantic_subtract_three_times",
        )

    # Pattern C: "Add {x} and {y}, then double the result."
    m = re.search(r'Add\s+(\d+)\s+and\s+(\d+),\s+then double', cleaned)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return ParseResult(
            expression=f"2 * ({x} + {y})",
            confidence=0.90,
            variables={"x": x, "y": y},
            parse_method="semantic_add_double",
        )

    # Pattern C: "What is half of {x} plus {y}?"
    m = re.search(r'half of\s+(\d+)\s+plus\s+(\d+)', cleaned)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return ParseResult(
            expression=f"{x} // 2 + {y}",
            confidence=0.85,
            variables={"x": x, "y": y},
            parse_method="semantic_half_plus",
        )

    # Pattern E: "Which is larger: {a1}*{b1} or {a2}*{b2}?"
    m = re.search(r'(\d+)\*(\d+)\s+or\s+(\d+)\*(\d+)', cleaned)
    if m:
        a1, b1, a2, b2 = (int(m.group(i)) for i in range(1, 5))
        left, right = a1 * b1, a2 * b2
        winner = left if left > right else right
        return ParseResult(
            expression=str(winner),  # pre-computed: comparison result
            confidence=0.95,
            variables={"a1": a1, "b1": b1, "a2": a2, "b2": b2,
                       "left": left, "right": right, "winner": winner},
            parse_method="semantic_comparison",
        )

    # Pattern F: "A tank has {total} L, loses {p}%, then gains {g} L."
    m = re.search(
        r'tank has\s+(\d+)\s+L,\s+loses\s+(\d+)%,\s+then gains\s+(\d+)\s+L',
        cleaned)
    if m:
        total, loss_pct, gain = (int(m.group(i)) for i in range(1, 4))
        return ParseResult(
            expression=f"{total} - {total} * {loss_pct} // 100 + {gain}",
            confidence=0.90,
            variables={"total": total, "loss_pct": loss_pct, "gain": gain},
            parse_method="semantic_tank_percentage",
        )

    # No pattern matched.
    return None


def parse_task(task: dict[str, Any]) -> ParseResult | None:
    """Parse a crossover task's specification into a symbolic
    expression. Convenience wrapper around
    :func:`parse_semantic_arithmetic`.
    """
    spec = str(task.get("specification", ""))
    return parse_semantic_arithmetic(spec)


__all__ = [
    "ParseResult",
    "parse_semantic_arithmetic",
    "parse_task",
]
