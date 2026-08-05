"""v0.3.10.3.2-alpha — strong hand-router baseline (Section 13).

A reasonable rule-based router using ONLY information available at
decision time (task fields, capability_ids, specification text). This
is NOT a deliberately weak baseline — it encodes the obvious
routing heuristics a competent engineer would write:

  * structured expression available (capability_ids include a supported
    symbolic capability AND structured inputs present) -> symbolic
  * unsupported syntax / no structured capability -> LLM
  * natural-language semantic extraction required -> LLM for small
    operands, symbolic for large operands (where LLM arithmetic is
    unreliable)
  * exact modular expression with structured inputs -> symbolic

Section 14: the hand router now considers operand magnitude in the
specification text. For NL tasks with large numbers, symbolic is
preferred (the semantic parser can extract and compute exactly, while
LLM arithmetic is unreliable on large numbers).

AutoLearn is compared against this hand router as Tier 4 of the
scientific success ladder (Section 41).
"""

from __future__ import annotations

import re
from typing import Any, Mapping


_SUPPORTED_SYMBOLIC_CAPS = {"integer_arithmetic", "modular_multiplication"}

# Threshold for "large" operands — above this, LLM arithmetic is
# unreliable and symbolic is preferred (Section 14).
_LARGE_OPERAND_THRESHOLD = 1000


def _has_structured_inputs(task: Mapping[str, Any]) -> bool:
    inputs = task.get("inputs") or {}
    return bool(inputs) and isinstance(inputs, Mapping)


def _max_operand_in_spec(spec: str) -> int:
    """Extract the maximum integer operand from the specification text."""
    numbers = [int(n) for n in re.findall(r'-?\d+', spec)]
    return max(abs(n) for n in numbers) if numbers else 0


def hand_router_route(task: Mapping[str, Any]) -> str:
    """Return the hand-router route for a task: ``"symbolic"`` or ``"llm"``.

    Uses only decision-time information. Never reads ``expected``,
    ``route_label``, ``best_backend``, or any post-execution field.
    """
    caps = set(task.get("capability_ids", []) or [])
    has_structured = _has_structured_inputs(task)
    spec = str(task.get("specification", ""))

    # Rule 1: structured expression available -> symbolic.
    if (caps & _SUPPORTED_SYMBOLIC_CAPS) and has_structured:
        inputs = task.get("inputs") or {}
        # Need the minimal fields for the typed action.
        if "integer_arithmetic" in caps and {"a", "b", "op"} <= inputs.keys():
            return "symbolic"
        if "modular_multiplication" in caps and {"a", "b"} <= inputs.keys() and (
                "modulus" in inputs or "mod" in inputs or "m" in inputs):
            return "symbolic"
    # Rule 2: exact modular expression with structured inputs -> symbolic.
    if has_structured and (task.get("inputs", {}) or {}).get("op") == "%":
        return "symbolic"
    # Rule 3: NL task with large operands -> symbolic (semantic parser
    # can handle it, LLM arithmetic is unreliable on large numbers).
    if not has_structured:
        max_operand = _max_operand_in_spec(spec)
        if max_operand > _LARGE_OPERAND_THRESHOLD:
            return "symbolic"
    # Rule 4: NL task with small operands -> LLM (cheaper, correct).
    return "llm"


def hand_router_routes(tasks: list[Mapping[str, Any]]) -> list[str]:
    """Route a batch of tasks with the hand router."""
    return [hand_router_route(t) for t in tasks]


__all__ = ["hand_router_route", "hand_router_routes"]
