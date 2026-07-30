"""v0.3.10.3.2-alpha — Section 6-8, 13 / G13: benchmark arithmetic
correctness.

Verifies that expected answers can always be recomputed from the
prompt/source fields, and that the specific bugs (E equal products,
C fractional half, F percentage truncation) are fixed.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from daph_learning.data.crossover_benchmark import (
    generate_crossover_split,
    SUBTYPES,
)


def _recompute_expected(task: dict[str, Any]) -> int | None:
    """Recompute the expected answer from the prompt text only."""
    spec = str(task.get("specification", ""))

    # Subtype E: "Which is larger: {a1}*{b1} or {a2}*{b2}? ..."
    # Check E before A to avoid the A regex matching E's products.
    m = re.search(r'(\d+)\*(\d+)\s+or\s+(\d+)\*(\d+)', spec)
    if m:
        a1, b1, a2, b2 = (int(m.group(i)) for i in range(1, 5))
        return max(a1 * b1, a2 * b2)

    # Subtype F: "A tank has {total} L, loses {loss_pct}%, then gains {gain} L. ..."
    m = re.search(r'tank has\s+(\d+)\s+L,\s+loses\s+(\d+)%,\s+then gains\s+(\d+)\s+L', spec)
    if m:
        total, loss_pct, gain = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return total - (total * loss_pct // 100) + gain

    # Subtype C: "What is {x} minus twice {y}?"
    m = re.search(r'What is\s+(\d+)\s+minus twice\s+(\d+)', spec)
    if m:
        return int(m.group(1)) - 2 * int(m.group(2))
    # "Take {x} and subtract three times {y}."
    m = re.search(r'Take\s+(\d+)\s+and subtract three times\s+(\d+)', spec)
    if m:
        return int(m.group(1)) - 3 * int(m.group(2))
    # "Add {x} and {y}, then double the result."
    m = re.search(r'Add\s+(\d+)\s+and\s+(\d+),\s+then double', spec)
    if m:
        return 2 * (int(m.group(1)) + int(m.group(2)))
    # "What is half of {x} plus {y}?"
    m = re.search(r'half of\s+(\d+)\s+plus\s+(\d+)', spec)
    if m:
        return int(m.group(1)) // 2 + int(m.group(2))

    # Subtype B: "... {a} {noun} with {b} ... total?"
    m = re.search(r'(\d+)\s+\w+\s+with\s+(\d+)', spec)
    if m:
        return int(m.group(1)) * int(m.group(2))

    # Subtype A/D: "{a} mod {modulus}"
    m = re.search(r'(\d+)\s+mod\s+(\d+)', spec)
    if m:
        return int(m.group(1)) % int(m.group(2))

    # Subtype A: "Compute {a} {op} {b}. ..." (with various wrappers)
    # Only match if "Compute" or "compute" appears in the spec.
    if re.search(r'[Cc]ompute', spec):
        m = re.search(r'(-?\d+)\s*([+\-*])\s*(-?\d+)', spec)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            if op == '+': return a + b
            if op == '-': return a - b
            if op == '*': return a * b

    return None


@pytest.mark.parametrize("subtype", SUBTYPES)
def test_expected_matches_recomputed(subtype):
    """Section 6: recomputing expected answer from prompt fields must
    equal the stored expected answer."""
    tasks = generate_crossover_split(
        split="train", n_per_subtype=30, seed=42)
    subtype_tasks = [t for t in tasks
                     if t["metadata"]["subtype"] == subtype]
    assert len(subtype_tasks) >= 10
    for t in subtype_tasks:
        recomputed = _recompute_expected(t)
        assert recomputed is not None, (
            f"could not recompute expected for task {t['task_id']!r}: "
            f"{t['specification']!r}")
        assert recomputed == t["expected"], (
            f"task {t['task_id']!r}: recomputed={recomputed} != "
            f"stored={t['expected']} — prompt={t['specification']!r}")


def test_subtype_e_no_equal_products():
    """Section 6: Subtype E must never generate left_product ==
    right_product."""
    tasks = generate_crossover_split(
        split="train", n_per_subtype=50, seed=99)
    e_tasks = [t for t in tasks if t["metadata"]["subtype"] == "E"]
    for t in e_tasks:
        spec = t["specification"]
        m = re.search(r'(\d+)\*(\d+)\s+or\s+(\d+)\*(\d+)', spec)
        assert m, f"could not parse E task: {spec!r}"
        a1, b1, a2, b2 = (int(m.group(i)) for i in range(1, 5))
        left, right = a1 * b1, a2 * b2
        assert left != right, (
            f"E task has equal products: {left} == {right}, "
            f"prompt={spec!r}")


def test_subtype_c_half_is_exact():
    """Section 7: 'half of {x}' must use even x so x//2 is exact."""
    tasks = generate_crossover_split(
        split="train", n_per_subtype=50, seed=77)
    c_tasks = [t for t in tasks if t["metadata"]["subtype"] == "C"]
    for t in c_tasks:
        spec = t["specification"]
        if "half of" in spec:
            m = re.search(r'half of\s+(\d+)', spec)
            assert m, f"could not parse half-of in: {spec!r}"
            x = int(m.group(1))
            assert x % 2 == 0, (
                f"C task uses odd value for 'half of {x}' — "
                f"floor division is ambiguous (Section 7)")


def test_subtype_f_percentage_is_exact():
    """Section 8: percentage loss must produce exact integer (no
    truncation)."""
    tasks = generate_crossover_split(
        split="train", n_per_subtype=50, seed=55)
    f_tasks = [t for t in tasks if t["metadata"]["subtype"] == "F"]
    for t in f_tasks:
        spec = t["specification"]
        m = re.search(r'loses\s+(\d+)%', spec)
        assert m, f"could not parse percentage in: {spec!r}"
        loss_pct = int(m.group(1))
        m2 = re.search(r'tank has\s+(\d+)\s+L', spec)
        assert m2
        total = int(m2.group(1))
        assert (total * loss_pct) % 100 == 0, (
            f"F task has truncating percentage: {total}*{loss_pct}%100="
            f"{(total * loss_pct) % 100} != 0 (Section 8)")
