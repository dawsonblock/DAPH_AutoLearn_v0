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
    # E unparseable: "Compare these two products: {a1} times {b1} versus {a2} times {b2}. ..."
    m = re.search(r'(\d+)\s+times\s+(\d+)\s+versus\s+(\d+)\s+times\s+(\d+)', spec)
    if m:
        a1, b1, a2, b2 = (int(m.group(i)) for i in range(1, 5))
        return max(a1 * b1, a2 * b2)

    # Subtype F: "A tank has {total} L, loses {loss_pct}%, then gains {gain} L. ..."
    m = re.search(r'tank has\s+(\d+)\s+L,\s+loses\s+(\d+)%,\s+then gains\s+(\d+)\s+L', spec)
    if m:
        total, loss_pct, gain = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return total - (total * loss_pct // 100) + gain
    # F unparseable: "You start with {total} liters ... {loss_pct}% ... pours in {gain} more liters."
    m = re.search(r'start with\s+(\d+)\s+liters.*?(\d+)%.*?pours in\s+(\d+)', spec)
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
    # C unparseable: "Start with {x}, then take away {y} two times."
    m = re.search(r'Start with\s+(\d+).*?take away\s+(\d+).*?two times', spec)
    if m:
        return int(m.group(1)) - 2 * int(m.group(2))
    # C unparseable: "Begin at {x} and remove {y} exactly three times."
    m = re.search(r'Begin at\s+(\d+).*?remove\s+(\d+).*?three times', spec)
    if m:
        return int(m.group(1)) - 3 * int(m.group(2))
    # C unparseable: "Combine {x} and {y}, then multiply the sum by two."
    m = re.search(r'Combine\s+(\d+)\s+and\s+(\d+).*?multiply.*?two', spec)
    if m:
        return 2 * (int(m.group(1)) + int(m.group(2)))
    # C unparseable: "Divide {x} by two and then add {y}."
    m = re.search(r'Divide\s+(\d+)\s+by two.*?add\s+(\d+)', spec)
    if m:
        return int(m.group(1)) // 2 + int(m.group(2))

    # Subtype B: "... {a} {noun} with {b} ... total?"
    m = re.search(r'(\d+)\s+\w+\s+with\s+(\d+)', spec)
    if m:
        return int(m.group(1)) * int(m.group(2))
    # B unparseable: various phrasings with "each" and small numbers
    # "There are {a} shelves ... each one holds {b} books"
    m = re.search(r'(\d+)\s+shelves.*?each.*?holds\s+(\d+)', spec)
    if m:
        return int(m.group(1)) * int(m.group(2))
    # "A baker made {a} batches ... {b} cookies in each batch"
    m = re.search(r'(\d+)\s+batches.*?(\d+)\s+cookies.*?each', spec)
    if m:
        return int(m.group(1)) * int(m.group(2))
    # "Each classroom has {a} desks ... {b} classrooms"
    m = re.search(r'(\d+)\s+desks.*?(\d+)\s+classrooms', spec)
    if m:
        return int(m.group(1)) * int(m.group(2))
    # "planted {a} rows ... {b} flowers in each row"
    m = re.search(r'(\d+)\s+rows.*?(\d+)\s+flowers.*?each', spec)
    if m:
        return int(m.group(1)) * int(m.group(2))
    # B unparseable: generic "{a} ... each with/holding {b} ..."
    m = re.search(r'(\d+)\s+\w+.*?each\s+\w+.*?(\d+)\s+', spec)
    if m:
        return int(m.group(1)) * int(m.group(2))
    # B unparseable: "{a} ... each containing {b} ..."
    m = re.search(r'(\d+)\s+\w+.*?each\s+\w+\s+(\d+)\s+', spec)
    if m:
        return int(m.group(1)) * int(m.group(2))
    # B unparseable: "each with {b} ..." after "{a} ..."
    m = re.search(r'(\d+)\s+\w+.*?,\s*each\s+\w+\s+(\d+)\s', spec)
    if m:
        return int(m.group(1)) * int(m.group(2))

    # Subtype A/D: "{a} mod {modulus}"
    m = re.search(r'(\d+)\s+mod\s+(\d+)', spec)
    if m:
        return int(m.group(1)) % int(m.group(2))
    # D unparseable: "What is the remainder when you divide {a} by {modulus}?"
    m = re.search(r'remainder when you divide\s+(\d+)\s+by\s+(\d+)', spec)
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

    # A NL variant: "If you have {a} apples and a friend gives you {b} more..."
    m = re.search(r'have\s+(\d+)\s+apples.*?gives you\s+(\d+)\s+more', spec)
    if m:
        return int(m.group(1)) + int(m.group(2))
    # A NL variant: "You had {a} marbles but lost {b} of them..."
    m = re.search(r'had\s+(\d+)\s+marbles.*?lost\s+(\d+)', spec)
    if m:
        return int(m.group(1)) - int(m.group(2))

    # Subtype G: unit conversion + arithmetic
    # "Convert {value} {unit_from} to {unit_to}, then add {addend} {unit_to}."
    # The converted value is already in the structured inputs, but for
    # test recompute we parse from spec.
    unit_factors = {"cm": 100, "m": 1000, "g": 1000, "minutes": 60, "mL": 1000}
    m = re.search(r'Convert\s+(\d+)\s+(\w+)\s+to\s+(\w+).*?add\s+(\d+)', spec)
    if m:
        value = int(m.group(1))
        unit_to = m.group(3)
        addend = int(m.group(4))
        factor = unit_factors.get(unit_to, 1)
        return value * factor + addend
    # G unparseable: "A runner goes {value} {unit_from}. Express this in {unit_to}..."
    m = re.search(r'goes\s+(\d+)\s+(\w+).*?in\s+(\w+).*?add\s+(\d+)', spec)
    if m:
        value = int(m.group(1))
        unit_to = m.group(3)
        addend = int(m.group(4))
        factor = unit_factors.get(unit_to, 1)
        return value * factor + addend

    # Subtype H: gcd/lcm
    import math
    m = re.search(r'gcd\((\d+),\s*(\d+)\)', spec)
    if m:
        return math.gcd(int(m.group(1)), int(m.group(2)))
    m = re.search(r'lcm\((\d+),\s*(\d+)\)', spec)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return a * b // math.gcd(a, b) if a and b else 0
    # H unparseable: "greatest common divisor of {a} and {b}"
    m = re.search(r'greatest common divisor of\s+(\d+)\s+and\s+(\d+)', spec)
    if m:
        return math.gcd(int(m.group(1)), int(m.group(2)))
    # H unparseable: "least common multiple of {a} and {b}"
    m = re.search(r'least common multiple of\s+(\d+)\s+and\s+(\d+)', spec)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return a * b // math.gcd(a, b) if a and b else 0

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
        # Try both parseable and unparseable patterns.
        m = re.search(r'(\d+)\*(\d+)\s+or\s+(\d+)\*(\d+)', spec)
        if not m:
            m = re.search(r'(\d+)\s+times\s+(\d+)\s+versus\s+(\d+)\s+times\s+(\d+)', spec)
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
        # Try parseable pattern: "loses {loss_pct}%"
        m = re.search(r'loses\s+(\d+)%', spec)
        if not m:
            # Try unparseable pattern: "{loss_pct}% of it evaporates"
            m = re.search(r'(\d+)%\s+of it evaporates', spec)
        assert m, f"could not parse percentage in: {spec!r}"
        loss_pct = int(m.group(1))
        # Try parseable: "tank has {total} L"
        m2 = re.search(r'tank has\s+(\d+)\s+L', spec)
        if not m2:
            # Try unparseable: "start with {total} liters"
            m2 = re.search(r'start with\s+(\d+)\s+liters', spec)
        assert m2, f"could not parse total in: {spec!r}"
        total = int(m2.group(1))
        assert (total * loss_pct) % 100 == 0, (
            f"F task has truncating percentage: {total}*{loss_pct}%100="
            f"{(total * loss_pct) % 100} != 0 (Section 8)")
