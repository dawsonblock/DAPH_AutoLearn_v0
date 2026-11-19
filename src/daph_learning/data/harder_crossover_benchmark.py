"""v0.3.10.6-alpha — harder crossover benchmark (magnitude-decoupled).

The original crossover_benchmark.py has a magnitude leak: large operands
→ symbolic wins (LLM arithmetic errors), small operands → LLM wins
(both correct, LLM cheaper). This means the surface feature
``max_operand_magnitude`` perfectly predicts routing, and hidden states
add zero value.

This module generates a **harder** benchmark where:

  1. **Operand magnitude is decoupled from routing.** Both parseable
     and unparseable variants use the SAME number ranges. The routing
     decision depends on whether the semantic parser can match the
     phrasing, NOT on number size.

  2. **Phrasing complexity determines parsability.** Parseable variants
     use phrasings that match the semantic parser's regex patterns.
     Unparseable variants use different phrasings that the parser
     cannot match — but the LLM can understand semantically.

  3. **Crossover is within-subtype and within-magnitude.** For each
     subtype, there are tasks with identical magnitude ranges but
     different phrasing, creating genuine routing ambiguity that
     surface features cannot resolve.

  4. **Hidden states encode semantic structure.** The LLM's hidden
     representations capture whether a prompt is structured (parseable)
     or natural (unparseable), which is the routing signal. Surface
     features (magnitude, length, keywords) cannot capture this.

Subtypes (same family "structured_math"):

  A. Direct arithmetic — parseable "Compute a op b" vs unparseable
     "If you have a items and receive b more..."
  B. Semantic extraction — parseable "A warehouse has a X with b Y"
     vs unparseable "There are a shelves each holding b books..."
  C. Semantic interpretation — parseable "What is x minus twice y?"
     vs unparseable "Start with x, then take away y two times..."
  D. Modular arithmetic — parseable "a mod b" vs unparseable
     "What is the remainder when you divide a by b?"
  E. Comparison — parseable "Which is larger: a*b or c*d?" vs
     unparseable "Compare a times b versus c times d..."
  F. Multi-step NL — parseable "A tank has t L, loses p%, gains g L"
     vs unparseable "You start with t liters, p% evaporates, then..."
  G. Unit conversion — parseable "Convert a X to Y, then add b" vs
     unparseable "A runner goes a X. Express in Y, then add b more..."
  H. Number theory — parseable "gcd(a, b)" vs unparseable
     "What is the greatest common divisor of a and b?"

Key change: ALL variants use the SAME number ranges. The 7B model
should be able to do arithmetic correctly on more of these, making
the routing decision depend on semantic structure rather than
arithmetic difficulty.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Mapping

GENERATOR_VERSION = "v0.3.10.6-harder-crossover"

FAMILY_ID = "structured_math"

SUBTYPES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")

SUBTYPE_DESCRIPTIONS: dict[str, str] = {
    "A": "direct exact arithmetic (magnitude-decoupled)",
    "B": "semantic extraction + exact arithmetic (magnitude-decoupled)",
    "C": "semantic interpretation (magnitude-decoupled)",
    "D": "structured modular arithmetic (magnitude-decoupled)",
    "E": "comparison / relation problem (magnitude-decoupled)",
    "F": "multi-step natural-language arithmetic (magnitude-decoupled)",
    "G": "unit conversion + arithmetic (magnitude-decoupled)",
    "H": "number theory GCD/LCM (magnitude-decoupled)",
}

SPLIT_TEMPLATE_SLOTS: dict[str, tuple[int, ...]] = {
    "train": (0, 1),
    "dev": (2, 3),
    "calibration": (4,),
    "final": (5, 6, 7),
}

FORBIDDEN_METADATA_FIELDS = (
    "best_backend",
    "symbolic_preferred",
    "llm_preferred",
    "route_label",
    "utility_oracle",
    "capability_oracle",
    "accuracy_oracle",
)

# Wording wrappers per slot (8 slots for 4 splits × 2 slots).
# Each slot has MANY wrappers to ensure diversity across groups.
_WRAPPERS = (
    "Compute {body}. Return only the integer.",
    "Please compute {body}. Reply with the exact integer only.",
    "Exact arithmetic request: Compute {body}. Output one integer.",
    "Task: Compute {body}. Give no prose, only the integer.",
    "Calibration wording — Compute {body}. Return the integer result only.",
    "Final wording A — Compute {body}. Respond with one integer.",
    "Final wording B — Evaluate {body}. Return only the integer.",
    "Final wording C — Calculate {body}. Output the integer value only.",
    "Determine the result of {body}. Provide only the integer answer.",
    "Find the value of {body}. Return just the integer.",
    "Solve {body}. Output the integer result with no explanation.",
    "What is {body}? Reply with the integer only.",
    "Calculate the exact value of {body}. Return only the number.",
    "Evaluate the expression {body}. Give the integer answer only.",
    "Compute the following: {body}. Return the integer.",
    "Arithmetic problem: {body}. Answer with the integer only.",
    "Please solve {body}. Provide the integer result.",
    "What is the result when you compute {body}? Return the integer.",
    "Find the answer to {body}. Output only the integer.",
    "Determine {body}. Reply with the integer value only.",
    "Calculate {body}. Give me just the integer.",
    "Evaluate {body} and return the integer result.",
    "Solve the arithmetic expression {body}. Output the integer.",
    "Compute {body} exactly. Return the integer answer.",
    "What does {body} equal? Provide the integer only.",
    "Find the numerical result of {body}. Return the integer.",
    "Calculate the value: {body}. Output only the integer.",
    "Determine the value of {body}. Give the integer answer.",
    "Solve for the result of {body}. Return the integer only.",
    "Compute the arithmetic: {body}. Provide the integer.",
    "What is the numerical answer to {body}? Return the integer.",
    "Evaluate and provide the integer for {body}.",
    "What is the solution to {body}? Return the integer.",
    "Compute the value: {body}. Give the integer.",
    "Find the result: {body}. Output the integer only.",
    "Calculate exactly: {body}. Return just the integer.",
    "Determine the answer: {body}. Provide the integer.",
    "Solve and return the integer: {body}.",
    "What is the computed value of {body}? Return the integer.",
    "Evaluate this: {body}. Give only the integer.",
    "Compute and answer: {body}. Return the integer result.",
    "Find the exact answer: {body}. Output the integer.",
    "Calculate the result: {body}. Provide only the integer.",
    "Determine the numerical result: {body}. Return the integer.",
    "Solve this problem: {body}. Give the integer answer.",
    "What is the answer: {body}? Reply with the integer.",
    "Compute the expression: {body}. Output the integer value.",
    "Find the value: {body}. Return the integer only.",
    "Calculate: {body}. Output just the integer.",
    "Evaluate the following: {body}. Return the integer.",
    "Determine the result: {body}. Give the integer only.",
    "Solve for the value: {body}. Provide the integer.",
    "What is the result of {body}? Return the integer answer.",
    "Compute and return: {body}. Give the integer.",
    "Find the answer: {body}. Reply with the integer only.",
    "Calculate the following: {body}. Return just the integer.",
    "Evaluate: {body}. Output the integer result.",
    "Determine: {body}. Provide the integer answer.",
    "Solve: {body}. Return the integer value.",
    "What is: {body}? Give the integer only.",
    "Compute: {body}. Reply with the integer.",
    "Find: {body}. Output the integer.",
    "Calculate: {body}. Give the integer result.",
    "Evaluate: {body}. Provide the integer.",
    "Determine: {body}. Return the integer.",
    "Solve: {body}. Give the integer.",
    "What is: {body}? Return the integer.",
    "Compute: {body}. Output the integer.",
    "Find: {body}. Return the integer.",
    "Calculate: {body}. Output the integer.",
    "Evaluate: {body}. Give the integer.",
    "Determine: {body}. Output the integer.",
    "Solve: {body}. Return the integer.",
)


def _normalize_template(prompt: str) -> str:
    import re
    normalized = re.sub(r'-?\d+', '{N}', prompt)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def linguistic_template_id(prompt: str) -> str:
    normalized = _normalize_template(prompt)
    h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"ling:{h}"


def _metadata(subtype: str, split: str, template_slot: int, seed: int,
              prompt: str = "") -> dict[str, Any]:
    return {
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "split": split,
        "family_id": FAMILY_ID,
        "subtype": subtype,
        "subtype_description": SUBTYPE_DESCRIPTIONS[subtype],
        "template_id": f"{FAMILY_ID}:{subtype}:template_{template_slot}",
        "linguistic_template_id": linguistic_template_id(prompt),
        "group_id": f"{FAMILY_ID}:{subtype}",
        "template_slot": template_slot,
    }


def _wrap(spec: str, slot: int) -> str:
    return _WRAPPERS[slot % len(_WRAPPERS)].format(body=spec)


# ──────────────────────────────────────────────────────────────────
# Subtype generators — magnitude-decoupled.
#
# Each generator produces two variants with the SAME number range:
#   - parseable (50%): phrasing matches the semantic parser → symbolic wins
#   - unparseable (50%): phrasing doesn't match → LLM wins
#
# The 7B model should handle arithmetic on these ranges correctly,
# so routing depends on PARSING, not arithmetic difficulty.
# ──────────────────────────────────────────────────────────────────


def _gen_a(rng: random.Random, slot: int) -> dict[str, Any]:
    """A. Direct exact arithmetic — magnitude-decoupled.

    Tuned for ~60% LLM success. Addition/subtraction uses larger
    numbers (1K-99K) the 7B handles well; multiplication uses smaller
    numbers (10-99) to keep products computable.
    """
    op = rng.choice(["+", "-", "*"])
    if op == "*":
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
    else:
        a = rng.randint(1000, 99999)
        b = rng.randint(1000, 99999)
    if op == "+":
        expected = a + b
    elif op == "-":
        expected = a - b
    else:
        expected = a * b

    if rng.random() < 0.5:
        # Parseable — symbolic parser handles structured "a op b".
        body = f"{a} {op} {b}"
        return {
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": a, "b": b, "op": op},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        # Unparseable — NL phrasing the parser can't match.
        # Same numbers, but phrased as a word problem.
        # Multiple phrasings for diversity.
        if op == "+":
            phrasings = [
                (f"If you have {a} apples and a friend gives you "
                 f"{b} more, how many apples do you have in total? "
                 f"Return the integer."),
                (f"A library has {a} books. A donation adds {b} more. "
                 f"What is the new total? Return the integer."),
                (f"There are {a} students enrolled. {b} more register. "
                 f"How many students are there now? Return the integer."),
                (f"You owe {a} dollars and pay back {b}. "
                 f"How much do you still owe? Return the integer."),
                (f"A parking lot has {a} cars. {b} more arrive. "
                 f"How many cars are in the lot? Return the integer."),
            ]
            spec = rng.choice(phrasings)
        elif op == "-":
            if a < b:
                a, b = b, a
                expected = a - b
            phrasings = [
                (f"You had {a} marbles but lost {b} of them. "
                 f"How many marbles do you have left? "
                 f"Return the integer."),
                (f"A store had {a} items in stock and sold {b}. "
                 f"How many remain? Return the integer."),
                (f"There were {a} birds on a wire. {b} flew away. "
                 f"How many stayed? Return the integer."),
                (f"You earned {a} points but lost {b} to a penalty. "
                 f"What is your score? Return the integer."),
                (f"A tank holds {a} gallons. {b} are drained. "
                 f"How many gallons remain? Return the integer."),
            ]
            spec = rng.choice(phrasings)
        else:
            phrasings = [
                (f"A store has {a} shelves, each with {b} items. "
                 f"How many items are there in total? "
                 f"Return the integer."),
                (f"A farmer plants {a} rows of {b} corn each. "
                 f"How many corn plants total? Return the integer."),
                (f"There are {a} classrooms with {b} desks each. "
                 f"What is the total number of desks? Return the integer."),
                (f"A choir has {a} sections, each with {b} singers. "
                 f"How many singers total? Return the integer."),
                (f"A bookshelf has {a} shelves with {b} books each. "
                 f"How many books are on the shelf? Return the integer."),
            ]
            spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_b(rng: random.Random, slot: int) -> dict[str, Any]:
    """B. Semantic extraction + exact arithmetic — magnitude-decoupled.

    Tuned for ~60% LLM success: products in 100-10K range that the 7B
    model can often compute correctly.
    """
    a = rng.randint(20, 200)
    b = rng.randint(5, 50)
    expected = a * b

    if rng.random() < 0.5:
        # Parseable — matches "A warehouse has {a} {noun} with {b} {suffix}".
        body = f"A warehouse has {a} crates with {b} units each. How many units total?"
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        # Unparseable — different phrasing the parser can't match.
        # Same numbers, different wording. Multiple phrasings for diversity.
        phrasings = [
            (f"There are {a} shelves in the library, and each one "
             f"holds {b} books. If every shelf is full, what's the "
             f"total number of books? Return the integer."),
            (f"A baker made {a} batches of cookies, putting {b} cookies "
             f"in each batch. How many cookies were baked in total? "
             f"Return the integer."),
            (f"Each classroom has {a} desks. If there are {b} classrooms, "
             f"how many desks are there altogether? Return the integer."),
            (f"A gardener planted {a} rows of flowers with {b} flowers "
             f"in each row. How many flowers were planted? "
             f"Return the integer."),
            (f"A library has {a} display cases, each showing {b} rare "
             f"coins. What is the total number of coins on display? "
             f"Return the integer."),
            (f"A farm has {a} chicken coops, each containing {b} hens. "
             f"How many hens are on the farm in total? "
             f"Return the integer."),
            (f"In a parking lot, {a} vans each have {b} seats. What is "
             f"the total seating capacity across all vans? "
             f"Return the integer."),
            (f"A stamp collection has {a} albums, each with {b} stamps. "
             f"How many stamps are in the collection altogether? "
             f"Return the integer."),
        ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_c(rng: random.Random, slot: int) -> dict[str, Any]:
    """C. Semantic interpretation — magnitude-decoupled.

    Tuned for ~60% LLM success: larger values that make the arithmetic
    harder for the 7B model, creating more symbolic-preferred tasks.
    """
    x = 2 * rng.randint(5000, 50000)  # always even
    y = rng.randint(500, 5000)

    if rng.random() < 0.5:
        # Parseable — matches "What is {x} minus twice {y}?".
        spec = f"What is {x} minus twice {y}?"
        expected = x - 2 * y
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }
    else:
        # Unparseable — same operation, different phrasing.
        phrasings = [
            (f"Start with {x}, then take away {y} two times. "
             f"What is the result? Return the integer."),
            (f"Begin at {x} and remove {y} exactly two times. "
             f"What do you have left? Return the integer."),
            (f"Take {x} and subtract {y} twice. What is the answer? "
             f"Return the integer."),
            (f"You have {x}. You spend {y} dollars twice. "
             f"How much do you have left? Return the integer."),
            (f"From {x}, deduct {y} two separate times. "
             f"What remains? Return the integer."),
        ]
        spec = rng.choice(phrasings)
        expected = x - 2 * y
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_d(rng: random.Random, slot: int) -> dict[str, Any]:
    """D. Modular arithmetic — magnitude-decoupled.

    Tuned for ~60% LLM success: small dividends/moduli the 7B model
    can often compute correctly.
    """
    a = rng.randint(100, 2000)
    modulus = rng.randint(3, 50)
    expected = a % modulus

    if rng.random() < 0.5:
        # Parseable — structured "a mod b".
        body = f"{a} mod {modulus}"
        return {
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": a, "b": modulus, "op": "%"},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        # Unparseable — NL phrasing.
        phrasings = [
            (f"What is the remainder when you divide {a} by "
             f"{modulus}? Return the integer."),
            (f"Divide {a} by {modulus}. What is the remainder? "
             f"Return the integer."),
            (f"How much is left over when {a} is divided by {modulus}? "
             f"Return the integer."),
            (f"If you split {a} into groups of {modulus}, how many "
             f"are left over? Return the integer."),
            (f"What is {a} modulo {modulus}? Return the integer."),
        ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_e(rng: random.Random, slot: int) -> dict[str, Any]:
    """E. Comparison / relation problem — magnitude-decoupled.

    Tuned for ~60% LLM success: small products (100-2500) the 7B model
    can often compare correctly.
    """
    lo, hi = 10, 50
    while True:
        a1, b1 = rng.randint(lo, hi), rng.randint(lo, hi)
        a2, b2 = rng.randint(lo, hi), rng.randint(lo, hi)
        left, right = a1 * b1, a2 * b2
        if left != right:
            break
    expected = max(left, right)

    if rng.random() < 0.5:
        # Parseable — matches "Which is larger: a*b or c*d?".
        spec = (f"Which is larger: {a1}*{b1} or {a2}*{b2}? "
                f"Reply with the larger product as an integer.")
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }
    else:
        # Unparseable — different phrasing.
        phrasings = [
            (f"Compare these two products: {a1} times {b1} versus "
             f"{a2} times {b2}. Which one gives a bigger result? "
             f"Return that product as an integer."),
            (f"Multiply {a1} by {b1}, then multiply {a2} by {b2}. "
             f"Which product is larger? Return it as an integer."),
            (f"Calculate {a1} × {b1} and {a2} × {b2}. "
             f"Which is the bigger result? Return that integer."),
            (f"Two products: {a1} multiplied by {b1}, and {a2} multiplied "
             f"by {b2}. Which is greater? Return the larger product."),
            (f"Find the larger of {a1}*{b1} and {a2}*{b2} by computing "
             f"both. Return the larger product as an integer."),
        ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_f(rng: random.Random, slot: int) -> dict[str, Any]:
    """F. Multi-step NL arithmetic — magnitude-decoupled."""
    while True:
        loss_pct = rng.randint(10, 40)
        total = rng.randint(10000, 999999)
        if (total * loss_pct) % 100 == 0:
            break
    gain = rng.randint(100, 9999)
    loss = total * loss_pct // 100
    after_loss = total - loss
    expected = after_loss + gain

    if rng.random() < 0.5:
        # Parseable — matches "A tank has {total} L, loses {p}%, then gains {g} L."
        spec = (f"A tank has {total} L, loses {loss_pct}%, then gains "
                f"{gain} L. How many litres remain? Return the integer.")
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }
    else:
        # Unparseable — different phrasing.
        phrasings = [
            (f"You start with {total} liters of water in a container. "
             f"First, {loss_pct}% of it evaporates. Then someone pours "
             f"in {gain} more liters. How much water is in the container "
             f"now? Return the integer."),
            (f"A pool contains {total} liters. {loss_pct}% drains out, "
             f"then {gain} liters are added back. What is the final "
             f"volume? Return the integer."),
            (f"You have {total} dollars. You lose {loss_pct}% to taxes, "
             f"then earn {gain} more. How much do you have? Return the integer."),
            (f"A reservoir has {total} gallons. {loss_pct}% evaporates, "
             f"then {gain} gallons flow in. What is the new level? "
             f"Return the integer."),
            (f"Your savings account has {total}. After a {loss_pct}% loss "
             f"and a {gain} deposit, what is the balance? Return the integer."),
        ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_g(rng: random.Random, slot: int) -> dict[str, Any]:
    """G. Unit conversion + arithmetic — magnitude-decoupled."""
    units = (
        ("meters", "cm", 100),
        ("km", "m", 1000),
        ("kg", "g", 1000),
        ("hours", "minutes", 60),
        ("L", "mL", 1000),
    )
    unit_from, unit_to, factor = rng.choice(units)
    value = rng.randint(1000, 99999)
    addend = rng.randint(100, 9999)
    converted = value * factor
    expected = converted + addend

    if rng.random() < 0.5:
        # Parseable — structured "Convert {value} {unit_from} to {unit_to}, then add {addend}".
        body = f"Convert {value} {unit_from} to {unit_to}, then add {addend} {unit_to}."
        return {
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": converted, "b": addend, "op": "+"},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        # Unparseable — NL phrasing.
        phrasings = [
            (f"A runner goes {value} {unit_from}. Express this in {unit_to}, "
             f"then add {addend} more {unit_to}. What is the total in "
             f"{unit_to}? Return the integer."),
            (f"Convert {value} {unit_from} into {unit_to}. Then add {addend} "
             f"{unit_to} to the result. What is the final amount in {unit_to}? "
             f"Return the integer."),
            (f"If {value} {unit_from} are converted to {unit_to} and then "
             f"{addend} {unit_to} are added, what is the total? Return the integer."),
            (f"A measurement of {value} {unit_from} is converted to {unit_to}. "
             f"Then {addend} {unit_to} is appended. What is the sum in {unit_to}? "
             f"Return the integer."),
            (f"Change {value} {unit_from} to {unit_to} units. Then increase "
             f"by {addend} {unit_to}. What is the result? Return the integer."),
        ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_h(rng: random.Random, slot: int) -> dict[str, Any]:
    """H. Number theory (GCD/LCM) — magnitude-decoupled.

    Tuned for ~60% LLM success: small values (50-500) the 7B model
    can often compute GCD/LCM for.
    """
    a = rng.randint(50, 500)
    b = rng.randint(50, 500)
    op = rng.choice(["gcd", "lcm"])
    if op == "gcd":
        expected = math.gcd(a, b)
    else:
        expected = a * b // math.gcd(a, b)

    if rng.random() < 0.5:
        # Parseable — structured "gcd(a, b)" or "lcm(a, b)".
        body = f"{op}({a}, {b})"
        return {
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": a, "b": b, "op": op},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        # Unparseable — NL phrasing.
        if op == "gcd":
            phrasings = [
                (f"What is the greatest common divisor of {a} and {b}? "
                 f"Return the integer."),
                (f"Find the GCD of {a} and {b}. Return the integer."),
                (f"What is the largest number that divides both {a} and {b}? "
                 f"Return the integer."),
                (f"Compute the greatest common factor of {a} and {b}. "
                 f"Return the integer."),
                (f"Determine the highest common divisor of {a} and {b}. "
                 f"Return the integer."),
            ]
        else:
            phrasings = [
                (f"What is the least common multiple of {a} and {b}? "
                 f"Return the integer."),
                (f"Find the LCM of {a} and {b}. Return the integer."),
                (f"What is the smallest number divisible by both {a} and {b}? "
                 f"Return the integer."),
                (f"Compute the lowest common multiple of {a} and {b}. "
                 f"Return the integer."),
                (f"Determine the least common multiple of {a} and {b}. "
                 f"Return the integer."),
            ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


_GENERATORS = {"A": _gen_a, "B": _gen_b, "C": _gen_c,
               "D": _gen_d, "E": _gen_e, "F": _gen_f,
               "G": _gen_g, "H": _gen_h}


def generate_crossover_task(
    task_id: str,
    rng: random.Random,
    subtype: str,
    *,
    split: str,
    template_slot: int,
    seed: int,
) -> dict[str, Any]:
    """Generate one magnitude-decoupled crossover task."""
    if split not in SPLIT_TEMPLATE_SLOTS:
        raise ValueError(f"unknown split: {split!r}")
    if template_slot not in SPLIT_TEMPLATE_SLOTS[split]:
        raise ValueError(
            f"template slot {template_slot} is not licensed for split {split!r}")
    if subtype not in SUBTYPES:
        raise ValueError(f"unknown subtype: {subtype!r}; expected one of {SUBTYPES}")
    body = _GENERATORS[subtype](rng, template_slot)
    metadata = _metadata(subtype, split, template_slot, seed,
                         prompt=str(body.get("specification", "")))
    task = {
        "task_id": task_id,
        **body,
        "metadata": metadata,
    }
    _assert_no_optimal_backend_encoded(task)
    return task


def _assert_no_optimal_backend_encoded(task: Mapping[str, Any]) -> None:
    for field in FORBIDDEN_METADATA_FIELDS:
        if field in task:
            raise ValueError(
                f"crossover task {task.get('task_id')!r} leaks forbidden "
                f"field {field!r}")
        meta = task.get("metadata", {})
        if isinstance(meta, Mapping) and field in meta:
            raise ValueError(
                f"crossover task {task.get('task_id')!r} metadata leaks "
                f"forbidden field {field!r}")


def generate_crossover_split(
    *,
    split: str,
    n_per_subtype: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate a full crossover split: n_per_subtype tasks per subtype."""
    from .integrity import normalize_prompt
    slots = SPLIT_TEMPLATE_SLOTS[split]
    tasks: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    rng = random.Random(seed)
    from daph_learning.provenance import deterministic_seed
    for subtype in SUBTYPES:
        count = 0
        attempts = 0
        max_attempts = n_per_subtype * 20
        while count < n_per_subtype and attempts < max_attempts:
            attempts += 1
            slot = slots[(count + sum(ord(c) for c in subtype)) % len(slots)]
            global_seed = deterministic_seed(
                split, subtype, str(count), str(attempts)) % (2**31)
            tid = f"{split}_{subtype}_{count:04d}"
            task = generate_crossover_task(
                tid, rng, subtype, split=split,
                template_slot=slot, seed=global_seed)
            spec = str(task.get("specification", ""))
            prompt_hash = hashlib.sha256(
                normalize_prompt(spec).encode("utf-8")).hexdigest()
            if prompt_hash in seen_prompts:
                continue
            seen_prompts.add(prompt_hash)
            tasks.append(task)
            count += 1
        if count < n_per_subtype:
            raise ValueError(
                f"could not generate {n_per_subtype} unique tasks for "
                f"subtype {subtype!r} in split {split!r} after "
                f"{max_attempts} attempts")
    return tasks


def assert_no_within_split_duplicates(tasks: list[dict[str, Any]]) -> None:
    from .integrity import normalize_prompt
    import hashlib
    seen: set[str] = set()
    for t in tasks:
        spec = str(t.get("specification", ""))
        h = hashlib.sha256(normalize_prompt(spec).encode("utf-8")).hexdigest()
        if h in seen:
            raise ValueError(
                f"within-split duplicate prompt in task {t.get('task_id')!r}")
        seen.add(h)


__all__ = [
    "GENERATOR_VERSION",
    "FAMILY_ID",
    "SUBTYPES",
    "SUBTYPE_DESCRIPTIONS",
    "SPLIT_TEMPLATE_SLOTS",
    "FORBIDDEN_METADATA_FIELDS",
    "generate_crossover_task",
    "generate_crossover_split",
    "assert_no_within_split_duplicates",
    "linguistic_template_id",
]
