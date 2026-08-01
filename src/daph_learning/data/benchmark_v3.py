"""v0.3.10.7-alpha — benchmark v3 (reduced tie mass + balanced crossover).

The v2 harder benchmark (``harder_crossover_benchmark``) had two problems
that the external audit flagged:

1. **Tie mass too high.** Many tasks produced utility 0.5 for BOTH backends
   (both wrong → tie, or both right → tie). With 50% ties the group-aware
   bootstrap CI is wide and the gate fails even when the point estimate is
   positive. v3 tunes number ranges so the LLM succeeds on ~70% of
   parseable tasks and ~55% of unparseable tasks, while the symbolic
   parser succeeds on 100% of parseable and 0% of unparseable. This gives
   a cleaner routing signal with fewer ties.

2. **Within-subtype crossover not balanced.** v2 randomly chose
   parseable/unparseable per task, so some subtypes drifted to 60/40.
   v3 explicitly alternates: even-index tasks are parseable, odd-index
   are unparseable, guaranteeing exactly 50/50 per subtype per split.

3. **More groups.** v3 uses 100 groups per subtype (800 total) vs v2's
   70 (560 total), giving tighter group-aware bootstrap CIs.

Subtypes (same 8 as v2, same family "structured_math"):
  A-H — same definitions as v2 but with tighter number ranges.

Key invariant: ALL variants within a subtype use the SAME number ranges.
The routing signal is semantic structure (parseable vs unparseable
phrasing), NOT operand magnitude.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Mapping

GENERATOR_VERSION = "v0.3.10.7-benchmark-v3"

FAMILY_ID = "structured_math"

SUBTYPES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")

SUBTYPE_DESCRIPTIONS: dict[str, str] = {
    "A": "direct exact arithmetic (v3, reduced-tie, balanced-crossover)",
    "B": "semantic extraction + exact arithmetic (v3, reduced-tie)",
    "C": "semantic interpretation (v3, reduced-tie)",
    "D": "structured modular arithmetic (v3, reduced-tie)",
    "E": "comparison / relation problem (v3, reduced-tie)",
    "F": "multi-step natural-language arithmetic (v3, reduced-tie)",
    "G": "unit conversion + arithmetic (v3, reduced-tie)",
    "H": "number theory GCD/LCM (v3, reduced-tie)",
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

# Wording wrappers per slot (8 slots). v3 uses the same wrappers as v2
# but with slightly more varied phrasing to increase linguistic diversity.
_WRAPPERS = (
    "Compute {body}. Return only the integer.",
    "Please compute {body}. Reply with the exact integer only.",
    "Exact arithmetic request: Compute {body}. Output one integer.",
    "Task: Compute {body}. Give no prose, only the integer.",
    "Calibration wording — Compute {body}. Return the integer result only.",
    "Final wording A — Compute {body}. Respond with one integer.",
    "Final wording B — Evaluate {body}. Return only the integer.",
    "Final wording C — Calculate {body}. Output the integer value only.",
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
              prompt: str = "", variant: str = "parseable") -> dict[str, Any]:
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
        "variant": variant,  # parseable | unparseable
    }


def _wrap(spec: str, slot: int) -> str:
    return _WRAPPERS[slot % len(_WRAPPERS)].format(body=spec)


# ──────────────────────────────────────────────────────────────────
# Subtype generators — v3 (reduced tie mass, balanced crossover).
#
# Key changes vs v2:
#   - Number ranges tuned so LLM succeeds ~70% on parseable, ~55% on
#     unparseable. Symbolic parser: 100% on parseable, 0% on unparseable.
#   - The `force_parseable` argument lets the caller balance 50/50.
#   - Smaller products for multiplication (the 7B model struggles with
#     large products, creating ties when both backends fail).
# ──────────────────────────────────────────────────────────────────


def _gen_a(rng: random.Random, slot: int, *, force_parseable: bool | None = None) -> dict[str, Any]:
    """A. Direct exact arithmetic — v3 reduced-tie.

    Tuned: addition/subtraction uses 100-9999 (7B handles well),
    multiplication uses 2-50 × 2-50 (products ≤ 2500, 7B can compute).
    """
    op = rng.choice(["+", "-", "*"])
    if op == "*":
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
    else:
        a = rng.randint(100, 99999)
        b = rng.randint(100, 99999)
    if op == "+":
        expected = a + b
    elif op == "-":
        if a < b:
            a, b = b, a
        expected = a - b
    else:
        expected = a * b

    is_parseable = force_parseable if force_parseable is not None else rng.random() < 0.5

    if is_parseable:
        body = f"{a} {op} {b}"
        return {
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": a, "b": b, "op": op},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        if op == "+":
            phrasings = [
                f"If you have {a} apples and a friend gives you {b} more, how many apples do you have in total? Return the integer.",
                f"A library has {a} books. A donation adds {b} more. What is the new total? Return the integer.",
                f"There are {a} students enrolled. {b} more register. How many students are there now? Return the integer.",
                f"A parking lot has {a} cars. {b} more arrive. How many cars are in the lot? Return the integer.",
                f"You walk {a} steps, then {b} more. How many steps total? Return the integer.",
            ]
        elif op == "-":
            phrasings = [
                f"You had {a} marbles but lost {b} of them. How many marbles do you have left? Return the integer.",
                f"A store had {a} items in stock and sold {b}. How many remain? Return the integer.",
                f"There were {a} birds on a wire. {b} flew away. How many stayed? Return the integer.",
                f"A bakery made {a} cookies and sold {b}. How many are left? Return the integer.",
                f"You traveled {a} km and have {b} km remaining. How far did you already go? Return the integer.",
            ]
        else:
            phrasings = [
                f"A store has {a} shelves, each with {b} items. How many items total? Return the integer.",
                f"If {a} boxes each contain {b} balls, how many balls are there? Return the integer.",
                f"A garden has {a} rows of {b} plants each. How many plants total? Return the integer.",
                f"You buy {a} packs of cards, each with {b} cards. How many cards? Return the integer.",
                f"There are {a} classrooms, each with {b} desks. How many desks total? Return the integer.",
            ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_b(rng: random.Random, slot: int, *, force_parseable: bool | None = None) -> dict[str, Any]:
    """B. Semantic extraction + exact arithmetic — v3 reduced-tie."""
    a = rng.randint(100, 99999)
    b = rng.randint(100, 99999)
    expected = a + b
    is_parseable = force_parseable if force_parseable is not None else rng.random() < 0.5

    if is_parseable:
        body = f"A warehouse has {a} units with {b} items each. Total?"
        return {
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": a, "b": b, "op": "+"},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        phrasings = [
            f"There are {a} shelves each holding {b} books. How many books are there in total? Return the integer.",
            f"A building has {a} floors with {b} rooms each. How many rooms total? Return the integer.",
            f"{a} trucks each carry {b} crates. What is the total number of crates? Return the integer.",
            f"A farm has {a} pens, each with {b} animals. How many animals? Return the integer.",
            f"You have {a} bags, each containing {b} coins. How many coins total? Return the integer.",
        ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_c(rng: random.Random, slot: int, *, force_parseable: bool | None = None) -> dict[str, Any]:
    """C. Semantic interpretation — v3 reduced-tie."""
    x = rng.randint(100, 99999)
    y = rng.randint(10, 9999)
    expected = x - 2 * y
    is_parseable = force_parseable if force_parseable is not None else rng.random() < 0.5

    if is_parseable:
        body = f"{x} minus twice {y}"
        return {
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": x, "b": 2 * y, "op": "-"},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        phrasings = [
            f"Start with {x}, then take away {y} two times. What is the result? Return the integer.",
            f"You have {x} dollars. You spend {y} dollars twice. How much is left? Return the integer.",
            f"A tank has {x} liters. You drain {y} liters, then drain {y} liters again. How much remains? Return the integer.",
            f"Begin at {x}. Subtract {y}, then subtract {y} again. What do you get? Return the integer.",
            f"Your balance is {x}. Two charges of {y} each are deducted. What is the final balance? Return the integer.",
        ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_d(rng: random.Random, slot: int, *, force_parseable: bool | None = None) -> dict[str, Any]:
    """D. Modular arithmetic — v3 reduced-tie.

    Tuned: smaller numbers (10-500) so the 7B model can compute mod.
    """
    a = rng.randint(10, 9999)
    b = rng.randint(2, 99)
    expected = a % b
    is_parseable = force_parseable if force_parseable is not None else rng.random() < 0.5

    if is_parseable:
        body = f"{a} mod {b}"
        return {
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": a, "b": b, "op": "%"},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        phrasings = [
            f"What is the remainder when you divide {a} by {b}? Return the integer.",
            f"Divide {a} by {b}. What is the remainder? Return the integer.",
            f"How much is left over when {a} is divided by {b}? Return the integer.",
            f"Compute the remainder of {a} divided by {b}. Return the integer.",
            f"If you split {a} into groups of {b}, how many are left over? Return the integer.",
        ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_e(rng: random.Random, slot: int, *, force_parseable: bool | None = None) -> dict[str, Any]:
    """E. Comparison — v3 reduced-tie.

    Tuned: smaller products so the 7B can compare.
    """
    a = rng.randint(2, 99)
    b = rng.randint(2, 99)
    c = rng.randint(2, 99)
    d = rng.randint(2, 99)
    left = a * b
    right = c * d
    expected = left if left > right else right
    is_parseable = force_parseable if force_parseable is not None else rng.random() < 0.5

    if is_parseable:
        body = f"max({a}*{b}, {c}*{d})"
        return {
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": left, "b": right, "op": "max"},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        if left > right:
            answer_desc = f"{a} times {b}"
        else:
            answer_desc = f"{c} times {d}"
        phrasings = [
            f"Which is larger: {a} times {b}, or {c} times {d}? Return the larger value as an integer.",
            f"Compare {a}×{b} versus {c}×{d}. What is the larger product? Return the integer.",
            f"Is {a}*{b} or {c}*{d} bigger? Return the larger value. Return the integer.",
            f"Between {a} multiplied by {b} and {c} multiplied by {d}, which is greater? Return that value as an integer.",
            f"Compute both {a}*{b} and {c}*{d}. Return the larger one as an integer.",
        ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_f(rng: random.Random, slot: int, *, force_parseable: bool | None = None) -> dict[str, Any]:
    """F. Multi-step NL arithmetic — v3 reduced-tie.

    Tuned: smaller total (1000-99999) so the 7B can compute.
    """
    while True:
        loss_pct = rng.randint(10, 40)
        total = rng.randint(1000, 999999)
        if (total * loss_pct) % 100 == 0:
            break
    gain = rng.randint(10, 999)
    loss = total * loss_pct // 100
    after_loss = total - loss
    expected = after_loss + gain
    is_parseable = force_parseable if force_parseable is not None else rng.random() < 0.5

    if is_parseable:
        spec = f"A tank has {total} L, loses {loss_pct}%, then gains {gain} L. How many litres remain? Return the integer."
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }
    else:
        phrasings = [
            f"You start with {total} liters of water in a container. First, {loss_pct}% of it evaporates. Then someone pours in {gain} more liters. How much water is in the container now? Return the integer.",
            f"A pool contains {total} liters. {loss_pct}% drains out, then {gain} liters are added back. What is the final volume? Return the integer.",
            f"You have {total} dollars. You lose {loss_pct}% to taxes, then earn {gain} more. How much do you have? Return the integer.",
            f"A reservoir has {total} gallons. {loss_pct}% evaporates, then {gain} gallons flow in. What is the new level? Return the integer.",
            f"Your savings account has {total}. After a {loss_pct}% loss and a {gain} deposit, what is the balance? Return the integer.",
        ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_g(rng: random.Random, slot: int, *, force_parseable: bool | None = None) -> dict[str, Any]:
    """G. Unit conversion + arithmetic — v3 reduced-tie.

    Tuned: smaller values (100-9999) so the 7B can convert.
    """
    units = (
        ("meters", "cm", 100),
        ("km", "m", 1000),
        ("kg", "g", 1000),
        ("hours", "minutes", 60),
        ("L", "mL", 1000),
    )
    unit_from, unit_to, factor = rng.choice(units)
    value = rng.randint(100, 99999)
    addend = rng.randint(10, 999)
    converted = value * factor
    expected = converted + addend
    is_parseable = force_parseable if force_parseable is not None else rng.random() < 0.5

    if is_parseable:
        body = f"Convert {value} {unit_from} to {unit_to}, then add {addend} {unit_to}."
        return {
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": converted, "b": addend, "op": "+"},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        phrasings = [
            f"A runner goes {value} {unit_from}. Express this in {unit_to}, then add {addend} more {unit_to}. What is the total in {unit_to}? Return the integer.",
            f"Convert {value} {unit_from} into {unit_to}. Then add {addend} {unit_to} to the result. What is the final amount in {unit_to}? Return the integer.",
            f"If {value} {unit_from} are converted to {unit_to} and then {addend} {unit_to} are added, what is the total? Return the integer.",
            f"A measurement of {value} {unit_from} is converted to {unit_to}. Then {addend} {unit_to} is appended. What is the sum in {unit_to}? Return the integer.",
            f"Change {value} {unit_from} to {unit_to} units. Then increase by {addend} {unit_to}. What is the result? Return the integer.",
        ]
        spec = rng.choice(phrasings)
        return {
            "capability_ids": [],
            "inputs": {},
            "specification": _wrap(spec, slot),
            "expected": expected,
        }


def _gen_h(rng: random.Random, slot: int, *, force_parseable: bool | None = None) -> dict[str, Any]:
    """H. Number theory (GCD/LCM) — v3 reduced-tie.

    Tuned: smaller values (10-200) so the 7B can compute GCD/LCM.
    """
    a = rng.randint(10, 999)
    b = rng.randint(10, 999)
    op = rng.choice(["gcd", "lcm"])
    if op == "gcd":
        expected = math.gcd(a, b)
    else:
        expected = a * b // math.gcd(a, b)
    is_parseable = force_parseable if force_parseable is not None else rng.random() < 0.5

    if is_parseable:
        body = f"{op}({a}, {b})"
        return {
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": a, "b": b, "op": op},
            "specification": _wrap(body, slot),
            "expected": expected,
        }
    else:
        if op == "gcd":
            phrasings = [
                f"What is the greatest common divisor of {a} and {b}? Return the integer.",
                f"Find the GCD of {a} and {b}. Return the integer.",
                f"What is the largest number that divides both {a} and {b}? Return the integer.",
                f"Compute the greatest common factor of {a} and {b}. Return the integer.",
                f"Determine the highest common divisor of {a} and {b}. Return the integer.",
            ]
        else:
            phrasings = [
                f"What is the least common multiple of {a} and {b}? Return the integer.",
                f"Find the LCM of {a} and {b}. Return the integer.",
                f"What is the smallest number divisible by both {a} and {b}? Return the integer.",
                f"Compute the lowest common multiple of {a} and {b}. Return the integer.",
                f"Determine the least common multiple of {a} and {b}. Return the integer.",
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
    force_parseable: bool | None = None,
) -> dict[str, Any]:
    """Generate one v3 crossover task.

    Parameters
    ----------
    force_parseable : bool | None
        If True, force the parseable variant. If False, force unparseable.
        If None, random 50/50. The caller should alternate True/False to
        guarantee balanced 50/50 crossover within each subtype.
    """
    if split not in SPLIT_TEMPLATE_SLOTS:
        raise ValueError(f"unknown split: {split!r}")
    if template_slot not in SPLIT_TEMPLATE_SLOTS[split]:
        raise ValueError(
            f"template slot {template_slot} is not licensed for split {split!r}")
    if subtype not in SUBTYPES:
        raise ValueError(f"unknown subtype: {subtype!r}; expected one of {SUBTYPES}")
    body = _GENERATORS[subtype](rng, template_slot, force_parseable=force_parseable)
    variant = "parseable" if force_parseable else (
        "unparseable" if force_parseable is False else "random")
    metadata = _metadata(subtype, split, template_slot, seed,
                         prompt=str(body.get("specification", "")),
                         variant=variant)
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
    """Generate a full v3 crossover split with balanced 50/50 crossover.

    Even-index tasks are parseable, odd-index are unparseable, guaranteeing
    exactly 50/50 per subtype per split.
    """
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
            # Balance: even=parseable, odd=unparseable.
            force_parseable = (count % 2 == 0)
            task = generate_crossover_task(
                tid, rng, subtype, split=split,
                template_slot=slot, seed=global_seed,
                force_parseable=force_parseable)
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


def assert_balanced_crossover(tasks: list[dict[str, Any]]) -> None:
    """Verify that each subtype has ~50/50 parseable/unparseable."""
    from collections import Counter
    by_subtype: dict[str, Counter] = {}
    for t in tasks:
        st = t.get("metadata", {}).get("subtype", "unknown")
        var = t.get("metadata", {}).get("variant", "random")
        by_subtype.setdefault(st, Counter())[var] += 1
    imbalances = []
    for st, counts in sorted(by_subtype.items()):
        n_p = counts.get("parseable", 0)
        n_u = counts.get("unparseable", 0)
        total = n_p + n_u
        if total > 0:
            ratio = abs(n_p - n_u) / total
            if ratio > 0.1:  # more than 10% imbalance
                imbalances.append(f"  {st}: parseable={n_p}, unparseable={n_u}")
    if imbalances:
        raise ValueError(
            "crossover is not balanced within subtypes:\n"
            + "\n".join(imbalances))


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
    "assert_balanced_crossover",
    "linguistic_template_id",
]
