"""DAPH v0.4 — B4-Redesigned task generator for /no_think conditional structure.

With /no_think mode, Qwen3-8B can handle simple arithmetic but fails on
problems requiring multi-step reasoning or large computations. This
generator creates tasks with a WIDER difficulty spread so that:

  - DIRECT (/no_think) wins on simple tasks it can solve in one shot
  - DECOMPOSE wins on multi-step tasks (each sub-step is simple enough
    for /no_think, but the full problem requires tracking intermediates)
  - RETRIEVAL wins on pattern tasks (examples help model see the pattern)
  - RETRIEVAL HURTS on trap tasks (misleading examples)

Subtype design (expected action advantage with /no_think):
  - simple_add:       a + b, a,b < 20
                      → DIRECT wins (trivial)
  - simple_compare:   "is a > b?" a,b < 50
                      → DIRECT wins (trivial)
  - digit_manip:      digit sum or digit count of a number
                      → DIRECT wins (simple, no retrieval match)
  - pattern_extend:   number sequence "2, 4, 8, 16, ?"
                      → RETRIEVAL wins (examples help see the pattern)
  - formula_apply:    apply a formula shown in examples
                      → RETRIEVAL wins (examples show the method)
  - multi_step_arith: 3-4 step arithmetic with medium numbers
                      → DECOMPOSE wins (each step simple, full problem
                        requires tracking intermediates)
  - multi_step_word:  3-4 step word problems with medium numbers
                      → DECOMPOSE wins (same logic)
  - hard_mul:         2-digit × 2-digit multiplication
                      → DECOMPOSE wins (break into partial products)
  - trap_near:        looks like stored pattern but different operation
                      → DIRECT wins, RETRIEVAL HURTS

Within-subtype difficulty variation creates the conditional structure
that hidden states need to detect: same subtype, different difficulty,
different winning action.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


# Subtypes where retrieval examples ARE in the store
RETRIEVAL_POSITIVE_SUBTYPES = {"pattern_extend", "formula_apply"}
# Subtypes where retrieval examples are MISLEADING
RETRIEVAL_TRAP_SUBTYPES = {"trap_near"}
# Subtypes where retrieval store has NO matching examples
RETRIEVAL_NEUTRAL_SUBTYPES = {
    "simple_add", "simple_compare", "digit_manip",
    "multi_step_arith", "multi_step_word", "hard_mul",
}

ALL_SUBTYPES = list(RETRIEVAL_POSITIVE_SUBTYPES | RETRIEVAL_TRAP_SUBTYPES | RETRIEVAL_NEUTRAL_SUBTYPES)


def generate_diverse_tasks(
    n_tasks: int = 600,
    n_groups: int = 40,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate diverse arithmetic tasks across 9 subtypes.

    Each subtype is designed so that a different action has an advantage,
    creating the conditional structure needed for executive routing.

    Parameters
    ----------
    n_tasks : int
        Total number of tasks to generate.
    n_groups : int
        Number of groups (for group-aware bootstrap).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Each dict has: task_id, prompt, answer, subtype, group_id
    """
    rng = np.random.RandomState(seed)
    subtypes = ALL_SUBTYPES
    n_subtypes = len(subtypes)
    tasks = []

    for i in range(n_tasks):
        subtype_idx = i % n_subtypes
        subtype = subtypes[subtype_idx]
        group = f"g{i % n_groups}"

        task = _generate_one_task(subtype, rng, i)
        task["task_id"] = f"diverse_{i:04d}"
        task["subtype"] = subtype
        task["group_id"] = group
        tasks.append(task)

    return tasks


def _generate_one_task(subtype: str, rng: np.random.RandomState, idx: int) -> dict:
    """Generate a single task of the given subtype."""
    if subtype == "simple_add":
        a = int(rng.randint(1, 20))
        b = int(rng.randint(1, 20))
        return {
            "prompt": f"Calculate: {a} + {b} = ?",
            "answer": a + b,
        }

    if subtype == "simple_compare":
        a = int(rng.randint(1, 50))
        b = int(rng.randint(1, 50))
        return {
            "prompt": f"Is {a} greater than {b}? Answer with 1 for yes or 0 for no.",
            "answer": 1 if a > b else 0,
        }

    if subtype == "digit_manip":
        # Simple digit operations that /no_think can handle
        # but retrieval store has no matching examples for
        ops = ["digit_sum", "digit_count"]
        op = ops[idx % len(ops)]
        if op == "digit_sum":
            n = int(rng.randint(100, 9999))
            answer = sum(int(d) for d in str(abs(n)))
            return {
                "prompt": f"What is the sum of all digits in the number {n}?",
                "answer": answer,
            }
        else:
            n = int(rng.randint(100, 1000000))
            return {
                "prompt": f"How many digits are in the number {n}?",
                "answer": len(str(abs(n))),
            }

    if subtype == "pattern_extend":
        # Number sequences — /no_think struggles to identify patterns
        # but retrieval examples of similar patterns help
        patterns = ["geometric", "arithmetic", "square", "fibonacci"]
        pat = patterns[idx % len(patterns)]
        if pat == "geometric":
            start = int(rng.randint(1, 5))
            ratio = int(rng.randint(2, 4))
            seq = [start * ratio**k for k in range(4)]
            answer = start * ratio**4
        elif pat == "arithmetic":
            start = int(rng.randint(1, 10))
            step = int(rng.randint(3, 12))
            seq = [start + step * k for k in range(4)]
            answer = start + step * 4
        elif pat == "square":
            start_n = int(rng.randint(1, 4))
            seq = [(start_n + k) ** 2 for k in range(4)]
            answer = (start_n + 4) ** 2
        else:  # fibonacci
            a, b = int(rng.randint(1, 5)), int(rng.randint(5, 15))
            seq = [a, b]
            for _ in range(3):
                seq.append(seq[-1] + seq[-2])
            answer = seq[-1] + seq[-2]
            seq = seq[:4]
        seq_str = ", ".join(str(x) for x in seq)
        return {
            "prompt": f"What is the next number in the sequence: {seq_str}, ?",
            "answer": answer,
        }

    if subtype == "formula_apply":
        # Apply a formula/method shown in retrieval examples
        # Triangular numbers: T(n) = n*(n+1)/2
        # The store has examples showing the formula
        formulas = ["triangular", "double_sum", "power_sum"]
        fmt = formulas[idx % len(formulas)]
        if fmt == "triangular":
            n = int(rng.randint(5, 20))
            return {
                "prompt": f"What is the {n}th triangular number? (T(n) = n*(n+1)/2)",
                "answer": n * (n + 1) // 2,
            }
        elif fmt == "double_sum":
            n = int(rng.randint(3, 12))
            return {
                "prompt": f"Calculate 2*(1+2+3+...+{n}). What is the result?",
                "answer": 2 * n * (n + 1) // 2,
            }
        else:  # power_sum
            n = int(rng.randint(2, 8))
            return {
                "prompt": f"Calculate 1^2 + 2^2 + 3^2 + ... + {n}^2. What is the result?",
                "answer": n * (n + 1) * (2 * n + 1) // 6,
            }

    if subtype == "multi_step_arith":
        # 3-4 step arithmetic with MEDIUM numbers
        # Each step is simple enough for /no_think
        # But tracking all steps at once is hard without thinking
        ops = rng.choice(["add_mul_sub", "mul_add_div", "add_add_mul", "sub_mul_add"])
        if ops == "add_mul_sub":
            a = int(rng.randint(100, 500))
            b = int(rng.randint(100, 500))
            c = int(rng.randint(2, 12))
            d = int(rng.randint(50, 300))
            return {
                "prompt": (
                    f"First calculate {a} + {b}, then multiply the result by {c}, "
                    f"then subtract {d}. What is the final answer?"
                ),
                "answer": (a + b) * c - d,
            }
        elif ops == "mul_add_div":
            a = int(rng.randint(20, 80))
            c = int(rng.randint(3, 12))
            b = int(rng.randint(100, 500))
            d = int(rng.randint(2, 10))
            product = a * c
            total = product + b
            total = (total // d) * d  # ensure even division
            return {
                "prompt": (
                    f"First calculate {a} × {c}, then add {b}, "
                    f"then divide the result by {d} (integer division). "
                    f"What is the final answer?"
                ),
                "answer": total // d,
            }
        elif ops == "add_add_mul":
            a = int(rng.randint(100, 500))
            b = int(rng.randint(100, 500))
            d = int(rng.randint(10, 100))
            c = int(rng.randint(2, 12))
            return {
                "prompt": (
                    f"First calculate {a} + {b}, then add {d}, "
                    f"then multiply the result by {c}. What is the final answer?"
                ),
                "answer": (a + b + d) * c,
            }
        else:  # sub_mul_add
            a = int(rng.randint(300, 800))
            b = int(rng.randint(50, 200))
            c = int(rng.randint(2, 9))
            d = int(rng.randint(100, 500))
            return {
                "prompt": (
                    f"First calculate {a} - {b}, then multiply the result by {c}, "
                    f"then add {d}. What is the final answer?"
                ),
                "answer": (a - b) * c + d,
            }

    if subtype == "multi_step_word":
        # 3-4 step word problems with medium numbers
        # Each step is simple but the full problem requires tracking
        templates = [
            ("apples", "bought", "gave away", "sold", "has"),
            ("books", "read", "borrowed", "returned", "has"),
            ("marbles", "found", "lost", "traded", "has"),
            ("stickers", "collected", "gave away", "bought", "has"),
        ]
        tpl = templates[idx % len(templates)]
        a = int(rng.randint(50, 200))
        b = int(rng.randint(10, 50))
        c = int(rng.randint(5, 30))
        d = int(rng.randint(2, 20))
        e = int(rng.randint(1, 10))
        return {
            "prompt": (
                f"Sarah {tpl[1]} {a} {tpl[0]}. "
                f"Then she {tpl[2]} {b} of them. "
                f"Later she {tpl[1]} {c} more and her friend gave her {d} more. "
                f"Finally she {tpl[3]} {e} of them. "
                f"How many {tpl[0]} does Sarah {tpl[4]} now?"
            ),
            "answer": a - b + c + d - e,
        }

    if subtype == "hard_mul":
        # 2-digit × 2-digit multiplication
        # /no_think direct struggles with this
        # Decompose can break into: a×10 + a×units, then sum
        a = int(rng.randint(12, 98))
        b = int(rng.randint(12, 98))
        return {
            "prompt": f"Calculate: {a} × {b} = ?",
            "answer": a * b,
        }

    if subtype == "trap_near":
        # Problems that share surface keywords with stored pattern examples
        # but require a DIFFERENT operation
        a = int(rng.randint(3, 13))
        b = int(rng.randint(10, 50))
        # Use "×" as a visual separator but ask for the sum
        return {
            "prompt": f"Given: {a} × {b}. Calculate the sum of these two numbers.",
            "answer": a + b,
        }

    raise ValueError(f"unknown subtype: {subtype}")


def build_retrieval_store(tasks: list[dict], n_per_subtype: int = 5) -> list[dict]:
    """Build a retrieval store with examples from retrieval-positive subtypes ONLY.

    The store contains examples for:
    - pattern_extend: sequence examples (helps pattern_extend tasks)
    - formula_apply: formula examples (helps formula_apply tasks)

    The store does NOT contain examples for:
    - simple_add, simple_compare, digit_manip (direct handles them)
    - multi_step_arith, multi_step_word, hard_mul (decompose handles them)
    - trap_near (store has pattern examples that MISLEAD these tasks)
    """
    by_subtype: dict[str, list[dict]] = {}
    for t in tasks:
        st = t.get("subtype", "unknown")
        by_subtype.setdefault(st, []).append(t)

    store = []
    for st in RETRIEVAL_POSITIVE_SUBTYPES:
        st_tasks = by_subtype.get(st, [])
        for t in st_tasks[:n_per_subtype]:
            store.append({
                "prompt": t["prompt"],
                "answer": t["answer"],
                "subtype": st,
            })
    return store


__all__ = [
    "generate_diverse_tasks",
    "build_retrieval_store",
    "RETRIEVAL_POSITIVE_SUBTYPES",
    "RETRIEVAL_TRAP_SUBTYPES",
    "RETRIEVAL_NEUTRAL_SUBTYPES",
    "ALL_SUBTYPES",
]
