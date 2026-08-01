"""DAPH v0.4 — Diverse task generator for executive qualification.

Generates arithmetic tasks across multiple subtypes designed to create
**conditional structure** — different actions should win on different
subtypes, so the executive policy must actually route conditionally.

Subtype design (expected action advantage):
  - simple_add:       a + b, a,b < 20
                      → DIRECT should win (trivial, no overhead needed)
  - simple_compare:   "is a > b?" a,b < 50
                      → DIRECT should win (simple yes/no)
  - medium_mul:       a × b, a in [3,12], b in [10,99]
                      → RETRIEVAL should win (examples of multiplication
                        patterns help; direct often gets these wrong)
  - pattern_extend:   number sequence "2, 4, 8, 16, ?"
                      → RETRIEVAL should win (similar sequence examples
                        in the store)
  - large_arithmetic: (a + b) × c with large numbers
                      → DECOMPOSE should win (needs step-by-step
                        breakdown; direct fails, retrieval has no
                        exact match)
  - multi_step_word:  word problems requiring 2-3 arithmetic steps
                      → DECOMPOSE should win (must decompose into
                        sub-problems)
  - trap_near:        problems that LOOK like a stored example but
                      have a different operation
                      → RETRIEVAL should HURT (retrieved example
                        suggests wrong answer), DIRECT/DECOMPOSE win
  - novel_pattern:    unusual formats not in the retrieval store
                      → DIRECT should win (retrieval has nothing
                        useful, decompose is overkill)

The retrieval store is built to contain examples for SOME subtypes
(medium_mul, pattern_extend) but NOT others (large_arithmetic,
novel_pattern, trap_near). This creates the conditional structure:
  - Retrieval helps when the store has matching examples
  - Retrieval hurts when the store has misleading examples (trap_near)
  - Retrieval is neutral when the store has no matching examples
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


# Subtypes where retrieval examples ARE in the store
RETRIEVAL_POSITIVE_SUBTYPES = {"medium_mul", "pattern_extend"}
# Subtypes where retrieval examples are MISLEADING
RETRIEVAL_TRAP_SUBTYPES = {"trap_near"}
# Subtypes where retrieval store has NO matching examples
RETRIEVAL_NEUTRAL_SUBTYPES = {
    "simple_add", "simple_compare", "large_arithmetic",
    "multi_step_word", "novel_pattern",
}

ALL_SUBTYPES = list(RETRIEVAL_POSITIVE_SUBTYPES | RETRIEVAL_TRAP_SUBTYPES | RETRIEVAL_NEUTRAL_SUBTYPES)


def generate_diverse_tasks(
    n_tasks: int = 600,
    n_groups: int = 40,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate diverse arithmetic tasks across 8 subtypes.

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

    if subtype == "medium_mul":
        a = int(rng.randint(3, 13))
        b = int(rng.randint(10, 100))
        return {
            "prompt": f"Calculate: {a} × {b} = ?",
            "answer": a * b,
        }

    if subtype == "pattern_extend":
        # Generate a simple geometric or arithmetic sequence
        patterns = ["geometric", "arithmetic", "square"]
        pat = patterns[idx % len(patterns)]
        if pat == "geometric":
            start = int(rng.randint(1, 5))
            ratio = int(rng.randint(2, 4))
            seq = [start * ratio**k for k in range(4)]
            answer = start * ratio**4
        elif pat == "arithmetic":
            start = int(rng.randint(1, 10))
            step = int(rng.randint(2, 8))
            seq = [start + step * k for k in range(4)]
            answer = start + step * 4
        else:  # square
            start_n = int(rng.randint(1, 4))
            seq = [(start_n + k) ** 2 for k in range(4)]
            answer = (start_n + 4) ** 2
        seq_str = ", ".join(str(x) for x in seq)
        return {
            "prompt": f"What is the next number in the sequence: {seq_str}, ?",
            "answer": answer,
        }

    if subtype == "large_arithmetic":
        a = int(rng.randint(100, 500))
        b = int(rng.randint(100, 500))
        c = int(rng.randint(2, 12))
        ops = rng.choice(["add_then_mul", "mul_then_add", "add_then_sub"])
        if ops == "add_then_mul":
            return {
                "prompt": f"First calculate {a} + {b}, then multiply the result by {c}. What is the final answer?",
                "answer": (a + b) * c,
            }
        elif ops == "mul_then_add":
            return {
                "prompt": f"First calculate {a} × {c}, then add {b}. What is the final answer?",
                "answer": a * c + b,
            }
        else:
            return {
                "prompt": f"First calculate {a} + {b}, then subtract {c * 10}. What is the final answer?",
                "answer": a + b - c * 10,
            }

    if subtype == "multi_step_word":
        templates = [
            ("apples", "bought", "gave away", "has"),
            ("books", "read", "borrowed", "has"),
            ("marbles", "found", "lost", "has"),
        ]
        tpl = templates[idx % len(templates)]
        a = int(rng.randint(10, 50))
        b = int(rng.randint(5, 30))
        c = int(rng.randint(1, 10))
        d = int(rng.randint(2, 8))
        return {
            "prompt": (
                f"Sarah {tpl[1]} {a} {tpl[0]}. "
                f"Then she {tpl[2]} {b} of them. "
                f"Later she {tpl[1]} {c} more and her friend gave her {d} more. "
                f"How many {tpl[0]} does Sarah {tpl[3]} now?"
            ),
            "answer": a - b + c + d,
        }

    if subtype == "trap_near":
        # Problems that LOOK like multiplication but are actually addition
        # The retrieval store has multiplication examples, so retrieval
        # will suggest multiplication — but the answer requires addition
        a = int(rng.randint(3, 13))
        b = int(rng.randint(10, 50))
        # Use × symbol but ask for the SUM of the two numbers, not the product
        return {
            "prompt": f"Consider the numbers {a} and {b}. What is their sum? (Note: not the product)",
            "answer": a + b,
        }

    if subtype == "novel_pattern":
        # Unusual problem formats not in the retrieval store
        formats = [
            ("digit_sum", "What is the sum of all digits in the number {n}?"),
            ("reverse_subtract", "Reverse the digits of {n} to get a new number, then subtract the original. What is the result?"),
            ("count_digits", "How many digits are in the number {n}?"),
        ]
        fmt = formats[idx % len(formats)]
        if fmt[0] == "digit_sum":
            n = int(rng.randint(1000, 99999))
            answer = sum(int(d) for d in str(abs(n)))
            return {"prompt": fmt[1].format(n=n), "answer": answer}
        elif fmt[0] == "reverse_subtract":
            n = int(rng.randint(100, 9999))
            rev = int(str(abs(n))[::-1])
            return {"prompt": fmt[1].format(n=n), "answer": rev - n}
        else:  # count_digits
            n = int(rng.randint(100, 1000000))
            return {"prompt": fmt[1].format(n=n), "answer": len(str(abs(n)))}

    raise ValueError(f"unknown subtype: {subtype}")


def build_retrieval_store(tasks: list[dict], n_per_subtype: int = 5) -> list[dict]:
    """Build a retrieval store with examples from retrieval-positive subtypes ONLY.

    The store contains examples for:
    - medium_mul: multiplication examples (helps medium_mul tasks)
    - pattern_extend: sequence examples (helps pattern_extend tasks)

    The store does NOT contain examples for:
    - simple_add, simple_compare (too simple, direct handles them)
    - large_arithmetic, multi_step_word (decompose handles them)
    - trap_near (store has multiplication examples that MISLEAD these tasks)
    - novel_pattern (no matching examples, retrieval is neutral)
    """
    by_subtype: dict[str, list[dict]] = {}
    for t in tasks:
        st = t.get("subtype", "unknown")
        by_subtype.setdefault(st, []).append(t)

    store = []
    # Only include examples from retrieval-positive subtypes
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
