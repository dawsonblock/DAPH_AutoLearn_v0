"""DAPH v0.4 — Diverse task generator for executive qualification.

Generates arithmetic tasks across multiple subtypes to create meaningful
action gaps:
  - small_add:       a + b where a,b < 20
  - medium_add:      a + b where a,b in [10, 100]
  - large_add:       a + b where a,b in [100, 1000]
  - small_mul:       a * b where a,b < 12
  - medium_mul:      a * b where a in [2,20], b in [10,100]
  - word_problem:    simple word problems requiring arithmetic
  - multi_step:      two-step arithmetic (a + b) * c or similar
  - comparison:      "is a > b?" style problems

Each subtype creates different difficulty profiles for the three actions:
  - direct reasoning: good for simple problems, fails on complex
  - retrieval: good when similar examples are in the store
  - decomposition: good for multi-step, overkill for simple
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def generate_diverse_tasks(
    n_tasks: int = 600,
    n_groups: int = 40,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate diverse arithmetic tasks across 8 subtypes.

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
    subtypes = [
        "small_add",
        "medium_add",
        "large_add",
        "small_mul",
        "medium_mul",
        "word_problem",
        "multi_step",
        "comparison",
    ]
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
    if subtype == "small_add":
        a = int(rng.randint(1, 20))
        b = int(rng.randint(1, 20))
        return {
            "prompt": f"Calculate: {a} + {b} = ?",
            "answer": a + b,
        }

    if subtype == "medium_add":
        a = int(rng.randint(10, 100))
        b = int(rng.randint(10, 100))
        return {
            "prompt": f"Calculate: {a} + {b} = ?",
            "answer": a + b,
        }

    if subtype == "large_add":
        a = int(rng.randint(100, 1000))
        b = int(rng.randint(100, 1000))
        return {
            "prompt": f"Calculate: {a} + {b} = ?",
            "answer": a + b,
        }

    if subtype == "small_mul":
        a = int(rng.randint(2, 12))
        b = int(rng.randint(2, 12))
        return {
            "prompt": f"Calculate: {a} × {b} = ?",
            "answer": a * b,
        }

    if subtype == "medium_mul":
        a = int(rng.randint(2, 20))
        b = int(rng.randint(10, 100))
        return {
            "prompt": f"Calculate: {a} × {b} = ?",
            "answer": a * b,
        }

    if subtype == "word_problem":
        templates = [
            ("apples", "bought", "has"),
            ("books", "read", "has"),
            ("marbles", "found", "has"),
            ("stickers", "collected", "has"),
        ]
        tpl = templates[idx % len(templates)]
        a = int(rng.randint(5, 50))
        b = int(rng.randint(5, 50))
        return {
            "prompt": (
                f"Sarah {tpl[1]} {a} {tpl[0]}. "
                f"Then she got {b} more. "
                f"How many {tpl[0]} does Sarah {tpl[2]} now?"
            ),
            "answer": a + b,
        }

    if subtype == "multi_step":
        a = int(rng.randint(2, 15))
        b = int(rng.randint(2, 15))
        c = int(rng.randint(2, 10))
        ops = rng.choice(["add_then_mul", "mul_then_add", "add_then_sub"])
        if ops == "add_then_mul":
            return {
                "prompt": f"First calculate {a} + {b}, then multiply the result by {c}. What is the final answer?",
                "answer": (a + b) * c,
            }
        elif ops == "mul_then_add":
            return {
                "prompt": f"First calculate {a} × {b}, then add {c}. What is the final answer?",
                "answer": a * b + c,
            }
        else:
            return {
                "prompt": f"First calculate {a} + {b}, then subtract {c}. What is the final answer?",
                "answer": a + b - c,
            }

    if subtype == "comparison":
        a = int(rng.randint(1, 100))
        b = int(rng.randint(1, 100))
        # Answer: 1 if a > b, 0 if a <= b
        return {
            "prompt": f"Is {a} greater than {b}? Answer with 1 for yes or 0 for no.",
            "answer": 1 if a > b else 0,
        }

    raise ValueError(f"unknown subtype: {subtype}")


def build_retrieval_store(tasks: list[dict], n_per_subtype: int = 5) -> list[dict]:
    """Build a retrieval store with examples from each subtype.

    This ensures the retrieval executor has diverse examples to retrieve from.
    """
    by_subtype: dict[str, list[dict]] = {}
    for t in tasks:
        st = t.get("subtype", "unknown")
        by_subtype.setdefault(st, []).append(t)

    store = []
    for st, st_tasks in by_subtype.items():
        # Take the first n_per_subtype from each subtype
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
]
