"""Deterministic v0.3.8 benchmark generator with template-disjoint splits.

The protocol holds out prompt-template families while retaining all thirteen
task categories in every split.  This is a stronger benchmark than random-row
splitting, but it is still a controlled synthetic benchmark—not proof of broad
real-world OOD generalization.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

from .integrity import assert_no_leakage, normalize_prompt


GENERATOR_VERSION = "v0.3.8"

SYMBOLIC_CATEGORIES = (
    "easy_add",
    "easy_sub",
    "multiply",
    "modulo",
    "modular_multiplication",
    "nested_arithmetic",
    "signed_nested_arithmetic",
)
NON_SYMBOLIC_CATEGORIES = (
    "text_question",
    "coding_question",
    "knowledge_question",
)
ADVERSARIAL_CATEGORIES = (
    "irrelevant_numeric",
    "ambiguous_tool",
)
MALFORMED_CATEGORIES = ("unsupported_arithmetic",)
ALL_CATEGORIES = (
    SYMBOLIC_CATEGORIES
    + NON_SYMBOLIC_CATEGORIES
    + ADVERSARIAL_CATEGORIES
    + MALFORMED_CATEGORIES
)

SPLIT_TEMPLATE_SLOTS: dict[str, tuple[int, ...]] = {
    "train": (0, 1, 2, 3),
    "dev": (4, 5),
    "calibration": (6,),
    "final_test": (7,),
}

_EXACT_WRAPPERS = (
    "Compute {body}. Return only the integer.",
    "Please compute {body}. Reply with the exact integer only.",
    "Exact arithmetic request: Compute {body}. Output one integer.",
    "Task: Compute {body}. Give no prose, only the integer.",
    "For verification, compute {body}. Return the integer result only.",
    "Backend decision task — Compute {body}. Respond with one integer.",
    "Evaluate exactly: Compute {body}. The response must be an integer.",
    "Held-out wording: Compute {body}. Print only its integer value.",
)

_PROSE_WRAPPERS = (
    "{body} Give a concise response of about {budget} words.",
    "Explain for a {audience}: {body} Keep it near {budget} words.",
    "Answer this non-computational request: {body} Limit: {budget} words.",
    "Write a clear response for a {audience}. Question: {body} About {budget} words.",
    "Held-out phrasing A — {body} Use at most {budget} words.",
    "Held-out phrasing B — For a {audience}, answer: {body} Target {budget} words.",
    "Calibration wording — {body} Be precise and stay under {budget} words.",
    "Final wording — {body} Explain it to a {audience} in roughly {budget} words.",
)

_TEXT_TOPICS = (
    "why the daytime sky appears blue",
    "why sunsets often appear red",
    "how transformer attention combines token information",
    "the trade-offs between recursion and iteration",
    "why regular exercise affects sleep quality",
    "how public-key cryptography supports secure communication",
    "why ocean tides vary over a month",
    "how a compiler differs from an interpreter",
    "why leaves change colour in autumn",
    "how vaccines train immune memory",
    "why database indexes accelerate some queries",
    "how photosynthesis stores solar energy",
    "why airplanes can generate lift",
    "how version control supports collaborative software work",
    "why seasons differ between hemispheres",
    "how error-correcting codes recover damaged messages",
)
_AUDIENCES = (
    "new programmer",
    "high-school student",
    "technical manager",
    "curious adult",
    "first-year engineering student",
    "non-specialist",
    "software tester",
    "science teacher",
)
_ALGORITHMS = (
    "reverse a singly linked list",
    "merge two sorted sequences",
    "detect a cycle in a linked list",
    "find the first non-repeating character",
    "compute Fibonacci numbers iteratively",
    "topologically sort a directed acyclic graph",
    "validate balanced brackets",
    "find the longest common prefix",
    "perform binary search with duplicate values",
    "group anagrams deterministically",
    "compute a running median",
    "serialize a binary tree",
    "find connected components in a graph",
    "rotate a matrix clockwise",
    "deduplicate records while preserving order",
    "implement an LRU cache",
    "find the kth smallest array element",
    "merge overlapping intervals",
    "parse a CSV row with quoted fields",
    "calculate edit distance",
)
_CODE_CONSTRAINTS = (
    "without recursion",
    "with type hints",
    "without mutating the input",
    "with O(n) auxiliary memory or less",
    "and include boundary-case tests",
    "using only the Python standard library",
    "with deterministic output ordering",
    "and raise ValueError on malformed input",
    "with an explanatory docstring",
    "without global state",
)
_KNOWLEDGE_QUESTIONS = (
    "Which city is the capital of France?",
    "Who wrote One Hundred Years of Solitude?",
    "What is the chemical formula for water?",
    "In what year did the Berlin Wall fall?",
    "What is the largest planet in the solar system?",
    "Which element has atomic number 6?",
    "Who developed the theory of general relativity?",
    "What ocean lies between Africa and Australia?",
    "Which language is primarily spoken in Brazil?",
    "What process converts liquid water into vapour?",
    "Which organ pumps blood through the human body?",
    "What is the SI unit of electric current?",
    "Which planet is known for its prominent rings?",
    "Who painted The Starry Night?",
    "What is the smallest prime number?",
    "Which continent contains the Sahara Desert?",
    "What gas do plants absorb during photosynthesis?",
    "What is the longest river in South America?",
    "Which scientist formulated the laws of motion?",
    "What is the freezing point of water in Celsius?",
)


def _oracles(capability: str, accuracy: str, utility: str) -> dict[str, str]:
    return {
        "capability_oracle": capability,
        "accuracy_oracle": accuracy,
        "utility_oracle": utility,
        "route_label": utility,
    }


def _metadata(category: str, split: str, template_slot: int, seed: int) -> dict[str, Any]:
    if category in SYMBOLIC_CATEGORIES:
        group = "symbolic"
    elif category in NON_SYMBOLIC_CATEGORIES:
        group = "non_symbolic"
    elif category in ADVERSARIAL_CATEGORIES:
        group = "adversarial"
    else:
        group = "malformed"
    return {
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "split": split,
        "category": category,
        "category_group": group,
        "template_id": f"{category}:template_{template_slot}",
        "split_family": f"{category}:template_{template_slot}",
    }


def _exact_prompt(body: str, slot: int) -> str:
    return _EXACT_WRAPPERS[slot].format(body=body)


def generate_task(
    task_id: str,
    rng: random.Random,
    category: str,
    *,
    split: str,
    template_slot: int,
    seed: int,
) -> dict[str, Any]:
    if split not in SPLIT_TEMPLATE_SLOTS:
        raise ValueError(f"unknown split: {split!r}")
    if template_slot not in SPLIT_TEMPLATE_SLOTS[split]:
        raise ValueError(
            f"template slot {template_slot} is not licensed for split {split!r}"
        )
    if category not in ALL_CATEGORIES:
        raise ValueError(f"unknown category: {category!r}")

    metadata = _metadata(category, split, template_slot, seed)
    if category in {"easy_add", "easy_sub", "multiply", "modulo"}:
        a = rng.randint(-5_000_000, 5_000_000)
        b = rng.randint(-5_000_000, 5_000_000)
        if category == "easy_add":
            op, expected = "+", a + b
            oracle = _oracles("symbolic", "llm", "llm")
        elif category == "easy_sub":
            op, expected = "-", a - b
            oracle = _oracles("symbolic", "llm", "llm")
        elif category == "multiply":
            op, expected = "*", a * b
            oracle = _oracles("symbolic", "symbolic", "symbolic")
        else:
            b = rng.randint(2, 100_000)
            op, expected = "%", a % b
            oracle = _oracles("symbolic", "symbolic", "symbolic")
        return {
            "task_id": task_id,
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": a, "b": b, "op": op},
            "specification": _exact_prompt(f"{a} {op} {b}", template_slot),
            "expected": expected,
            **oracle,
            "metadata": metadata,
        }

    if category == "modular_multiplication":
        a = rng.randint(-1_000_000, 1_000_000)
        b = rng.randint(-1_000_000, 1_000_000)
        modulus = rng.randint(2, 100_000)
        return {
            "task_id": task_id,
            "capability_ids": ["modular_multiplication"],
            "inputs": {"a": a, "b": b, "modulus": modulus},
            "specification": _exact_prompt(
                f"({a} * {b}) mod {modulus}",
                template_slot,
            ),
            "expected": (a * b) % modulus,
            **_oracles("symbolic", "symbolic", "symbolic"),
            "metadata": metadata,
        }

    if category in {"nested_arithmetic", "signed_nested_arithmetic"}:
        a = rng.randint(-50_000 if category.startswith("signed") else 100, 50_000)
        b = rng.randint(-10_000, 10_000)
        c = rng.randint(2, 100)
        operator = "-" if category.startswith("signed") else "+"
        expression = f"({a} {operator} {b}) * {c}"
        expected = (a - b) * c if operator == "-" else (a + b) * c
        return {
            "task_id": task_id,
            "capability_ids": ["integer_arithmetic"],
            "inputs": {},
            "specification": _exact_prompt(expression, template_slot),
            "expected": expected,
            **_oracles("symbolic", "symbolic", "symbolic"),
            "metadata": metadata,
        }

    if category == "text_question":
        body = f"Explain {_TEXT_TOPICS[rng.randrange(len(_TEXT_TOPICS))]}."
        audience = _AUDIENCES[rng.randrange(len(_AUDIENCES))]
        budget = rng.randint(70, 320)
        specification = _PROSE_WRAPPERS[template_slot].format(
            body=body,
            audience=audience,
            budget=budget,
        )
        capabilities: list[str] = []
    elif category == "coding_question":
        algorithm = _ALGORITHMS[rng.randrange(len(_ALGORITHMS))]
        constraint = _CODE_CONSTRAINTS[rng.randrange(len(_CODE_CONSTRAINTS))]
        body = f"Write a Python function to {algorithm} {constraint}."
        specification = _PROSE_WRAPPERS[template_slot].format(
            body=body,
            audience="software engineer",
            budget=rng.randint(90, 360),
        )
        capabilities = ["code_generation"]
    elif category == "knowledge_question":
        body = _KNOWLEDGE_QUESTIONS[rng.randrange(len(_KNOWLEDGE_QUESTIONS))]
        specification = _PROSE_WRAPPERS[template_slot].format(
            body=body,
            audience=_AUDIENCES[rng.randrange(len(_AUDIENCES))],
            budget=rng.randint(20, 140),
        )
        capabilities = ["knowledge"]
    elif category == "irrelevant_numeric":
        a = rng.randint(1000, 999999)
        b = rng.randint(100, 99999)
        specification = (
            f"Discuss the historical or cultural significance of {a}; use {b} "
            "only as contextual detail and do not perform arithmetic."
        )
        capabilities = []
    elif category == "ambiguous_tool":
        a = rng.randint(2, 100_000)
        body = (
            f"Is {a} likely to be prime? Explain a verification strategy "
            "without claiming an unverified result."
        )
        specification = _PROSE_WRAPPERS[template_slot].format(
            body=body,
            audience="mathematics student",
            budget=rng.randint(70, 250),
        )
        capabilities = []
    else:
        a = rng.randint(1, 1_000_000)
        b = rng.randint(2, 100_000)
        return {
            "task_id": task_id,
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": a, "b": b, "op": "/"},
            "specification": _exact_prompt(f"{a} / {b}", template_slot),
            "expected": None,
            **_oracles("symbolic", "llm", "llm"),
            "metadata": metadata,
        }

    return {
        "task_id": task_id,
        "capability_ids": capabilities,
        "inputs": {},
        "specification": specification,
        "expected": None,
        **_oracles("llm", "llm", "llm"),
        "metadata": metadata,
    }


def generate_protocol_split(
    num_tasks: int,
    *,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    if num_tasks < len(ALL_CATEGORIES):
        raise ValueError(
            f"num_tasks must be at least {len(ALL_CATEGORIES)} for category coverage"
        )
    slots = SPLIT_TEMPLATE_SLOTS.get(split)
    if slots is None:
        raise ValueError(f"unknown split: {split!r}")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for index in range(num_tasks):
        category = ALL_CATEGORIES[index % len(ALL_CATEGORIES)]
        slot = slots[(index // len(ALL_CATEGORIES)) % len(slots)]
        for _attempt in range(10_000):
            task = generate_task(
                f"{split}-{index + 1:06d}",
                rng,
                category,
                split=split,
                template_slot=slot,
                seed=seed,
            )
            normalized = normalize_prompt(task["specification"])
            if normalized not in seen_prompts:
                seen_prompts.add(normalized)
                rows.append(task)
                break
        else:
            raise RuntimeError(
                f"could not generate a unique {category!r} prompt for {split!r}"
            )
    rng.shuffle(rows)
    return rows


def generate_protocol_splits(
    sizes: Mapping[str, int] | None = None,
    *,
    seed: int = 1337,
) -> dict[str, list[dict[str, Any]]]:
    requested = dict(
        sizes
        or {
            "train": 2500,
            "dev": 1000,
            "calibration": 500,
            "final_test": 1000,
        }
    )
    if set(requested) != set(SPLIT_TEMPLATE_SLOTS):
        raise ValueError(
            f"sizes must define exactly {sorted(SPLIT_TEMPLATE_SLOTS)}"
        )
    splits = {
        split: generate_protocol_split(
            int(size),
            split=split,
            seed=seed + index * 1009,
        )
        for index, (split, size) in enumerate(requested.items())
    }
    assert_no_leakage(splits)
    return splits


__all__ = [
    "ADVERSARIAL_CATEGORIES",
    "ALL_CATEGORIES",
    "GENERATOR_VERSION",
    "MALFORMED_CATEGORIES",
    "NON_SYMBOLIC_CATEGORIES",
    "SPLIT_TEMPLATE_SLOTS",
    "SYMBOLIC_CATEGORIES",
    "generate_protocol_split",
    "generate_protocol_splits",
    "generate_task",
]
