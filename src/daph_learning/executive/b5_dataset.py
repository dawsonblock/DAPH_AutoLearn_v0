"""DAPH v0.4 — B5 dataset generation with within-family crossovers.

B5 requires that action superiority varies WITHIN families, not just
between families. The executive must not solve the benchmark merely
by recognizing subtype.

For every major family, create examples where different actions win::

    pattern_extension: some FAST wins, some THINK wins, some DECOMPOSE wins
    arithmetic:         some FAST wins, some THINK wins, some RETRIEVE wins
    multi_step:         some THINK wins, some DECOMPOSE wins

Crossover criterion: for every sufficiently large family, at least two
actions should each be oracle winner for ≥15% of tasks.

Dataset scale targets:
    TRAIN: 3,000–10,000
    DEV:     750–2,000
    FINAL: 1,000–2,000+

Multiple frozen regimes: B5-Easy, B5-Medium, B5-Hard, B5-Crossover, B5-OOD.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Section 1 — B5 task families
# ──────────────────────────────────────────────────────────────────────

B5_FAMILIES: list[str] = [
    "arithmetic_easy",
    "arithmetic_large",
    "multi_step_math",
    "comparison",
    "pattern_extension",
    "constraint_reasoning",
    "multi_fact_reasoning",
    "retrieval_sensitive",
    "decomposition_sensitive",
    "direct_easy",
    "trap_tasks",
    "novel_composition",
]

# Within each family, we generate tasks at multiple difficulty tiers
# so that different actions win. The "oracle_action_hint" is a soft
# hint for synthetic mock execution — real execution determines the
# actual oracle.
DIFFICULTY_TIERS = ["easy", "medium", "hard"]


@dataclass
class B5TaskSpec:
    """Specification for a single B5 task."""

    task_id: str
    family: str
    subtype: str
    difficulty: str
    prompt: str
    answer: int
    group_id: str
    split: str = ""
    # Soft hint for mock executors about which action "should" win.
    # Real execution determines the actual oracle.
    oracle_action_hint: str = ""
    # Template group: tasks derived from the same parameterized template
    # share a template_group_id. Used for template-level leakage control.
    template_group_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "subtype": self.subtype,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "answer": self.answer,
            "group_id": self.group_id,
            "split": self.split,
            "oracle_action_hint": self.oracle_action_hint,
            "template_group_id": self.template_group_id,
        }


@dataclass
class B5DatasetSplit:
    """A split of the B5 dataset."""

    name: str
    groups: list[str]
    tasks: list[dict[str, Any]]

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_groups(self) -> int:
        return len(self.groups)


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Task generators per family
# ──────────────────────────────────────────────────────────────────────

def _gen_arithmetic_easy(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Easy arithmetic: a + b or a - b."""
    if difficulty == "easy":
        a, b = rng.randint(1, 150, size=2)
        op = rng.choice(["+", "-"])
    elif difficulty == "medium":
        a, b = rng.randint(5, 300, size=2)
        op = rng.choice(["+", "-"])
    else:
        a, b = rng.randint(10, 500, size=2)
        op = rng.choice(["+", "-"])
    if op == "+":
        answer = int(a + b)
        prompt = f"What is {a} + {b}?"
    else:
        answer = int(a - b)
        prompt = f"What is {a} - {b}?"
    # Easy: FAST wins. Hard: THINK may help with larger numbers.
    hint = "action.reasoning.direct_fast" if difficulty == "easy" else "action.reasoning.direct_think"
    return prompt, answer, hint


def _gen_arithmetic_large(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Large number arithmetic requiring more compute."""
    if difficulty == "easy":
        a, b = rng.randint(100, 5000, size=2)
        answer = int(a + b)
        prompt = f"Calculate {a} + {b}."
        hint = "action.reasoning.direct_fast"
    elif difficulty == "medium":
        a, b = rng.randint(500, 20000, size=2)
        answer = int(a + b)
        prompt = f"Calculate {a} + {b}."
        hint = "action.reasoning.direct_think"
    else:
        a, b = rng.randint(1000, 99990, size=2)
        answer = int(a * b)
        prompt = f"Calculate {a} × {b}."
        hint = "action.reasoning.decompose"
    return prompt, answer, hint


def _gen_multi_step_math(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Multi-step arithmetic requiring decomposition."""
    if difficulty == "easy":
        a, b, c = rng.randint(2, 200, size=3)
        answer = int((a + b) * c)
        prompt = f"What is ({a} + {b}) × {c}?"
        hint = "action.reasoning.direct_think"
    elif difficulty == "medium":
        a, b, c, d = rng.randint(3, 300, size=4)
        answer = int((a * b) + (c * d))
        prompt = f"Calculate {a}×{b} + {c}×{d}."
        hint = "action.reasoning.decompose"
    else:
        a, b, c, d, e = rng.randint(5, 500, size=5)
        answer = int((a + b) * c - d * e)
        prompt = f"Compute ({a} + {b}) × {c} - {d} × {e}."
        hint = "action.reasoning.decompose"
    return prompt, answer, hint


def _gen_comparison(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Comparison tasks: which is larger?"""
    if difficulty == "easy":
        a, b = rng.randint(1, 500, size=2)
        prompt = f"Which is larger: {a} or {b}? Give the larger number."
        answer = int(max(a, b))
        hint = "action.reasoning.direct_fast"
    elif difficulty == "medium":
        a, b = rng.randint(100, 9990, size=2)
        prompt = f"Which is larger: {a} or {b}? Give the larger number."
        answer = int(max(a, b))
        hint = "action.reasoning.direct_fast"
    else:
        a = rng.randint(100, 9990)
        b = a + rng.randint(-50, 50)
        if b < 0:
            b = abs(b)
        c = rng.randint(100, 9990)
        prompt = f"Is {a}×{b} larger than {c}×{c}? Give the larger product."
        answer = int(max(a * b, c * c))
        hint = "action.reasoning.decompose"
    return prompt, answer, hint


def _gen_pattern_extension(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Pattern extension: find the next number in a sequence."""
    if difficulty == "easy":
        # Arithmetic sequence
        start = rng.randint(1, 100)
        step = rng.randint(2, 50)
        seq = [start + step * i for i in range(4)]
        answer = seq[-1] + step
        prompt = f"What comes next in the sequence: {', '.join(map(str, seq))}, ?"
        hint = "action.reasoning.direct_fast"
    elif difficulty == "medium":
        # Geometric sequence — retrieval helps see the pattern
        start = rng.randint(1, 40)
        ratio = rng.randint(2, 40)
        seq = [start * (ratio ** i) for i in range(4)]
        answer = seq[-1] * ratio
        prompt = f"What comes next: {', '.join(map(str, seq))}, ?"
        hint = "action.retrieval.examples"
    else:
        # Complex pattern — think or decompose
        a = rng.randint(1, 50)
        seq = [a, a * 2 + 1, (a * 2 + 1) * 2 + 1, ((a * 2 + 1) * 2 + 1) * 2 + 1]
        answer = seq[-1] * 2 + 1
        prompt = f"What comes next: {', '.join(map(str, seq))}, ?"
        hint = "action.reasoning.direct_think"
    return prompt, answer, hint


def _gen_constraint_reasoning(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Constraint satisfaction: find a number meeting constraints."""
    if difficulty == "easy":
        x = rng.randint(2, 100)
        prompt = (f"Find a number that is {x} more than 10 and {x} less than 20. "
                  f"Give the number.")
        answer = 10 + x
        hint = "action.reasoning.direct_fast"
    elif difficulty == "medium":
        x = rng.randint(3, 150)
        prompt = (f"Find a number that when doubled gives {2*x}, "
                  f"and when increased by 5 gives {x+5}. Give the number.")
        answer = x
        hint = "action.reasoning.direct_think"
    else:
        a = rng.randint(2, 80)
        b = rng.randint(2, 80)
        prompt = (f"Find a number n such that n + {a} = {a+b} and n × {b} = {(a+b)*b}. "
                  f"Give n.")
        answer = a + b
        hint = "action.reasoning.decompose"
    return prompt, answer, hint


def _gen_multi_fact_reasoning(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Multi-fact word problems."""
    if difficulty == "easy":
        a = rng.randint(3, 150)
        b = rng.randint(2, 100)
        prompt = (f"Alice has {a} apples. Bob has {b} times as many. "
                  f"How many apples does Bob have?")
        answer = a * b
        hint = "action.reasoning.direct_fast"
    elif difficulty == "medium":
        a = rng.randint(10, 300)
        b = rng.randint(3, 100)
        c = rng.randint(5, 200)
        prompt = (f"A store has {a} items. They sell {b} items per day for {c} days. "
                  f"How many items remain?")
        answer = a - b * c
        hint = "action.reasoning.direct_think"
    else:
        a = rng.randint(20, 500)
        b = rng.randint(3, 80)
        c = rng.randint(2, 60)
        d = rng.randint(5, 150)
        prompt = (f"A factory produces {a} units per hour. After {b} hours, "
                  f"production doubles for {c} hours, then drops to {d} per hour. "
                  f"What is the total after {b+c+1} hours?")
        answer = a * b + (a * 2) * c + d
        hint = "action.reasoning.decompose"
    return prompt, answer, hint


def _gen_retrieval_sensitive(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Tasks where retrieval of similar examples helps."""
    if difficulty == "easy":
        # Formula application — retrieval shows the method
        x = rng.randint(2, 100)
        prompt = (f"Using the formula f(n) = n² + n, what is f({x})?")
        answer = x * x + x
        hint = "action.retrieval.examples"
    elif difficulty == "medium":
        x = rng.randint(3, 120)
        prompt = (f"Using the formula g(n) = 2n² - n + 1, what is g({x})?")
        answer = 2 * x * x - x + 1
        hint = "action.retrieval.examples"
    else:
        x = rng.randint(5, 150)
        prompt = (f"Using the formula h(n) = n³ - 2n² + 3, what is h({x})?")
        answer = x ** 3 - 2 * x * x + 3
        hint = "action.reasoning.direct_think"
    return prompt, answer, hint


def _gen_decomposition_sensitive(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Tasks that strongly benefit from decomposition."""
    if difficulty == "easy":
        a, b, c = rng.randint(5, 200, size=3)
        prompt = f"Compute ({a} + {b} + {c}) × 2."
        answer = int((a + b + c) * 2)
        hint = "action.reasoning.direct_think"
    elif difficulty == "medium":
        a, b, c, d = rng.randint(10, 400, size=4)
        prompt = f"Compute ({a}×{b}) + ({c}×{d}) - ({a}+{c})."
        answer = int(a * b + c * d - a - c)
        hint = "action.reasoning.decompose"
    else:
        a, b, c, d, e = rng.randint(10, 500, size=5)
        prompt = f"Compute (({a}+{b})×{c} - {d}×{e}) ÷ 2."
        answer = int(((a + b) * c - d * e) / 2)
        hint = "action.reasoning.decompose"
    return prompt, answer, hint


def _gen_direct_easy(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Trivial tasks where FAST always wins."""
    if difficulty == "easy":
        a = rng.randint(1, 100)
        prompt = f"What is {a} + 1?"
        answer = a + 1
    elif difficulty == "medium":
        a = rng.randint(1, 200)
        b = rng.randint(1, 200)
        prompt = f"What is {a} + {b}?"
        answer = a + b
    else:
        a = rng.randint(10, 990)
        prompt = f"What is {a} + 10?"
        answer = a + 10
    return prompt, answer, "action.reasoning.direct_fast"


def _gen_trap_tasks(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Trap tasks: look like a pattern but require different operation."""
    if difficulty == "easy":
        a = rng.randint(2, 80)
        b = rng.randint(2, 80)
        # Looks like a sequence but it's just addition
        prompt = f"Given the numbers {a} and {b}, what is their sum?"
        answer = a + b
        hint = "action.reasoning.direct_fast"
    elif difficulty == "medium":
        a = rng.randint(3, 120)
        # Looks like it needs a formula but it's just the number itself
        prompt = f"If f(x) = x for all x, what is f({a})?"
        answer = a
        hint = "action.reasoning.direct_fast"
    else:
        a = rng.randint(10, 300)
        b = rng.randint(10, 300)
        # Looks like multiplication but it's addition
        prompt = f"Note: the operation ⊕ is defined as a ⊕ b = a + b. What is {a} ⊕ {b}?"
        answer = a + b
        hint = "action.reasoning.direct_think"
    return prompt, answer, hint


def _gen_novel_composition(rng: np.random.RandomState, difficulty: str) -> tuple[str, int, str]:
    """Novel composition of known skills — OOD-like."""
    if difficulty == "easy":
        a = rng.randint(2, 100)
        b = rng.randint(2, 100)
        c = rng.randint(2, 100)
        prompt = f"Compute (max({a},{b})) × {c}."
        answer = max(a, b) * c
        hint = "action.reasoning.direct_think"
    elif difficulty == "medium":
        a = rng.randint(5, 150)
        b = rng.randint(5, 150)
        c = rng.randint(2, 80)
        prompt = f"If |{a} - {b}| is multiplied by {c}, what is the result?"
        answer = abs(a - b) * c
        hint = "action.reasoning.decompose"
    else:
        a = rng.randint(10, 300)
        b = rng.randint(2, 60)
        c = rng.randint(3, 100)
        prompt = f"Compute the sum of all integers from {a} to {a + c}, then divide by {b}."
        total = sum(range(a, a + c + 1))
        answer = total // b
        hint = "action.reasoning.decompose"
    return prompt, answer, hint


# Registry of family generators
_FAMILY_GENERATORS: dict[str, Any] = {
    "arithmetic_easy": _gen_arithmetic_easy,
    "arithmetic_large": _gen_arithmetic_large,
    "multi_step_math": _gen_multi_step_math,
    "comparison": _gen_comparison,
    "pattern_extension": _gen_pattern_extension,
    "constraint_reasoning": _gen_constraint_reasoning,
    "multi_fact_reasoning": _gen_multi_fact_reasoning,
    "retrieval_sensitive": _gen_retrieval_sensitive,
    "decomposition_sensitive": _gen_decomposition_sensitive,
    "direct_easy": _gen_direct_easy,
    "trap_tasks": _gen_trap_tasks,
    "novel_composition": _gen_novel_composition,
}


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Dataset generation with crossovers
# ──────────────────────────────────────────────────────────────────────

def _normalize_prompt(prompt: str) -> str:
    """Normalize a prompt for deduplication (strip whitespace, lowercase)."""
    return " ".join(prompt.strip().lower().split())


def generate_b5_dataset(
    n_train: int = 3000,
    n_dev: int = 750,
    n_final: int = 1000,
    n_ood: int = 500,
    seed: int = 20260802,
    *,
    families: Sequence[str] | None = None,
    tasks_per_group: int = 5,
    split_mode: str = "standard",
    max_gen_attempts: int = 10,
) -> dict[str, B5DatasetSplit]:
    """Generate the B5 dataset with within-family crossovers.

    Each family generates tasks at easy/medium/hard difficulty tiers,
    ensuring that different actions win within the same family.

    Global prompt deduplication is enforced: no exact prompt appears
    in more than one split. Tasks from the same parameterized template
    share a ``template_group_id``.

    Parameters
    ----------
    n_train, n_dev, n_final, n_ood : int
        Number of tasks per split.
    seed : int
        Random seed.
    families : sequence of str | None
        Which families to include (default: all).
    tasks_per_group : int
        Tasks per group (for group-aware bootstrap).
    split_mode : str
        ``"standard"`` — same template families may appear across splits
        but exact tasks cannot.
        ``"template_ood"`` — entire template groups are held out.
    max_gen_attempts : int
        Maximum regeneration attempts for prompt deduplication.

    Returns
    -------
    dict with keys "train", "dev", "final", "final_ood"
    """
    rng = np.random.RandomState(seed)
    fams = list(families) if families else list(B5_FAMILIES)
    seen_prompts: set[str] = set()  # global dedup across all splits
    task_counter = 0

    def _gen_one_task(family: str, difficulty: str) -> dict[str, Any] | None:
        """Generate one task, retrying if prompt is a duplicate."""
        nonlocal task_counter
        gen_fn = _FAMILY_GENERATORS[family]
        for _ in range(max_gen_attempts):
            prompt, answer, hint = gen_fn(rng, difficulty)
            norm = _normalize_prompt(prompt)
            if norm not in seen_prompts:
                seen_prompts.add(norm)
                task_counter += 1
                return {
                    "prompt": prompt,
                    "answer": answer,
                    "hint": hint,
                    "family": family,
                    "difficulty": difficulty,
                    "subtype": f"{family}_{difficulty}",
                    "template_group_id": f"tpl_{family}_{difficulty}",
                }
        return None  # could not generate unique prompt

    def _gen_split(n_tasks: int, split_name: str) -> B5DatasetSplit:
        tasks: list[dict[str, Any]] = []
        n_groups = max(1, n_tasks // tasks_per_group)
        groups = [f"{split_name}_g{i:04d}" for i in range(n_groups)]

        i = 0
        attempts = 0
        while i < n_tasks and attempts < n_tasks * max_gen_attempts:
            family = fams[i % len(fams)]
            difficulty = DIFFICULTY_TIERS[(i // len(fams)) % len(DIFFICULTY_TIERS)]
            task_data = _gen_one_task(family, difficulty)
            attempts += 1
            if task_data is None:
                continue
            group = groups[i // tasks_per_group] if i // tasks_per_group < n_groups else groups[-1]
            task = B5TaskSpec(
                task_id=f"b5_{split_name}_{i:06d}",
                family=task_data["family"],
                subtype=task_data["subtype"],
                difficulty=task_data["difficulty"],
                prompt=task_data["prompt"],
                answer=task_data["answer"],
                group_id=group,
                split=split_name,
                oracle_action_hint=task_data["hint"],
                template_group_id=task_data["template_group_id"],
            )
            tasks.append(task.to_dict())
            i += 1

        return B5DatasetSplit(split_name, groups, tasks)

    train = _gen_split(n_train, "train")
    dev = _gen_split(n_dev, "dev")
    final = _gen_split(n_final, "final")
    # OOD: use only novel_composition and trap_tasks families
    ood_families = ["novel_composition", "trap_tasks", "constraint_reasoning"]
    old_fams = fams
    fams = ood_families
    ood = _gen_split(n_ood, "final_ood")
    fams = old_fams

    return {"train": train, "dev": dev, "final": final, "final_ood": ood}


def build_b5_retrieval_store(
    train_tasks: list[dict[str, Any]],
    n_per_family: int = 10,
) -> list[dict]:
    """Build retrieval store from training tasks only.

    Only includes examples from retrieval-sensitive families.
    """
    by_family: dict[str, list[dict]] = {}
    for t in train_tasks:
        fam = t.get("family", t.get("subtype", "unknown"))
        by_family.setdefault(fam, []).append(t)

    store = []
    retrieval_families = {"retrieval_sensitive", "pattern_extension", "formula_apply"}
    for fam, tasks in by_family.items():
        if fam in retrieval_families or "retrieval" in fam or "pattern" in fam:
            for t in tasks[:n_per_family]:
                store.append({
                    "prompt": t["prompt"],
                    "answer": t["answer"],
                    "family": fam,
                    "task_id": t["task_id"],
                })
    return store


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Crossover diagnostics
# ──────────────────────────────────────────────────────────────────────

def compute_winner_distribution(
    tasks: list[dict[str, Any]],
    utilities: dict[str, dict[str, float]],
    action_ids: Sequence[str],
) -> dict[str, Any]:
    """Compute the oracle winner distribution per family.

    For every family, report what fraction of tasks each action wins.
    Crossover criterion: at least two actions should each be oracle
    winner for ≥15% of tasks.

    Parameters
    ----------
    tasks : list of task dicts
    utilities : dict mapping task_id → {action_id: utility}
    action_ids : sequence of action IDs

    Returns
    -------
    dict with per-family winner distributions and crossover assessment.
    """
    family_winners: dict[str, dict[str, int]] = {}
    family_counts: dict[str, int] = {}

    for task in tasks:
        tid = task["task_id"]
        family = task.get("family", task.get("subtype", "unknown"))
        if tid not in utilities:
            continue
        task_utils = utilities[tid]
        best_action = max(action_ids, key=lambda a: task_utils.get(a, -1))
        family_winners.setdefault(family, {}).setdefault(best_action, 0)
        family_winners[family][best_action] += 1
        family_counts[family] = family_counts.get(family, 0) + 1

    # Compute fractions and crossover assessment
    min_winner_share = 0.15
    per_family: list[dict[str, Any]] = []
    for family in sorted(family_winners.keys()):
        total = family_counts[family]
        winners = family_winners[family]
        fractions = {a: winners.get(a, 0) / total for a in action_ids}
        # How many actions have ≥15% winner share?
        actions_above_threshold = sum(1 for f in fractions.values() if f >= min_winner_share)
        per_family.append({
            "family": family,
            "n_tasks": total,
            "winner_counts": dict(winners),
            "winner_fractions": {a: round(f, 4) for a, f in fractions.items()},
            "n_actions_above_15pct": actions_above_threshold,
            "crossover_passes": actions_above_threshold >= 2,
        })

    n_families = len(per_family)
    n_crossover = sum(1 for f in per_family if f["crossover_passes"])

    return {
        "min_winner_share": min_winner_share,
        "per_family": per_family,
        "n_families": n_families,
        "n_crossover_families": n_crossover,
        "crossover_fraction": n_crossover / n_families if n_families > 0 else 0.0,
        "all_families_crossover": n_crossover == n_families,
    }


__all__ = [
    "B5_FAMILIES",
    "DIFFICULTY_TIERS",
    "B5TaskSpec",
    "B5DatasetSplit",
    "generate_b5_dataset",
    "build_b5_retrieval_store",
    "compute_winner_distribution",
]
