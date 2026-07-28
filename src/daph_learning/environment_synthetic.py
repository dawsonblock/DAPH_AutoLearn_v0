"""v0.3.10 — deterministic synthetic closed-loop environment (Section 35).

A deterministic test environment for the full AutoLearn loop:

    experience collection
    -> utility calculation
    -> sample weighting
    -> feature extraction
    -> router training
    -> candidate evaluation
    -> promotion

Hidden representation:
    family S:  h ~ around [+1, 0, ...]
    family L:  h ~ around [-1, 0, ...]

Utilities:
    S-family:  U(S)=1, U(L)=0
    L-family:  U(S)=0, U(L)=1

Add near-ties: U(S)=0.51, U(L)=0.49
Add noisy confidence.

Required: ``regret_candidate < regret_incumbent`` and
``utility_candidate > utility_incumbent``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .policy.types import BackendOutcome, Route


@dataclass(frozen=True)
class SyntheticTask:
    """A task in the synthetic environment."""

    task_id: str
    family: str  # "S" or "L" or "tie"
    specification: str
    expected: Any

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default) if hasattr(self, key) else default


def make_synthetic_tasks(
    n_per_family: int = 50,
    n_ties: int = 10,
    dim: int = 8,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate synthetic tasks for the closed-loop environment.

    Returns a list of task dicts with ``task_id``, ``family``,
    ``specification``, ``expected``, and a hidden ``activation`` feature.
    """
    rng = np.random.default_rng(seed)
    tasks: list[dict[str, Any]] = []
    idx = 0
    # Family S: symbolic is correct.
    for _ in range(n_per_family):
        h = np.zeros(dim, dtype=np.float32)
        h[0] = 1.0 + rng.normal(0, 0.1)
        tasks.append({
            "task_id": f"S{idx}",
            "family": "S",
            "specification": f"symbolic_task_{idx}",
            "expected": 42,
            "activation": h,
        })
        idx += 1
    # Family L: LLM is correct.
    for _ in range(n_per_family):
        h = np.zeros(dim, dtype=np.float32)
        h[0] = -1.0 + rng.normal(0, 0.1)
        tasks.append({
            "task_id": f"L{idx}",
            "family": "L",
            "specification": f"llm_task_{idx}",
            "expected": 42,
            "activation": h,
        })
        idx += 1
    # Near-ties: both backends ~0.5.
    for _ in range(n_ties):
        h = np.zeros(dim, dtype=np.float32)
        h[0] = rng.normal(0, 0.05)
        tasks.append({
            "task_id": f"T{idx}",
            "family": "tie",
            "specification": f"tie_task_{idx}",
            "expected": 42,
            "activation": h,
        })
        idx += 1
    return tasks


def synthetic_execute_fn(task: dict[str, Any], backend: str) -> dict[str, Any]:
    """Execute a backend on a synthetic task.

    Family S: symbolic is correct, LLM is wrong.
    Family L: LLM is correct, symbolic is wrong.
    Tie: both ~0.5 quality (near-tie).
    """
    family = task.get("family", "tie")
    if family == "S":
        if backend == "symbolic":
            return {"output": 42, "execution_succeeded": True, "latency_ms": 10.0,
                    "compute_cost": 0.01, "failure_type": None, "correct": True}
        return {"output": 99, "execution_succeeded": True, "latency_ms": 100.0,
                "compute_cost": 0.1, "failure_type": None, "correct": False}
    if family == "L":
        if backend == "llm":
            return {"output": 42, "execution_succeeded": True, "latency_ms": 100.0,
                    "compute_cost": 0.1, "failure_type": None, "correct": True}
        return {"output": 99, "execution_succeeded": True, "latency_ms": 10.0,
                "compute_cost": 0.01, "failure_type": None, "correct": False}
    # Tie: both ~0.5
    quality = 0.5
    correct = True  # near-tie, both "correct" with quality 0.5
    return {"output": 42, "execution_succeeded": True, "latency_ms": 50.0,
            "compute_cost": 0.05, "failure_type": None, "correct": correct,
            "quality": quality}


def synthetic_capture_fn(tasks: Sequence[dict[str, Any]], class_label: str) -> dict[str, np.ndarray]:
    """Capture activations for synthetic tasks (returns task_id -> activation)."""
    result = {}
    for task in tasks:
        tid = str(task.get("task_id", ""))
        result[tid] = np.asarray(task.get("activation"), dtype=np.float32)
    return result


def synthetic_utility(task: dict[str, Any], route: Route | str) -> float:
    """Compute the utility of a route on a synthetic task.

    U = quality - 0.0*latency - 0.0*cost - 1.0*risk
    For the synthetic env, quality is 1.0 for the correct backend, 0.0 for
    the wrong one, 0.5 for ties.
    """
    if isinstance(route, Route):
        route = route.value
    family = task.get("family", "tie")
    if route == "abstain":
        return 0.0
    if family == "S":
        return 1.0 if route == "symbolic" else 0.0
    if family == "L":
        return 1.0 if route == "llm" else 0.0
    return 0.5  # tie


def synthetic_oracle_utility(task: dict[str, Any]) -> float:
    """``max_a U(a)`` for a synthetic task."""
    family = task.get("family", "tie")
    if family == "S" or family == "L":
        return 1.0
    return 0.5


__all__ = [
    "SyntheticTask",
    "make_synthetic_tasks",
    "synthetic_capture_fn",
    "synthetic_execute_fn",
    "synthetic_oracle_utility",
    "synthetic_utility",
]
