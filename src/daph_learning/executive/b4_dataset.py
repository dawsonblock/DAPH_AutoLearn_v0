"""DAPH v0.4 — B4 dataset generation with proper group-based splits.

Generates tasks and splits them by GROUP (not by index) into
train/dev/final sets. This prevents information leakage between
splits.

Also fixes the subtype assignment bug: subtypes are assigned within
each group so every split has balanced subtype coverage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from daph_learning.executive.task_generator import (
    _generate_one_task,
    ALL_SUBTYPES,
    RETRIEVAL_POSITIVE_SUBTYPES,
)


@dataclass
class B4DatasetSplit:
    """A split of the B4 dataset."""
    name: str
    groups: list[str]
    tasks: list[dict[str, Any]]

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_groups(self) -> int:
        return len(self.groups)


def generate_b4_dataset(
    n_groups: int = 160,
    tasks_per_group: int = 8,
    seed: int = 20260801,
    train_frac: float = 0.50,
    dev_frac: float = 0.20,
    # final_frac = 1 - train_frac - dev_frac = 0.30
) -> dict[str, B4DatasetSplit]:
    """Generate the B4 dataset with group-based splits.

    Parameters
    ----------
    n_groups : int
        Total number of groups.
    tasks_per_group : int
        Tasks per group (should be divisible by n_subtypes for balance).
    seed : int
        Random seed.
    train_frac : float
        Fraction of groups for training.
    dev_frac : float
        Fraction of groups for development.

    Returns
    -------
    dict[str, B4DatasetSplit]
        Keys: "train", "dev", "final"
    """
    rng = np.random.RandomState(seed)
    n_subtypes = len(ALL_SUBTYPES)

    # Ensure tasks_per_group is divisible by n_subtypes
    assert tasks_per_group % n_subtypes == 0, \
        f"tasks_per_group ({tasks_per_group}) must be divisible by n_subtypes ({n_subtypes})"

    # Assign groups to splits
    all_groups = [f"g{i:03d}" for i in range(n_groups)]
    rng.shuffle(all_groups)

    n_train = int(n_groups * train_frac)
    n_dev = int(n_groups * dev_frac)

    train_groups = all_groups[:n_train]
    dev_groups = all_groups[n_train:n_train + n_dev]
    final_groups = all_groups[n_train + n_dev:]

    # Generate tasks for each group
    # Within each group, assign one task per subtype (cycling)
    task_counter = 0
    all_tasks = []

    for group in all_groups:
        for j in range(tasks_per_group):
            subtype_idx = j % n_subtypes
            subtype = ALL_SUBTYPES[subtype_idx]
            task = _generate_one_task(subtype, rng, task_counter)
            task["task_id"] = f"b4_{task_counter:05d}"
            task["subtype"] = subtype
            task["group_id"] = group
            all_tasks.append(task)
            task_counter += 1

    # Split tasks by group
    def _filter_tasks(groups):
        group_set = set(groups)
        return [t for t in all_tasks if t["group_id"] in group_set]

    return {
        "train": B4DatasetSplit("train", train_groups, _filter_tasks(train_groups)),
        "dev": B4DatasetSplit("dev", dev_groups, _filter_tasks(dev_groups)),
        "final": B4DatasetSplit("final", final_groups, _filter_tasks(final_groups)),
    }


def build_b4_retrieval_store(
    train_tasks: list[dict[str, Any]],
    n_per_subtype: int = 5,
) -> list[dict]:
    """Build retrieval store from training tasks only.

    Same as B3: only includes examples from retrieval-positive subtypes.
    """
    by_subtype: dict[str, list[dict]] = {}
    for t in train_tasks:
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
    "B4DatasetSplit",
    "generate_b4_dataset",
    "build_b4_retrieval_store",
]
