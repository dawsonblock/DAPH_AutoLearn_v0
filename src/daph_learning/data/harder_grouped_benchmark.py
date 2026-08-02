"""Section 9 — group-first harder crossover benchmark generator.

Thin wrapper around :mod:`harder_crossover_benchmark` that provides
the same group-first split assignment as :mod:`grouped_benchmark`
but uses the magnitude-decoupled generators.

This gives 420 groups (70 per subtype × 6 subtypes) with the harder
benchmark where operand magnitude does NOT predict routing.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from .harder_crossover_benchmark import (
    FAMILY_ID,
    FORBIDDEN_METADATA_FIELDS,
    SUBTYPES,
    SUBTYPE_DESCRIPTIONS,
    _GENERATORS,
    _assert_no_optimal_backend_encoded,
    linguistic_template_id,
)
from .integrity import normalize_prompt

GENERATOR_VERSION = "v0.4.0a3-harder-grouped"

N_GROUPS_PER_SUBTYPE = 70
N_GROUPS_TOTAL = N_GROUPS_PER_SUBTYPE * len(SUBTYPES)  # 560

SPLIT_FRACTIONS = {"train": 0.50, "development": 0.20,
                   "calibration": 0.15, "final": 0.15}


def _assign_groups_to_splits(
    n_groups: int = N_GROUPS_TOTAL,
    *,
    seed: int = 42,
    fractions: dict[str, float] | None = None,
) -> dict[str, list[int]]:
    frac = fractions or SPLIT_FRACTIONS
    rng = random.Random(seed)
    indices = list(range(n_groups))
    rng.shuffle(indices)
    n_train = int(n_groups * frac["train"])
    n_dev = int(n_groups * frac["development"])
    n_cal = int(n_groups * frac["calibration"])
    n_final = n_groups - n_train - n_dev - n_cal
    return {
        "train": indices[:n_train],
        "development": indices[n_train:n_train + n_dev],
        "calibration": indices[n_train + n_dev:n_train + n_dev + n_cal],
        "final": indices[n_train + n_dev + n_cal:],
    }


def _group_id(subtype: str, group_idx: int) -> str:
    return f"{FAMILY_ID}:{subtype}:group_{group_idx:02d}"


def _template_family(subtype: str, group_idx: int) -> str:
    return f"{FAMILY_ID}:{subtype}:family_{group_idx:02d}"


def generate_grouped_crossover_split(
    *,
    split: str,
    n_per_group: int = 8,
    seed: int = 42,
    n_groups: int = N_GROUPS_TOTAL,
    fractions: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    assignments = _assign_groups_to_splits(
        n_groups, seed=seed, fractions=fractions)
    group_indices = assignments.get(split, [])
    if not group_indices:
        return []

    from daph_learning.provenance import deterministic_seed
    tasks: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    rng = random.Random(seed + hash(split) % (2**31))

    for gidx in group_indices:
        subtype = SUBTYPES[gidx % len(SUBTYPES)]
        group_n = gidx // len(SUBTYPES)
        for i in range(n_per_group):
            global_seed = deterministic_seed(
                split, subtype, str(gidx), str(i)) % (2**31)
            tid = f"{split}_{subtype}_g{gidx:02d}_{i:04d}"
            task_rng = random.Random(global_seed)
            body = _GENERATORS[subtype](task_rng, group_n)
            spec = str(body.get("specification", ""))
            prompt_hash = hashlib.sha256(
                normalize_prompt(spec).encode("utf-8")).hexdigest()
            if prompt_hash in seen_prompts:
                for _retry in range(50):
                    retry_seed = (global_seed + _retry * 7919) % (2**31)
                    task_rng = random.Random(retry_seed)
                    body = _GENERATORS[subtype](task_rng, (group_n + _retry + 1))
                    spec = str(body.get("specification", ""))
                    prompt_hash = hashlib.sha256(
                        normalize_prompt(spec).encode("utf-8")).hexdigest()
                    if prompt_hash not in seen_prompts:
                        break
                else:
                    continue
            seen_prompts.add(prompt_hash)
            metadata = {
                "generator": GENERATOR_VERSION,
                "seed": global_seed,
                "split": split,
                "family_id": FAMILY_ID,
                "subtype": subtype,
                "subtype_description": SUBTYPE_DESCRIPTIONS[subtype],
                "template_id": _template_family(subtype, group_n),
                "linguistic_template_id": linguistic_template_id(spec),
                "group_id": _group_id(subtype, gidx),
                "group_index": gidx,
                "template_family": _template_family(subtype, group_n),
                "instance_id": i,
            }
            task = {
                "task_id": tid,
                **body,
                "metadata": metadata,
            }
            _assert_no_optimal_backend_encoded(task)
            tasks.append(task)
    return tasks


def generate_all_grouped_splits(
    *,
    n_per_group: int = 8,
    seed: int = 42,
    n_groups: int = N_GROUPS_TOTAL,
    fractions: dict[str, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Generate all four splits with group-first assignment."""
    return {
        split: generate_grouped_crossover_split(
            split=split, n_per_group=n_per_group, seed=seed,
            n_groups=n_groups, fractions=fractions)
        for split in ("train", "development", "calibration", "final")
    }


def group_assignments(
    *, seed: int = 42, n_groups: int = N_GROUPS_TOTAL,
    fractions: dict[str, float] | None = None,
) -> dict[str, list[int]]:
    return _assign_groups_to_splits(
        n_groups, seed=seed, fractions=fractions)


__all__ = [
    "GENERATOR_VERSION",
    "N_GROUPS_PER_SUBTYPE",
    "N_GROUPS_TOTAL",
    "SPLIT_FRACTIONS",
    "generate_grouped_crossover_split",
    "generate_all_grouped_splits",
    "group_assignments",
]
