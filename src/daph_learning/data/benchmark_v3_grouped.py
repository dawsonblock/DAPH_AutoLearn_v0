"""Section 9 — group-first benchmark v3 wrapper.

Thin wrapper around :mod:`benchmark_v3` that provides the same group-first
split assignment as :mod:`harder_grouped_benchmark` but uses the v3
benchmark (reduced tie mass + balanced within-subtype crossover).

This gives 100 groups (100 per subtype × 8 subtypes) with the v3
benchmark where operand magnitude does NOT predict routing and crossover
is balanced 50/50 within each subtype.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from .benchmark_v3 import (
    FAMILY_ID,
    FORBIDDEN_METADATA_FIELDS,
    SUBTYPES,
    SUBTYPE_DESCRIPTIONS,
    SPLIT_TEMPLATE_SLOTS,
    _GENERATORS,
    _assert_no_optimal_backend_encoded,
    linguistic_template_id,
)
from .integrity import normalize_prompt

GENERATOR_VERSION = "v0.3.10.7-benchmark-v3-grouped"

N_GROUPS_PER_SUBTYPE = 100
N_GROUPS_TOTAL = N_GROUPS_PER_SUBTYPE * len(SUBTYPES)  # 800

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
    return f"{FAMILY_ID}:{subtype}:group_{group_idx:03d}"


def _template_family(subtype: str, group_n: int) -> str:
    return f"{FAMILY_ID}:{subtype}:template_{group_n % 8}"


def generate_grouped_crossover_split(
    *,
    split: str,
    n_per_group: int = 8,
    seed: int = 42,
    n_groups: int = N_GROUPS_TOTAL,
    fractions: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Generate one split with group-first assignment and balanced crossover."""
    assignments = _assign_groups_to_splits(
        n_groups, seed=seed, fractions=fractions)
    group_indices = assignments.get(split, [])
    if not group_indices:
        return []

    # Map group-level split to benchmark split name.
    bench_split = split
    if bench_split == "development":
        bench_split = "dev"
    valid_slots = SPLIT_TEMPLATE_SLOTS[bench_split]

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
            tid = f"{split}_{subtype}_g{gidx:03d}_{i:04d}"
            task_rng = random.Random(global_seed)
            # Balance parseable/unparseable within each group.
            force_parseable = (i % 2 == 0)
            slot = valid_slots[(i + group_n) % len(valid_slots)]
            body = _GENERATORS[subtype](task_rng, slot, force_parseable=force_parseable)
            spec = str(body.get("specification", ""))
            prompt_hash = hashlib.sha256(
                normalize_prompt(spec).encode("utf-8")).hexdigest()
            if prompt_hash in seen_prompts:
                for _retry in range(50):
                    retry_seed = (global_seed + _retry * 7919) % (2**31)
                    task_rng = random.Random(retry_seed)
                    retry_slot = valid_slots[(slot + _retry + 1) % len(valid_slots)]
                    body = _GENERATORS[subtype](
                        task_rng, retry_slot, force_parseable=force_parseable)
                    spec = str(body.get("specification", ""))
                    prompt_hash = hashlib.sha256(
                        normalize_prompt(spec).encode("utf-8")).hexdigest()
                    if prompt_hash not in seen_prompts:
                        break
                else:
                    continue
            seen_prompts.add(prompt_hash)
            variant = "parseable" if force_parseable else "unparseable"
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
                "variant": variant,
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
    "FAMILY_ID",
    "N_GROUPS_PER_SUBTYPE",
    "N_GROUPS_TOTAL",
    "SPLIT_FRACTIONS",
    "generate_grouped_crossover_split",
    "generate_all_grouped_splits",
    "group_assignments",
]
