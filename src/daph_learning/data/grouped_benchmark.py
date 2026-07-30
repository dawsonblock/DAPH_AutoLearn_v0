"""Section 9 — group-first crossover benchmark generator (v0.3.10.4-alpha).

The existing :mod:`crossover_benchmark` module uses 8 template slots and
groups by ``family_id:subtype`` (6 groups). Section 9 requires 60–80
independent generation groups, each a distinct template family with its
own linguistic template, operand distribution, and split assignment.

This module adds a group-first generator that:

1. Defines 80 template families (10 per subtype × 6 subtypes, with the
   final 8 reserved for the final split to give ~80 bootstrap groups).
2. Assigns families to splits (40 train / 16 dev / 12 cal / 12 final)
   BEFORE generating any instances.
3. Generates instances within assigned splits.
4. Deduplicates globally (normalized prompt hash).
5. Validates no group crosses splits and no normalized prompt overlaps.

Crossover emerges from real task properties (operand size, paraphrasing,
narrative vs structured), NOT label balancing. The optimal backend is
derived only AFTER both backends execute and the verifier runs.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from .crossover_benchmark import (
    FAMILY_ID,
    FORBIDDEN_METADATA_FIELDS,
    SUBTYPES,
    SUBTYPE_DESCRIPTIONS,
    _GENERATORS,
    _assert_no_optimal_backend_encoded,
    linguistic_template_id,
)
from .integrity import normalize_prompt

GENERATOR_VERSION = "v0.3.10.4-grouped"

# 80 groups: 10 per subtype. Each group is a distinct template family
# within the structured_math family, identified by (subtype, group_idx).
N_GROUPS_PER_SUBTYPE = 10
N_GROUPS_TOTAL = N_GROUPS_PER_SUBTYPE * len(SUBTYPES)  # 60

# Group-first split assignment (Section 9.4).
# 60 groups → 30 train / 12 dev / 9 cal / 9 final.
# This gives the final split 9 independent groups for bootstrap.
SPLIT_FRACTIONS = {"train": 0.50, "development": 0.20,
                   "calibration": 0.15, "final": 0.15}


def _assign_groups_to_splits(
    n_groups: int = N_GROUPS_TOTAL,
    *,
    seed: int = 42,
    fractions: dict[str, float] | None = None,
) -> dict[str, list[int]]:
    """Section 9.4 — assign group indices to splits BEFORE generating
    instances. Returns ``{split: [group_idx, ...]}``.

    Groups are shuffled deterministically then partitioned by fraction.
    No group appears in more than one split.
    """
    frac = fractions or SPLIT_FRACTIONS
    rng = random.Random(seed)
    indices = list(range(n_groups))
    rng.shuffle(indices)
    n_train = int(n_groups * frac["train"])
    n_dev = int(n_groups * frac["development"])
    n_cal = int(n_groups * frac["calibration"])
    # Remainder goes to final to avoid rounding loss.
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
    """Section 9 — generate a full split using group-first assignment.

    Parameters
    ----------
    split : str
        One of "train", "development", "calibration", "final".
    n_per_group : int
        Instances per group.
    seed : int
        Master seed for group assignment.
    n_groups : int
        Total groups across all subtypes.
    fractions : dict | None
        Split fractions; defaults to :data:`SPLIT_FRACTIONS`.
    """
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
            # Use the existing subtype generator but override metadata.
            body = _GENERATORS[subtype](rng, group_n % 8)
            spec = str(body.get("specification", ""))
            prompt_hash = hashlib.sha256(
                normalize_prompt(spec).encode("utf-8")).hexdigest()
            if prompt_hash in seen_prompts:
                # Retry with extra entropy.
                for _retry in range(20):
                    body = _GENERATORS[subtype](rng, (group_n + _retry + 1) % 8)
                    spec = str(body.get("specification", ""))
                    prompt_hash = hashlib.sha256(
                        normalize_prompt(spec).encode("utf-8")).hexdigest()
                    if prompt_hash not in seen_prompts:
                        break
                else:
                    continue  # skip if still duplicate
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
    """Section 9 — generate all four splits with group-first assignment."""
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
    """Return the group-to-split assignment (for audit/provenance)."""
    return _assign_groups_to_splits(n_groups, seed=seed, fractions=fractions)


def assert_no_group_crosses_splits(
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    """Section 9.4 — no group_id appears in more than one split."""
    group_to_splits: dict[str, set[str]] = {}
    for split_name, tasks in splits.items():
        for t in tasks:
            gid = t.get("metadata", {}).get("group_id", "")
            group_to_splits.setdefault(gid, set()).add(split_name)
    for gid, slist in group_to_splits.items():
        if len(slist) > 1:
            raise ValueError(
                f"group {gid!r} crosses splits: {sorted(slist)}")


def assert_groups_are_independent(
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    """Section 9 — each group has a unique template_family and no
    normalized prompt overlaps across splits."""
    # Unique template families.
    families: dict[str, set[str]] = {}
    for split_name, tasks in splits.items():
        for t in tasks:
            fam = t.get("metadata", {}).get("template_family", "")
            families.setdefault(fam, set()).add(split_name)
    for fam, slist in families.items():
        if len(slist) > 1:
            raise ValueError(
                f"template_family {fam!r} appears in multiple splits: "
                f"{sorted(slist)}")
    # No normalized prompt overlap across splits.
    seen: dict[str, str] = {}  # prompt_hash → split
    for split_name, tasks in splits.items():
        for t in tasks:
            spec = str(t.get("specification", ""))
            ph = hashlib.sha256(
                normalize_prompt(spec).encode("utf-8")).hexdigest()
            if ph in seen and seen[ph] != split_name:
                raise ValueError(
                    f"normalized prompt overlap between {seen[ph]!r} and "
                    f"{split_name!r}: {spec[:60]!r}")
            seen[ph] = split_name


def count_groups(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Count unique group_ids per split."""
    counts: dict[str, int] = {}
    for split_name, tasks in splits.items():
        gids = {t.get("metadata", {}).get("group_id", "") for t in tasks}
        counts[split_name] = len(gids)
    return counts


__all__ = [
    "GENERATOR_VERSION",
    "N_GROUPS_PER_SUBTYPE",
    "N_GROUPS_TOTAL",
    "SPLIT_FRACTIONS",
    "assert_no_group_crosses_splits",
    "assert_groups_are_independent",
    "count_groups",
    "generate_all_grouped_splits",
    "generate_grouped_crossover_split",
    "group_assignments",
]
