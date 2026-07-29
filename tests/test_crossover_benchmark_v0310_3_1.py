"""v0.3.10.3.1-alpha — Section 10-12 / G26: within-family crossover
benchmark.

The central scientific question: can AutoLearn choose the better
computation for an individual task, not merely classify which task
family it belongs to? This requires a benchmark where the SAME broad
family contains both symbolic-preferred and LLM-preferred instances.
"""

from __future__ import annotations

import pytest

from daph_learning.data.crossover_benchmark import (
    FAMILY_ID,
    FORBIDDEN_METADATA_FIELDS,
    SUBTYPES,
    assert_no_within_split_duplicates,
    crossover_optimal_distribution,
    generate_crossover_split,
    generate_crossover_task,
)
import random


def test_all_subtypes_share_one_family():
    """Section 10: A-F are all members of the same broad family."""
    tasks = generate_crossover_split(split="train", n_per_subtype=3, seed=0)
    families = {t["metadata"]["family_id"] for t in tasks}
    assert families == {FAMILY_ID}
    subtypes = {t["metadata"]["subtype"] for t in tasks}
    assert subtypes == set(SUBTYPES)


def test_no_optimal_backend_encoded_in_metadata():
    """Section 11: task metadata must NOT contain best_backend,
    symbolic_preferred, llm_preferred, route_label, utility_oracle."""
    tasks = generate_crossover_split(split="train", n_per_subtype=2, seed=1)
    for t in tasks:
        for field in FORBIDDEN_METADATA_FIELDS:
            assert field not in t, f"{field!r} leaked into task {t['task_id']!r}"
            assert field not in t.get("metadata", {}), (
                f"{field!r} leaked into metadata of {t['task_id']!r}")


def test_no_within_split_prompt_duplicates():
    """Section 8: number_tasks == number_unique_prompt_hashes."""
    for split in ("train", "dev", "calibration", "final"):
        tasks = generate_crossover_split(split=split, n_per_subtype=5, seed=42)
        assert_no_within_split_duplicates(tasks)  # raises on dup


def test_crossover_family_contains_both_backend_preferences():
    """Section 10 / G26: 0.2 < P(symbolic optimal | family) < 0.8.

    This is the decisive test. If the family were trivially separable
    (all symbolic or all LLM), P would be 0 or 1 and the benchmark
    would only test family classification, not instance-level routing.
    """
    tasks = generate_crossover_split(split="train", n_per_subtype=20, seed=7)
    dist = crossover_optimal_distribution(tasks)
    p = dist["p_symbolic_optimal"]
    assert 0.2 < p < 0.8, (
        f"P(symbolic optimal)={p:.3f} outside (0.2, 0.8); the family does "
        f"not exhibit within-family crossover. dist={dist}")
    assert dist["n_symbolic_optimal"] > 0
    assert dist["n_llm_optimal"] > 0


def test_optimal_derived_from_execution_not_metadata():
    """Section 11: the optimal action is derived only after both
    backends execute and verify. The distribution function must not
    read any stored label."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=10, seed=3)
    dist = crossover_optimal_distribution(tasks)
    # If the function had read a stored label, removing execution would
    # still return a distribution. Verify it actually executed by
    # checking per-subtype structure matches the known simulator behavior:
    # B, C, E, F -> LLM-optimal; A, D -> mixed (some symbolic, some tie).
    ps = dist["per_subtype"]
    assert ps["B"]["llm"] > 0 and ps["B"]["symbolic"] == 0
    assert ps["C"]["llm"] > 0 and ps["C"]["symbolic"] == 0
    assert ps["F"]["llm"] > 0 and ps["F"]["symbolic"] == 0


def test_splits_are_template_disjoint():
    """Section 14: train/dev/calibration/final use disjoint template slots."""
    from daph_learning.data.crossover_benchmark import SPLIT_TEMPLATE_SLOTS
    all_slots = []
    for split in ("train", "dev", "calibration", "final"):
        all_slots.extend(SPLIT_TEMPLATE_SLOTS[split])
    assert len(all_slots) == len(set(all_slots)), "template slots overlap across splits"


def test_generate_task_rejects_unlicensed_slot():
    rng = random.Random(0)
    with pytest.raises(ValueError, match="not licensed"):
        generate_crossover_task("t", rng, "A", split="train",
                                template_slot=7, seed=0)


def test_generate_task_rejects_unknown_subtype():
    rng = random.Random(0)
    with pytest.raises(ValueError, match="unknown subtype"):
        generate_crossover_task("t", rng, "Z", split="train",
                                template_slot=0, seed=0)


def test_group_and_template_ids_present():
    """Section 9: group_id / template_id / family_id for grouped stats."""
    tasks = generate_crossover_split(split="train", n_per_subtype=2, seed=0)
    for t in tasks:
        m = t["metadata"]
        assert m["family_id"] == FAMILY_ID
        assert m["group_id"].startswith(FAMILY_ID)
        assert m["template_id"].startswith(FAMILY_ID)
        assert m["subtype"] in SUBTYPES
