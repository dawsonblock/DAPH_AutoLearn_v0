"""Section 9 — group-first crossover benchmark tests."""

from __future__ import annotations

import pytest

from daph_learning.data.grouped_benchmark import (
    GENERATOR_VERSION,
    N_GROUPS_TOTAL,
    assert_no_group_crosses_splits,
    assert_groups_are_independent,
    count_groups,
    generate_all_grouped_splits,
    generate_grouped_crossover_split,
    group_assignments,
)
from daph_learning.data.crossover_benchmark import FORBIDDEN_METADATA_FIELDS


def test_total_groups_is_420():
    assert N_GROUPS_TOTAL == 420


def test_group_assignment_covers_all_groups():
    a = group_assignments()
    all_idx = set()
    for gs in a.values():
        all_idx.update(gs)
    assert all_idx == set(range(420))


def test_no_group_in_multiple_splits():
    a = group_assignments()
    seen = set()
    for gs in a.values():
        for g in gs:
            assert g not in seen, f"group {g} in multiple splits"
            seen.add(g)


def test_split_proportions():
    a = group_assignments()
    assert len(a["train"]) == 210
    assert len(a["development"]) == 84
    assert len(a["calibration"]) == 63
    assert len(a["final"]) == 63


def test_generate_all_splits():
    splits = generate_all_grouped_splits(n_per_group=4)
    counts = count_groups(splits)
    assert counts["train"] == 210
    assert counts["development"] == 84
    assert counts["calibration"] == 63
    assert counts["final"] == 63


def test_no_group_crosses_splits():
    splits = generate_all_grouped_splits(n_per_group=4)
    assert_no_group_crosses_splits(splits)


def test_groups_are_independent():
    splits = generate_all_grouped_splits(n_per_group=4)
    assert_groups_are_independent(splits)


def test_no_forbidden_metadata_fields():
    splits = generate_all_grouped_splits(n_per_group=2)
    for split_name, tasks in splits.items():
        for t in tasks:
            for field in FORBIDDEN_METADATA_FIELDS:
                assert field not in t
                assert field not in t.get("metadata", {})


def test_every_task_has_group_id():
    splits = generate_all_grouped_splits(n_per_group=2)
    for split_name, tasks in splits.items():
        for t in tasks:
            assert t["metadata"]["group_id"]
            assert t["metadata"]["template_family"]
            assert t["metadata"]["generator"] == GENERATOR_VERSION


def test_final_split_has_at_least_8_groups():
    splits = generate_all_grouped_splits(n_per_group=4)
    assert count_groups(splits)["final"] >= 8


def test_single_split_generation():
    tasks = generate_grouped_crossover_split(
        split="final", n_per_group=3)
    assert len(tasks) > 0
    for t in tasks:
        assert t["metadata"]["split"] == "final"


def test_deterministic_with_same_seed():
    s1 = generate_all_grouped_splits(n_per_group=2, seed=42)
    s2 = generate_all_grouped_splits(n_per_group=2, seed=42)
    assert len(s1["final"]) == len(s2["final"])
    # Same task_ids
    ids1 = [t["task_id"] for t in s1["final"]]
    ids2 = [t["task_id"] for t in s2["final"]]
    assert ids1 == ids2


def test_different_seed_different_assignment():
    a1 = group_assignments(seed=42)
    a2 = group_assignments(seed=99)
    # Different seeds should produce different assignments (with high prob).
    assert a1["final"] != a2["final"]


def test_group_crosses_splits_detected():
    # Manually construct a bad splits dict.
    bad = {
        "train": [{"metadata": {"group_id": "g1"}}],
        "final": [{"metadata": {"group_id": "g1"}}],
    }
    with pytest.raises(ValueError, match="crosses splits"):
        assert_no_group_crosses_splits(bad)
