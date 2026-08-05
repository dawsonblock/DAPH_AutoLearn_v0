"""v0.3.10.3.2-alpha — Section 10 / G12: linguistic template split
disjointness.

Verifies that splits are disjoint by actual linguistic template ID,
not just slot numbers.
"""

from __future__ import annotations

from daph_learning.data.crossover_benchmark import generate_crossover_split


def _template_ids(tasks: list) -> set[str]:
    return {t["metadata"]["linguistic_template_id"] for t in tasks}


def test_linguistic_template_ids_are_present():
    tasks = generate_crossover_split(split="train", n_per_subtype=5, seed=1)
    for t in tasks:
        assert "linguistic_template_id" in t["metadata"], (
            f"task {t['task_id']!r} missing linguistic_template_id")
        assert t["metadata"]["linguistic_template_id"].startswith("ling:")


def test_train_dev_template_disjoint():
    train = generate_crossover_split(split="train", n_per_subtype=10, seed=1)
    dev = generate_crossover_split(split="dev", n_per_subtype=10, seed=1)
    train_ids = _template_ids(train)
    dev_ids = _template_ids(dev)
    overlap = train_ids & dev_ids
    assert not overlap, (
        f"train/dev linguistic template overlap: {overlap}")


def test_train_calibration_template_disjoint():
    train = generate_crossover_split(split="train", n_per_subtype=10, seed=1)
    cal = generate_crossover_split(split="calibration", n_per_subtype=10, seed=1)
    train_ids = _template_ids(train)
    cal_ids = _template_ids(cal)
    overlap = train_ids & cal_ids
    assert not overlap, (
        f"train/calibration linguistic template overlap: {overlap}")


def test_train_final_template_disjoint():
    train = generate_crossover_split(split="train", n_per_subtype=10, seed=1)
    final = generate_crossover_split(split="final", n_per_subtype=10, seed=1)
    train_ids = _template_ids(train)
    final_ids = _template_ids(final)
    overlap = train_ids & final_ids
    assert not overlap, (
        f"train/final linguistic template overlap: {overlap}")


def test_same_numbers_different_wording_different_template():
    """Two prompts with the same numbers but different wording should
    get different linguistic template IDs."""
    from daph_learning.data.crossover_benchmark import linguistic_template_id
    p1 = "Compute 42 + 17. Return only the integer."
    p2 = "Please compute 42 + 17. Reply with the exact integer only."
    assert linguistic_template_id(p1) != linguistic_template_id(p2)


def test_different_numbers_same_wording_same_template():
    """Two prompts with different numbers but same wording should get
    the same linguistic template ID."""
    from daph_learning.data.crossover_benchmark import linguistic_template_id
    p1 = "Compute 42 + 17. Return only the integer."
    p2 = "Compute 999 + 3. Return only the integer."
    assert linguistic_template_id(p1) == linguistic_template_id(p2)
