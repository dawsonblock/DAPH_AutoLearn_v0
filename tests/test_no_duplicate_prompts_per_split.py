"""v0.3.10.3.2-alpha — Section 11 / G11: no within-split duplicates.

Verifies that each split has no duplicate prompts (by normalized
prompt hash).
"""

from __future__ import annotations

import hashlib

from daph_learning.data.crossover_benchmark import (
    generate_crossover_split,
    assert_no_within_split_duplicates,
)
from daph_learning.data.integrity import normalize_prompt


def _prompt_hashes(tasks: list) -> list[str]:
    hashes = []
    for t in tasks:
        spec = str(t.get("specification", ""))
        h = hashlib.sha256(normalize_prompt(spec).encode("utf-8")).hexdigest()
        hashes.append(h)
    return hashes


def test_train_no_duplicates():
    tasks = generate_crossover_split(split="train", n_per_subtype=20, seed=7)
    hashes = _prompt_hashes(tasks)
    assert len(hashes) == len(set(hashes)), (
        f"train split has {len(hashes) - len(set(hashes))} duplicate prompts")


def test_dev_no_duplicates():
    tasks = generate_crossover_split(split="dev", n_per_subtype=20, seed=7)
    hashes = _prompt_hashes(tasks)
    assert len(hashes) == len(set(hashes))


def test_calibration_no_duplicates():
    tasks = generate_crossover_split(
        split="calibration", n_per_subtype=20, seed=7)
    hashes = _prompt_hashes(tasks)
    assert len(hashes) == len(set(hashes))


def test_final_no_duplicates():
    tasks = generate_crossover_split(split="final", n_per_subtype=20, seed=7)
    hashes = _prompt_hashes(tasks)
    assert len(hashes) == len(set(hashes))


def test_assert_no_within_split_duplicates_passes():
    """The assert_no_within_split_duplicates function should not raise
    for a properly generated split."""
    tasks = generate_crossover_split(split="train", n_per_subtype=20, seed=7)
    assert_no_within_split_duplicates(tasks)  # should not raise


def test_cross_split_no_prompt_overlap():
    """Section 10: train and final should not share any prompt."""
    train = generate_crossover_split(split="train", n_per_subtype=20, seed=7)
    final = generate_crossover_split(split="final", n_per_subtype=20, seed=7)
    train_hashes = set(_prompt_hashes(train))
    final_hashes = set(_prompt_hashes(final))
    overlap = train_hashes & final_hashes
    assert not overlap, (
        f"train/final prompt overlap: {len(overlap)} shared prompts")
