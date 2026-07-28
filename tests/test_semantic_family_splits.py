"""V038-002 — semantic-family splits are template-disjoint.

Item 5: replace template OOD with semantic-family splits. The protocol
generator already holds out complete template families (via
``SPLIT_TEMPLATE_SLOTS``) and the leakage scanner treats any family overlap
as a protocol failure. These tests confirm the generated splits are
family-disjoint and that every split retains all 13 categories (so the
held-out split is a semantic-family OOD split, not a category-removal split).
"""

from __future__ import annotations

import random

from daph_learning.data.benchmark import (
    ALL_CATEGORIES,
    SPLIT_TEMPLATE_SLOTS,
    generate_task,
)
from daph_learning.data.integrity import (
    assert_no_leakage,
    detect_leakage,
    split_family,
)


def _generate_splits(seed=1337):
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {name: [] for name in SPLIT_TEMPLATE_SLOTS}
    # distribute categories across template slots licensed for each split
    for split, slots in SPLIT_TEMPLATE_SLOTS.items():
        for cat in ALL_CATEGORIES:
            for slot in slots:
                for i in range(2):  # a couple of tasks per (cat, slot)
                    tid = f"{split}:{cat}:{slot}:{i}"
                    task = generate_task(
                        tid, rng, cat, split=split, template_slot=slot,
                        seed=seed,
                    )
                    splits[split].append(task)
    return splits


def test_generated_splits_have_no_family_overlap():
    splits = _generate_splits()
    report = detect_leakage(splits)
    # no cross-split family (or any other) overlap
    family_overlaps = [o for o in report.cross_split if o.kind == "split_family"]
    assert family_overlaps == [], (
        f"family overlap across splits: {[o.to_dict() for o in family_overlaps]}"
    )
    assert report.is_clean


def test_assert_no_leakage_passes_on_generated_splits():
    splits = _generate_splits()
    assert_no_leakage(splits)  # must not raise


def test_every_split_retains_all_categories():
    """A semantic-family OOD split holds out templates, not categories."""
    splits = _generate_splits()
    for split_name, rows in splits.items():
        cats = {r["metadata"]["category"] for r in rows}
        assert cats == set(ALL_CATEGORIES), (
            f"split {split_name} missing categories: "
            f"{set(ALL_CATEGORIES) - cats}"
        )


def test_template_slots_are_disjoint_across_splits():
    """The licensed template slots for each split are pairwise disjoint."""
    names = list(SPLIT_TEMPLATE_SLOTS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert set(SPLIT_TEMPLATE_SLOTS[a]).isdisjoint(SPLIT_TEMPLATE_SLOTS[b]), (
                f"template slots overlap between {a} and {b}"
            )


def test_split_family_resolves_to_template_id():
    splits = _generate_splits()
    families = {split_family(r) for rows in splits.values() for r in rows}
    # every family is a template_id of the form "<category>:template_<slot>"
    assert all(":" in f and "template_" in f for f in families)
