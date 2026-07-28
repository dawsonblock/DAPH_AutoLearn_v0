from __future__ import annotations

from daph_learning.data.benchmark import (
    ALL_CATEGORIES,
    generate_protocol_splits,
)
from daph_learning.data.integrity import (
    detect_leakage,
    task_fingerprint,
)


def _small_splits():
    return generate_protocol_splits(
        {
            "train": 52,
            "dev": 26,
            "calibration": 13,
            "final_test": 13,
        },
        seed=1337,
    )


def test_generated_splits_are_template_disjoint_and_unique() -> None:
    splits = _small_splits()
    report = detect_leakage(splits)
    assert report.is_clean
    for rows in splits.values():
        assert {row["metadata"]["category"] for row in rows} == set(ALL_CATEGORIES)


def test_task_fingerprint_ignores_task_id() -> None:
    row = _small_splits()["train"][0]
    changed = {**row, "task_id": "different-id"}
    assert task_fingerprint(row) == task_fingerprint(changed)


def test_leakage_scanner_detects_family_overlap() -> None:
    splits = _small_splits()
    leaked = {name: list(rows) for name, rows in splits.items()}
    copied = dict(leaked["train"][0])
    copied["task_id"] = "dev-leaked-family"
    copied["specification"] = copied["specification"] + " extra wording"
    leaked["dev"].append(copied)
    report = detect_leakage(leaked)
    assert not report.is_clean
    assert any(item.kind == "split_family" for item in report.cross_split)
