"""Section 10 — dataset leakage audit tests."""

from __future__ import annotations

import pytest

from daph_learning.benchmark.audit import (
    DatasetAudit,
    DatasetAuditError,
    assert_dataset_clean,
    audit_dataset,
)
from daph_learning.data.grouped_benchmark import generate_all_grouped_splits


def _task(tid, spec, *, split="train", group_id="g1",
          template_family="fam1", subtype="A"):
    return {
        "task_id": tid,
        "specification": spec,
        "metadata": {"split": split, "group_id": group_id,
                     "template_family": template_family, "subtype": subtype},
    }


def test_clean_dataset_passes():
    splits = {
        "train": [_task("t1", "Compute 1 + 1.", group_id="g1",
                        template_family="f1")],
        "final": [_task("t2", "Compute 2 * 3.", group_id="g2",
                        template_family="f2")],
    }
    audit = audit_dataset(splits)
    assert audit.valid
    assert audit.errors == []


def test_exact_duplicate_detected():
    splits = {
        "train": [
            _task("t1", "Compute 1 + 1.", group_id="g1",
                  template_family="f1"),
            _task("t2", "Compute 1 + 1.", group_id="g2",
                  template_family="f2"),
        ],
    }
    audit = audit_dataset(splits)
    assert not audit.valid
    assert any("exact prompt duplicate" in e for e in audit.errors)


def test_normalized_duplicate_detected():
    splits = {
        "train": [
            _task("t1", "Compute  1 + 1.", group_id="g1",
                  template_family="f1"),
            _task("t2", "Compute 1 + 1.", group_id="g2",
                  template_family="f2"),
        ],
    }
    audit = audit_dataset(splits)
    assert not audit.valid
    assert any("normalized prompt duplicate" in e for e in audit.errors)


def test_cross_split_group_leak_detected():
    splits = {
        "train": [_task("t1", "Compute 1 + 1.", group_id="g1",
                        template_family="f1")],
        "final": [_task("t2", "Compute 2 * 3.", group_id="g1",
                        template_family="f2")],
    }
    audit = audit_dataset(splits)
    assert not audit.valid
    assert any("crosses splits" in e for e in audit.errors)


def test_template_family_leak_detected():
    splits = {
        "train": [_task("t1", "Compute 1 + 1.", group_id="g1",
                        template_family="fam1")],
        "final": [_task("t2", "Compute 2 * 3.", group_id="g2",
                        template_family="fam1")],
    }
    audit = audit_dataset(splits)
    assert not audit.valid
    assert any("template_family" in e for e in audit.errors)


def test_minimum_groups_enforced():
    splits = {
        "final": [_task("t1", "Compute 1 + 1.", group_id="g1",
                        template_family="f1")],
    }
    audit = audit_dataset(
        splits, min_groups_per_split={"final": 8})
    assert not audit.valid
    assert any("minimum required is 8" in e for e in audit.errors)


def test_assert_dataset_clean_raises():
    splits = {
        "train": [
            _task("t1", "Compute 1 + 1.", group_id="g1",
                  template_family="f1"),
            _task("t2", "Compute 1 + 1.", group_id="g2",
                  template_family="f2"),
        ],
    }
    audit = audit_dataset(splits)
    with pytest.raises(DatasetAuditError):
        assert_dataset_clean(audit)


def test_assert_dataset_clean_passes_on_clean():
    splits = {
        "train": [_task("t1", "Compute 1 + 1.", group_id="g1",
                        template_family="f1")],
        "final": [_task("t2", "Compute 2 * 3.", group_id="g2",
                        template_family="f2")],
    }
    audit = audit_dataset(splits)
    assert_dataset_clean(audit)  # no raise


def test_crossover_insufficient_subtypes():
    crossover = {
        "crossover_subtypes": ["A"],  # only 1
        "per_subtype_summary": {"A": {"has_crossover": True, "p_symbolic": 0.5}},
        "n_decisive": 100, "n": 100,
    }
    splits = {
        "train": [_task("t1", "Compute 1 + 1.", group_id="g1",
                        template_family="f1")],
    }
    audit = audit_dataset(splits, crossover_distribution=crossover,
                          min_crossover_subtypes=3)
    assert not audit.valid
    assert any("crossover subtypes" in e for e in audit.errors)


def test_crossover_minority_win_fraction_too_low():
    crossover = {
        "crossover_subtypes": ["A", "B", "C"],
        "per_subtype_summary": {
            "A": {"has_crossover": True, "p_symbolic": 0.95},
            "B": {"has_crossover": True, "p_symbolic": 0.5},
            "C": {"has_crossover": True, "p_symbolic": 0.5},
        },
        "n_decisive": 100, "n": 100,
    }
    splits = {
        "train": [_task("t1", "Compute 1 + 1.", group_id="g1",
                        template_family="f1")],
    }
    audit = audit_dataset(splits, crossover_distribution=crossover,
                          min_backend_win_fraction=0.20)
    assert not audit.valid
    assert any("minority backend win fraction" in e for e in audit.errors)


def test_decisive_fraction_too_low():
    crossover = {
        "crossover_subtypes": ["A", "B", "C"],
        "per_subtype_summary": {
            "A": {"has_crossover": True, "p_symbolic": 0.5},
            "B": {"has_crossover": True, "p_symbolic": 0.5},
            "C": {"has_crossover": True, "p_symbolic": 0.5},
        },
        "n_decisive": 10, "n": 100,  # 10% decisive
    }
    splits = {
        "train": [_task("t1", "Compute 1 + 1.", group_id="g1",
                        template_family="f1")],
    }
    audit = audit_dataset(splits, crossover_distribution=crossover,
                          min_decisive_fraction=0.35)
    assert not audit.valid
    assert any("decisive fraction" in e for e in audit.errors)


def test_generated_grouped_dataset_passes_audit():
    """The actual grouped benchmark from Section 9 should pass a clean
    audit (no leakage, no duplicates)."""
    splits = generate_all_grouped_splits(n_per_group=4)
    audit = audit_dataset(splits)
    assert audit.valid, f"audit errors: {audit.errors}"
    assert audit.cross_split_group_leaks == []
    assert audit.normalized_duplicates == []


def test_to_dict_serializable():
    splits = {
        "train": [_task("t1", "Compute 1 + 1.", group_id="g1",
                        template_family="f1")],
    }
    audit = audit_dataset(splits)
    d = audit.to_dict()
    assert isinstance(d, dict)
    assert "valid" in d
    assert "errors" in d
