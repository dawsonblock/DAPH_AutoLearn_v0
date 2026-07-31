"""Section 10 — tests for empirical crossover audit (post-execution)."""

from __future__ import annotations

import pytest

from daph_learning.benchmark.audit import (
    EmpiricalCrossoverAudit,
    audit_empirical_crossover,
)


def _exp(delta_u, preferred, subtype="A"):
    return {
        "delta_utility": delta_u,
        "preferred_action": preferred,
        "subtype": subtype,
    }


def test_empty_experiences():
    audit = audit_empirical_crossover([])
    assert not audit.valid
    assert "no experiences" in audit.errors[0]


def test_all_decisive():
    exps = [
        _exp(0.5, "symbolic", "A"),
        _exp(-0.5, "llm", "A"),
        _exp(0.3, "symbolic", "B"),
        _exp(-0.3, "llm", "B"),
        _exp(0.4, "symbolic", "C"),
        _exp(-0.4, "llm", "C"),
    ]
    audit = audit_empirical_crossover(
        exps, min_decisive_fraction=0.5, min_crossover_subtypes=3)
    assert audit.n_decisive == 6
    assert audit.decisive_fraction == 1.0
    assert audit.valid


def test_low_decisive_fraction():
    exps = [_exp(0.001, "symbolic"), _exp(0.001, "llm"), _exp(0.5, "symbolic")]
    audit = audit_empirical_crossover(exps, min_decisive_fraction=0.5)
    assert not audit.valid
    assert any("decisive" in e for e in audit.errors)


def test_crossover_subtypes():
    exps = [
        _exp(0.5, "symbolic", "A"),
        _exp(-0.5, "llm", "A"),
        _exp(0.5, "symbolic", "B"),
        _exp(-0.5, "llm", "B"),
        _exp(0.5, "symbolic", "C"),
        _exp(-0.5, "llm", "C"),
    ]
    audit = audit_empirical_crossover(
        exps, min_crossover_subtypes=3, min_backend_win_fraction=0.2)
    assert audit.valid
    assert len(audit.crossover_subtypes) == 3


def test_no_crossover():
    exps = [
        _exp(0.5, "symbolic", "A"),
        _exp(0.5, "symbolic", "A"),
        _exp(0.5, "symbolic", "B"),
        _exp(0.5, "symbolic", "B"),
    ]
    audit = audit_empirical_crossover(
        exps, min_crossover_subtypes=1, min_backend_win_fraction=0.2)
    assert not audit.valid
    assert len(audit.crossover_subtypes) == 0


def test_to_dict():
    audit = audit_empirical_crossover([_exp(0.5, "symbolic")])
    d = audit.to_dict()
    assert "valid" in d
    assert "n_tasks" in d
    assert "decisive_fraction" in d
    assert "crossover_subtypes" in d
