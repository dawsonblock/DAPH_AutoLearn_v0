"""v0.3.10.3.2-alpha — Section 12-17 / G14: within-subtype crossover.

The central scientific repair: at least 3 subtypes must individually
contain both symbolic-preferred and LLM-preferred examples, with the
optimal backend emerging from actual executed utilities.
"""

from __future__ import annotations

import pytest

from daph_learning.data.crossover_benchmark import (
    generate_crossover_split,
    crossover_optimal_distribution,
)


def test_within_subtype_crossover_exists():
    """Section 12, 17: at least 3 subtypes must have both symbolic and
    LLM wins (within-subtype crossover)."""
    tasks = generate_crossover_split(
        split="train", n_per_subtype=30, seed=42)
    dist = crossover_optimal_distribution(tasks)
    crossover = dist["crossover_subtypes"]
    assert len(crossover) >= 3, (
        f"only {len(crossover)} crossover subtypes: {crossover} — "
        f"need at least 3 (Section 12). per_subtype_summary="
        f"{dist['per_subtype_summary']}")


def test_crossover_subtypes_have_balanced_wins():
    """Section 17: for crossover subtypes, 20% <= symbolic_win_fraction
    <= 80%."""
    tasks = generate_crossover_split(
        split="train", n_per_subtype=30, seed=42)
    dist = crossover_optimal_distribution(tasks)
    for subtype in dist["crossover_subtypes"]:
        summary = dist["per_subtype_summary"][subtype]
        p_sym = summary["p_symbolic"]
        assert 0.20 <= p_sym <= 0.80, (
            f"subtype {subtype}: p_symbolic={p_sym:.2f} outside "
            f"[0.20, 0.80] (Section 17)")


def test_optimal_backend_not_in_metadata():
    """Section 11: the optimal backend must never be stored in task
    metadata — it is derived from executed utility."""
    from daph_learning.data.crossover_benchmark import FORBIDDEN_METADATA_FIELDS
    tasks = generate_crossover_split(
        split="train", n_per_subtype=10, seed=1)
    for t in tasks:
        for field in FORBIDDEN_METADATA_FIELDS:
            assert field not in t, (
                f"task {t['task_id']!r} contains forbidden field {field!r}")
            assert field not in t.get("metadata", {}), (
                f"task {t['task_id']!r} metadata contains forbidden field "
                f"{field!r}")


def test_both_backends_available_on_required_fraction():
    """Section 13: >= 70% of qualification tasks must have both
    backends available."""
    from daph_learning.execution.real_backends import execute_symbolic_backend
    tasks = generate_crossover_split(
        split="train", n_per_subtype=20, seed=42)
    n_both = 0
    for t in tasks:
        sym_outcome, _ = execute_symbolic_backend(t)
        # LLM is always "available" in the simulator.
        if sym_outcome.available:
            n_both += 1
    fraction = n_both / len(tasks)
    assert fraction >= 0.70, (
        f"only {fraction:.1%} of tasks have both backends available — "
        f"need >= 70% (Section 13)")


def test_crossover_distribution_has_per_subtype_summary():
    """Section 17: the distribution must include per-subtype summary
    with crossover flags."""
    tasks = generate_crossover_split(
        split="train", n_per_subtype=10, seed=1)
    dist = crossover_optimal_distribution(tasks)
    assert "per_subtype_summary" in dist
    assert "crossover_subtypes" in dist
    assert "n_crossover_subtypes" in dist
    for subtype, summary in dist["per_subtype_summary"].items():
        assert "symbolic" in summary
        assert "llm" in summary
        assert "tie" in summary
        assert "p_symbolic" in summary
        assert "has_crossover" in summary
