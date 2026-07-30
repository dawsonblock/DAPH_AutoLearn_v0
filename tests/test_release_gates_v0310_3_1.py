"""v0.3.10.3.1-alpha — Section 39-40 / G31-G32: release-gate claim
contract and G1-G32 gate registry."""

from __future__ import annotations

import pytest

from daph_learning.evaluation.release_gates import (
    GATE_REGISTRY,
    all_claims,
    all_gate_ids,
    gates_for_claim,
    get_gate,
)


def test_gate_registry_has_32_gates():
    """Section 40: G1-G32 all defined."""
    assert len(GATE_REGISTRY) == 32
    ids = all_gate_ids()
    assert ids == [f"G{i}" for i in range(1, 33)]


def test_every_gate_has_section_description_claims_tests():
    """Section 39: every gate is fully specified."""
    for g in GATE_REGISTRY:
        assert g.gate_id, "missing gate_id"
        assert g.section, f"{g.gate_id} missing section"
        assert g.description, f"{g.gate_id} missing description"
        assert len(g.claims_supported) > 0, f"{g.gate_id} supports no claims"
        assert len(g.test_files) > 0, f"{g.gate_id} has no test files"


def test_every_claim_has_at_least_one_gate():
    """Section 39: every claim is backed by at least one gate."""
    claims = all_claims()
    assert len(claims) > 0
    for c in claims:
        gates = gates_for_claim(c)
        assert len(gates) > 0, f"claim {c!r} has no backing gate"


def test_get_gate_by_id():
    g = get_gate("G9")
    assert g.gate_id == "G9"
    # v0.3.10.3.2: G9 is now "Linguistic template split disjointness"
    assert "template" in g.description.lower() or "disjoint" in g.description.lower()


def test_get_gate_unknown_raises():
    with pytest.raises(KeyError):
        get_gate("G99")


def test_gate_ids_are_unique():
    ids = [g.gate_id for g in GATE_REGISTRY]
    assert len(ids) == len(set(ids))


def test_gate_test_files_exist():
    """Section 39: every referenced test file must exist."""
    from pathlib import Path
    tests_dir = Path(__file__).resolve().parent
    repo = tests_dir.parent
    for g in GATE_REGISTRY:
        for tf in g.test_files:
            assert (repo / tf).exists(), (
                f"{g.gate_id} references non-existent test file {tf}")


def test_g32_all_gates_pass_is_self_referential():
    """v0.3.10.3.2: G32 is now 'CLI paths completed'. The all-gates-pass
    check is distributed across the gate registry itself."""
    g32 = get_gate("G32")
    assert "cli" in g32.description.lower() or "path" in g32.description.lower()
