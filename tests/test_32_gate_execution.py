"""v0.3.10.3.2-alpha — Section 48-50 / G48: 32-gate execution.

Verifies that the full G1-G32 gate registry is defined, all gates
have test files, and the gate registry is self-consistent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daph_learning.evaluation.release_gates import (
    GATE_REGISTRY,
    GateResult,
    ReleaseGate,
    ReleaseGateReport,
    all_gate_ids,
    get_gate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_gate_registry_has_32_gates():
    """Section 48: the registry must have exactly 32 gates (G1-G32)."""
    assert len(GATE_REGISTRY) == 32, (
        f"gate registry has {len(GATE_REGISTRY)} gates, expected 32")
    ids = all_gate_ids()
    assert ids == [f"G{i}" for i in range(1, 33)], (
        f"gate IDs are {ids}, expected G1-G32 in order")


def test_every_gate_has_test_files():
    """Section 48: every gate must have at least one test file."""
    for gate in GATE_REGISTRY:
        assert len(gate.test_files) > 0, (
            f"gate {gate.gate_id} has no test files")


def test_every_gate_test_file_exists():
    """Section 48: every gate's test files must exist in the repo."""
    for gate in GATE_REGISTRY:
        for tf in gate.test_files:
            path = REPO_ROOT / tf
            assert path.exists(), (
                f"gate {gate.gate_id} references non-existent test file: {tf}")


def test_every_gate_has_claims_supported():
    """Section 48: every gate must support at least one claim."""
    for gate in GATE_REGISTRY:
        assert len(gate.claims_supported) > 0, (
            f"gate {gate.gate_id} supports no claims")


def test_every_gate_has_section():
    """Section 48: every gate must reference a section."""
    for gate in GATE_REGISTRY:
        assert gate.section, f"gate {gate.gate_id} has no section"


def test_get_gate_by_id():
    """Section 48: get_gate returns the correct gate."""
    g = get_gate("G1")
    assert g.gate_id == "G1"
    assert "version" in g.description.lower()


def test_get_gate_unknown_raises():
    """Section 48: get_gate raises for unknown gate IDs."""
    with pytest.raises(KeyError, match="unknown gate"):
        get_gate("G99")


def test_gate_report_serialization():
    """Section 49: the gate report can be serialized to dict."""
    report = ReleaseGateReport(
        results=[
            GateResult(gate_id="G1", status="pass", n_tests=5),
            GateResult(gate_id="G2", status="pass", n_tests=3),
        ],
        n_pass=2, n_fail=0, n_skip=0, all_pass=True,
    )
    d = report.to_dict()
    assert d["n_pass"] == 2
    assert d["all_pass"] is True
    assert len(d["results"]) == 2


def test_gate_registry_covers_new_sections():
    """Section 48: the registry must cover the new v0.3.10.3.2 sections."""
    # G2: canonical source hash
    g2 = get_gate("G2")
    assert "canonical" in g2.description.lower() or "source" in g2.description.lower()
    # G6: benchmark arithmetic correctness
    g6 = get_gate("G6")
    assert "arithmetic" in g6.description.lower() or "truncation" in g6.description.lower()
    # G9: linguistic template disjointness
    g9 = get_gate("G9")
    assert "template" in g9.description.lower() or "disjoint" in g9.description.lower()
    # G31: within-subtype crossover
    g31 = get_gate("G31")
    assert "crossover" in g31.description.lower() or "subtype" in g31.description.lower()
    # G32: CLI path completion
    g32 = get_gate("G32")
    assert "cli" in g32.description.lower() or "path" in g32.description.lower()
