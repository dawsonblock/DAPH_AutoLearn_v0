"""Section 8 — frozen UtilityConfig tests."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from daph_learning.policy.types import BackendOutcome
from daph_learning.policy.utility import (
    GATE_A_ACCURACY_PRIMARY,
    GATE_A_COST_SENSITIVE_SECONDARY,
    PROTOCOL_PRESETS,
    UtilityConfig,
    compute_utility,
    get_protocol,
)


def _outcome(*, backend="symbolic", correct=True, execution_success=True,
             verifier_status="verified_correct", latency_sec=0.01,
             quality=1.0) -> BackendOutcome:
    return BackendOutcome(
        task_id="t", backend=backend, available=True, executed=True,
        execution_success=execution_success, output_text="42",
        output_hash="x", verifier_status=verifier_status,
        correct=correct, quality=quality, latency_sec=latency_sec,
        normalized_cost=0.05, risk=0.0, verifier_confidence=1.0,
    )


def test_utility_config_is_frozen():
    cfg = UtilityConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.accuracy_weight = 2.0  # type: ignore[misc]


def test_utility_config_hash_is_deterministic():
    cfg = UtilityConfig()
    assert cfg.utility_config_hash == cfg.utility_config_hash
    assert len(cfg.utility_config_hash) == 64


def test_utility_config_hash_changes_on_change():
    cfg1 = UtilityConfig(accuracy_weight=1.0)
    cfg2 = UtilityConfig(accuracy_weight=2.0)
    assert cfg1.utility_config_hash != cfg2.utility_config_hash


def test_two_protocols_have_different_hashes():
    assert (GATE_A_ACCURACY_PRIMARY.utility_config_hash
            != GATE_A_COST_SENSITIVE_SECONDARY.utility_config_hash)


def test_primary_has_smaller_compute_weight_than_secondary():
    assert (GATE_A_ACCURACY_PRIMARY.compute_weight
            < GATE_A_COST_SENSITIVE_SECONDARY.compute_weight)


def test_get_protocol_returns_presets():
    assert get_protocol("gate_a_accuracy_primary") is GATE_A_ACCURACY_PRIMARY
    assert (get_protocol("gate_a_cost_sensitive_secondary")
            is GATE_A_COST_SENSITIVE_SECONDARY)


def test_get_protocol_rejects_unknown():
    with pytest.raises(ValueError):
        get_protocol("gate_a_whatever_passes")


def test_compute_utility_correct_symbolic():
    o = _outcome(backend="symbolic", correct=True)
    u = compute_utility(o, GATE_A_ACCURACY_PRIMARY)
    assert u > 0.9  # high accuracy reward, tiny cost


def test_compute_utility_failed_executes_negative():
    o = _outcome(backend="llm", correct=False, execution_success=False,
                 verifier_status="not_verified", latency_sec=2.0)
    u = compute_utility(o, GATE_A_ACCURACY_PRIMARY)
    assert u == pytest.approx(-1.0)  # clipped to min


def test_compute_utility_clips_to_max():
    cfg = UtilityConfig(utility_clip_max=0.5)
    o = _outcome(correct=True, latency_sec=0.0)
    u = compute_utility(o, cfg)
    assert u == pytest.approx(0.5)


def test_compute_utility_clips_to_min():
    cfg = UtilityConfig(utility_clip_min=-0.5)
    o = _outcome(correct=False, execution_success=False,
                 verifier_status="not_verified", latency_sec=10.0)
    u = compute_utility(o, cfg)
    assert u == pytest.approx(-0.5)


def test_compute_utility_no_nan():
    cfg = UtilityConfig()
    o = _outcome(latency_sec=float("nan"))
    # BackendOutcome may reject NaN latency; if it does, that's fine.
    # If it accepts, compute_utility must raise.
    try:
        u = compute_utility(o, cfg)
        assert not math.isnan(u)
    except ValueError:
        pass


def test_compute_utility_symbolic_cheaper_than_llm():
    o_sym = _outcome(backend="symbolic", correct=True, latency_sec=0.01)
    o_llm = _outcome(backend="llm", correct=True, latency_sec=0.01)
    cfg = GATE_A_COST_SENSITIVE_SECONDARY
    u_sym = compute_utility(o_sym, cfg)
    u_llm = compute_utility(o_llm, cfg)
    assert u_sym > u_llm  # symbolic has lower compute cost


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        UtilityConfig(latency_normalizer_ms=0)
    with pytest.raises(ValueError):
        UtilityConfig(compute_cost_symbolic=-1)
    with pytest.raises(ValueError):
        UtilityConfig(utility_clip_min=1.0, utility_clip_max=-1.0)


def test_from_dict_roundtrip():
    cfg = UtilityConfig(accuracy_weight=2.0, compute_weight=0.5)
    d = cfg.to_dict()
    cfg2 = UtilityConfig.from_dict(d)
    assert cfg == cfg2
    assert cfg.utility_config_hash == cfg2.utility_config_hash


def test_protocol_presets_are_frozen():
    # The preset objects themselves must be immutable UtilityConfig instances.
    for name, cfg in PROTOCOL_PRESETS.items():
        assert isinstance(cfg, UtilityConfig)
        with pytest.raises((AttributeError, TypeError)):
            cfg.accuracy_weight = 99  # type: ignore[misc]
