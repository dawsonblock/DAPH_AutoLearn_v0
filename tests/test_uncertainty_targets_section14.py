"""Section 14 — uncertainty-aware targets tests."""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.policy.targets import (
    TargetMode,
    TrainingTargets,
    build_uncertainty_aware_targets,
    estimate_sigma,
)


def test_target_mode_has_new_modes():
    assert TargetMode.HARD_GAP.value == "hard_gap"
    assert TargetMode.SOFT_TEMPERATURE.value == "soft_temperature"
    assert TargetMode.SIGNAL_TO_NOISE.value == "signal_to_noise"


def test_from_str_accepts_new_modes():
    assert TargetMode.from_str("hard_gap") == TargetMode.HARD_GAP
    assert TargetMode.from_str("soft_temperature") == TargetMode.SOFT_TEMPERATURE
    assert TargetMode.from_str("signal_to_noise") == TargetMode.SIGNAL_TO_NOISE


def test_estimate_sigma_single_replicate():
    u_s = np.array([1.0, 0.5, -0.2])
    u_l = np.array([0.0, 0.3, 0.1])
    sigma = estimate_sigma(u_s, u_l)
    assert sigma.shape == (3,)
    # σ_i proportional to |ΔU|
    assert sigma[0] > sigma[1]  # |1.0| > |0.2|


def test_signal_to_noise_targets():
    delta_u = np.array([1.0, 0.01, -0.5])
    sigma = np.array([0.1, 0.5, 0.2])
    result = build_uncertainty_aware_targets(
        delta_u, sigma, mode=TargetMode.SIGNAL_TO_NOISE,
        temperature=1.0, w_max=1.0)
    assert isinstance(result, TrainingTargets)
    assert result.targets.shape == (3,)
    assert result.mask.all()  # all valid
    # High SNR example (|ΔU|=1.0, σ=0.1) → weight = min(1, 10) = 1.0
    assert result.weights[0] == pytest.approx(1.0)
    # Low SNR example (|ΔU|=0.01, σ=0.5) → weight = min(1, 0.02) = 0.02
    assert result.weights[1] < 0.1
    assert result.mode == TargetMode.SIGNAL_TO_NOISE


def test_signal_to_noise_downweights_uncertain():
    """Examples with high uncertainty relative to gap get low weight."""
    delta_u = np.array([0.5, 0.5])
    sigma = np.array([0.01, 1.0])  # second example very uncertain
    result = build_uncertainty_aware_targets(
        delta_u, sigma, mode=TargetMode.SIGNAL_TO_NOISE)
    assert result.weights[0] > result.weights[1]


def test_hard_gap_mode():
    delta_u = np.array([0.5, 0.01, -0.5])
    result = build_uncertainty_aware_targets(
        delta_u, np.ones(3), mode=TargetMode.HARD_GAP,
        gap_threshold=0.02)
    assert isinstance(result, TrainingTargets)
    assert result.targets[0] == 1.0  # ΔU > gap
    assert result.targets[2] == 0.0  # ΔU < -gap
    assert not result.mask[1]  # |ΔU| <= gap → masked
    # Weights are all 1.0 for hard_gap mode.
    assert np.all(result.weights == 1.0)
    assert result.mode == TargetMode.HARD_GAP


def test_soft_temperature_mode():
    delta_u = np.array([1.0, -1.0])
    result = build_uncertainty_aware_targets(
        delta_u, np.ones(2), mode=TargetMode.SOFT_TEMPERATURE,
        temperature=2.0)
    assert isinstance(result, TrainingTargets)
    assert result.mask.all()
    assert 0.5 < result.targets[0] < 1.0  # sigmoid(0.5)
    assert 0.0 < result.targets[1] < 0.5  # sigmoid(-0.5)
    assert np.all(result.weights == 1.0)
    assert result.mode == TargetMode.SOFT_TEMPERATURE


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_uncertainty_aware_targets(
            np.array([1.0]), np.array([0.1]), mode="bogus")


def test_consistent_return_type_all_modes():
    """All modes return TrainingTargets — no tuple-length branching."""
    delta_u = np.array([0.5, -0.5, 0.01])
    sigma = np.array([0.1, 0.1, 0.1])
    for mode in (TargetMode.SIGNAL_TO_NOISE, TargetMode.HARD_GAP,
                 TargetMode.SOFT_TEMPERATURE):
        result = build_uncertainty_aware_targets(
            delta_u, sigma, mode=mode, gap_threshold=0.02, temperature=1.0)
        assert isinstance(result, TrainingTargets)
        assert result.targets.shape == (3,)
        assert result.mask.shape == (3,)
        assert result.weights.shape == (3,)
        assert result.utility_gaps.shape == (3,)
