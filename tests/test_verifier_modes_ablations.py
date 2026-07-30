"""v0.3.10.3.2-alpha — Section 37-43 / G37: verifier modes + ablations.

Verifies that verifier modes are explicit and recorded, and that the
ablation harnesses (capture ablation, policy comparison) work
correctly with the canonical utility.
"""

from __future__ import annotations

import pytest

from daph_learning.execution.real_backends import (
    VerifierMode,
    VerificationResult,
    verify_arithmetic,
)


def test_verifier_modes_are_explicit():
    """Section 37: verifier modes must be explicit and recorded."""
    assert VerifierMode.EXACT.value == "exact"
    assert VerifierMode.NUMERIC_EXTRACTOR.value == "numeric_extractor"
    assert VerifierMode.TOKEN_EXTRACTOR.value == "token_extractor"
    assert VerifierMode.FINAL_MARKER.value == "final_marker"


def test_verification_result_records_mode():
    """Section 37: every verification result records its mode."""
    task = {"expected": 42, "specification": "Compute 6 * 7."}
    vr = verify_arithmetic(task, "42")
    assert vr.verifier_mode in [m.value for m in VerifierMode]
    assert vr.verified_correct is True


def test_exact_verification_correct():
    """Section 37: exact verification of correct output."""
    task = {"expected": 42, "specification": "Compute 6 * 7."}
    vr = verify_arithmetic(task, "42")
    assert vr.verified_correct is True


def test_exact_verification_incorrect():
    """Section 37: exact verification of incorrect output."""
    task = {"expected": 42, "specification": "Compute 6 * 7."}
    vr = verify_arithmetic(task, "43")
    assert vr.verified_correct is False


def test_numeric_extractor_handles_prose():
    """Section 37: numeric extractor handles prose-wrapped answers."""
    task = {"expected": 42, "specification": "Compute 6 * 7."}
    vr = verify_arithmetic(task, "The answer is 42.")
    assert vr.verified_correct is True


def test_numeric_extractor_handles_wrong_prose():
    """Section 37: numeric extractor rejects wrong prose answers."""
    task = {"expected": 42, "specification": "Compute 6 * 7."}
    vr = verify_arithmetic(task, "The answer is 43.")
    assert vr.verified_correct is False


def test_verification_of_none_output():
    """Section 37: verifying None output returns not verified."""
    task = {"expected": 42, "specification": "Compute 6 * 7."}
    vr = verify_arithmetic(task, None)
    assert vr.verified_correct is not True


def test_capture_ablation_forbids_final():
    """Section 28: capture ablation on FINAL is forbidden."""
    from daph_learning.evaluation.capture_ablation import run_capture_ablation
    with pytest.raises(ValueError, match="FINAL is forbidden"):
        run_capture_ablation(
            tasks=[],
            layers=[1, 2],
            policy_proba_fn=lambda t, l: 0.5,
            utility_fn=lambda t, r: 0.0,
            config=type("C", (), {"confidence_threshold": 0.7})(),
            split="final",
        )


def test_grouped_stats_effective_sample_size():
    """Section 31: effective sample size is computed correctly."""
    from daph_learning.evaluation.grouped_stats import effective_sample_size
    import numpy as np
    # Uniform weights → ESS = n.
    weights = np.ones(100)
    ess = effective_sample_size(weights)
    assert ess == pytest.approx(100.0, rel=0.01)
    # Concentrated weights → ESS < n.
    weights = np.zeros(100)
    weights[:10] = 1.0
    ess = effective_sample_size(weights)
    assert ess < 100.0


def test_grouped_bootstrap_ci():
    """Section 31: grouped bootstrap CI is computed."""
    from daph_learning.evaluation.grouped_stats import grouped_bootstrap_mean_delta
    import numpy as np
    # Build records with group_key, value_a, value_b.
    records = []
    a_vals = [0.8, 0.9, 0.7, 0.85, 0.95, 0.75, 0.8, 0.9, 0.85, 0.7]
    b_vals = [0.5, 0.6, 0.4, 0.55, 0.65, 0.45, 0.5, 0.6, 0.55, 0.4]
    groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    for i in range(10):
        records.append({
            "group": groups[i],
            "util_a": a_vals[i],
            "util_b": b_vals[i],
        })
    result = grouped_bootstrap_mean_delta(
        records, group_key="group",
        value_a_key="util_a", value_b_key="util_b",
        n_bootstrap=1000, seed=42)
    assert "ci_low" in result
    assert "ci_high" in result
    assert result["ci_low"] > 0  # A is clearly better than B
    assert result["ci_high"] > result["ci_low"]
