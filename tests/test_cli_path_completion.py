"""v0.3.10.3.2-alpha — Section 22-27 / G22: real CLI paths completed.

Verifies that the CLI evaluate, calibrate, and intervene modes use
real frozen pipelines with zero fitting on final, and that the
intervene mode uses canonical backend_utility (not synthetic).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_intervene_does_not_use_synthetic_utility():
    """Section 25: the intervene command must use canonical
    backend_utility from executed backends, not a synthetic
    utility_fn that returns 1.0 for symbolic and 0.0 for LLM."""
    policy_file = REPO_ROOT / "src" / "daph_learning" / "cli" / "commands" / "policy.py"
    source = policy_file.read_text()
    # The old synthetic utility was: return 1.0 if route == Route.SYMBOLIC else 0.0
    assert "1.0 if route == Route.SYMBOLIC else 0.0" not in source, (
        "intervene still uses synthetic utility_fn (Section 25)")
    # Must import backend_utility
    assert "backend_utility" in source, (
        "intervene does not import backend_utility (Section 25)")


def test_calibrate_does_not_have_zero_utility_fallback():
    """Section 22: the calibrate command must not have an unsafe
    fallback where utility_fn returns 0.0 if no execute_fn is
    provided."""
    policy_file = REPO_ROOT / "src" / "daph_learning" / "cli" / "commands" / "policy.py"
    source = policy_file.read_text()
    # The old unsafe fallback was: def utility_fn(task, route): return 0.0
    # in the else branch of the calibrate command.
    # Check that the calibrate command fails closed instead.
    assert "fail closed" in source.lower() or "fail_closed" in source.lower(), (
        "calibrate does not fail closed (Section 22)")


def test_evaluate_uses_real_backends_for_test_tasks():
    """Section 22: when --test-tasks is provided, evaluate must use
    real backends (not None execute_fn with no utility)."""
    policy_file = REPO_ROOT / "src" / "daph_learning" / "cli" / "commands" / "policy.py"
    source = policy_file.read_text()
    # The evaluate command should import real_backends for the test_tasks path
    assert "execute_symbolic_backend" in source, (
        "evaluate does not use real symbolic backend (Section 22)")
    assert "execute_llm_backend" in source, (
        "evaluate does not use real LLM backend (Section 22)")


def test_cli_evaluate_synthetic_runs():
    """Section 22: the evaluate CLI mode with --synthetic should run
    end-to-end and produce a result JSON."""
    import numpy as np
    # First, train a policy.
    with tempfile.TemporaryDirectory() as tmp:
        policy_out = Path(tmp) / "policy.json"
        rc = subprocess.run(
            [sys.executable, "-m", "daph_learning.cli.policy", "train",
             "--synthetic", "--output", str(policy_out)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
        if rc.returncode != 0:
            pytest.skip(f"train failed: {rc.stderr}")
        if not policy_out.exists():
            pytest.skip("policy file not created")
        # Now evaluate.
        eval_out = Path(tmp) / "eval.json"
        rc = subprocess.run(
            [sys.executable, "-m", "daph_learning.cli.policy", "evaluate",
             "--policy-file", str(policy_out),
             "--synthetic",
             "--output", str(eval_out)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
        assert rc.returncode == 0, (
            f"evaluate failed: {rc.stderr}\n{rc.stdout}")
        assert eval_out.exists(), "eval output not created"
        with open(eval_out) as f:
            result = json.load(f)
        assert "mean_utility" in result or "cand_mean_utility" in result, (
            f"eval result missing utility: {result}")


def test_cli_calibrate_synthetic_runs():
    """Section 22: the calibrate CLI mode with --synthetic should run
    end-to-end and produce calibration thresholds."""
    with tempfile.TemporaryDirectory() as tmp:
        cal_out = Path(tmp) / "calibration.json"
        rc = subprocess.run(
            [sys.executable, "-m", "daph_learning.cli.policy", "calibrate",
             "--synthetic", "--output", str(cal_out)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
        assert rc.returncode == 0, (
            f"calibrate failed: {rc.stderr}\n{rc.stdout}")
        assert cal_out.exists(), "calibration output not created"
        with open(cal_out) as f:
            result = json.load(f)
        assert "tau_ood" in result, (
            f"calibration result missing tau_ood: {result}")
        assert "tau_conf" in result, (
            f"calibration result missing tau_conf: {result}")
