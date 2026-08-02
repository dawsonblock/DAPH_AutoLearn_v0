#!/usr/bin/env python3
"""DAPH v0.4.0a3 — Generate IMPLEMENTATION_SUMMARY.json from actual validation results.

This script derives the summary from real test results, artifact scans,
and pipeline execution — never from hand-maintained optimistic counts.

Usage:
    python scripts/build_implementation_summary.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _run_pytest(marker: str | None = None, *args: str) -> dict:
    """Run pytest and return results."""
    cmd = [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--tb=no",
           "--no-header", "--junitxml", str(ROOT / ".test_results.xml")]
    if marker:
        cmd.extend(["-m", marker])
    cmd.extend(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # Parse JUnit XML for accurate counts
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    junit_path = ROOT / ".test_results.xml"
    if junit_path.exists():
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(junit_path)
            for suite in tree.iter("testsuite"):
                passed += int(suite.get("tests", 0)) - int(suite.get("failures", 0)) - int(suite.get("errors", 0)) - int(suite.get("skipped", 0))
                failed += int(suite.get("failures", 0))
                errors += int(suite.get("errors", 0))
                skipped += int(suite.get("skipped", 0))
        except ET.ParseError:
            pass
        finally:
            junit_path.unlink(missing_ok=True)
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "exit_code": result.returncode,
    }
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "exit_code": result.returncode,
    }


def _check_version_consistency() -> bool:
    """Check that all version surfaces agree on 0.4.0a3."""
    target = "0.4.0a3"
    # pyproject.toml
    pyproject = (ROOT / "pyproject.toml").read_text()
    if f'version = "{target}"' not in pyproject:
        return False
    # __init__.py
    init = (ROOT / "src" / "daph_learning" / "__init__.py").read_text()
    if f'__version__ = "{target}"' not in init:
        return False
    return True


def _check_artifact_scan() -> bool:
    """Check that the artifact scanner passes."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         str(ROOT / "tests" / "test_artifact_scanner.py"),
         "-q", "--tb=no", "--no-header"],
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0


def _check_mock_b5() -> dict:
    """Check that the mock B5 pipeline runs end-to-end."""
    # Clean any existing mock test artifacts
    mock_dir = ROOT / "artifacts" / "executive_b5_mock_test"
    if mock_dir.exists():
        import shutil
        shutil.rmtree(mock_dir)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_b5_staged.py"),
         "all", "--mock", "--config", str(ROOT / "configs" / "executive_b5_mock.yaml")],
        capture_output=True, text=True, timeout=120,
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    qualified = "QUALIFIED" in output and "FAILED" not in output.split("QUALIFIED")[0].split("\n")[-1]

    # Check final state
    status_path = mock_dir / "status.json"
    final_state = ""
    if status_path.exists():
        state = json.loads(status_path.read_text())
        final_state = state.get("status", "")

    return {
        "passed": passed,
        "final_state": final_state,
        "exit_code": result.returncode,
    }


def _check_b4_no_legacy_sham() -> bool:
    """Check that B4 runner has no legacy sham fallback."""
    b4 = (ROOT / "scripts" / "run_b4_staged.py").read_text()
    # The fallback pattern: np.percentile on sham regrets
    if "np.percentile(h_vs_sham" in b4:
        return False
    if "sham_regrets - hidden_regret" in b4:
        return False
    return True


def _check_real_b5_execution_path() -> bool:
    """Check that real B5 execution path exists (not just 'not implemented')."""
    b5 = (ROOT / "scripts" / "run_b5_staged.py").read_text()
    if "ERROR: Real execution not implemented" in b5:
        return False
    if "_real_execute" not in b5:
        return False
    return True


def _check_real_hidden_state_path() -> bool:
    """Check that real hidden-state capture path exists."""
    b5 = (ROOT / "scripts" / "run_b5_staged.py").read_text()
    if "ERROR: Real representation capture not implemented" in b5:
        return False
    if "capture_hidden_states" not in b5:
        return False
    return True


def main():
    print("Building IMPLEMENTATION_SUMMARY.json from actual validation results...")

    # Version consistency
    version_ok = _check_version_consistency()
    print(f"  Version consistency: {'PASS' if version_ok else 'FAIL'}")

    # Artifact scan
    artifact_scan_ok = _check_artifact_scan()
    print(f"  Artifact scan: {'PASS' if artifact_scan_ok else 'FAIL'}")

    # B4 no legacy sham
    b4_no_legacy = _check_b4_no_legacy_sham()
    print(f"  B4 no legacy sham: {'PASS' if b4_no_legacy else 'FAIL'}")

    # Real B5 execution path
    real_b5 = _check_real_b5_execution_path()
    print(f"  Real B5 execution path: {'PASS' if real_b5 else 'FAIL'}")

    # Real hidden state path
    real_hs = _check_real_hidden_state_path()
    print(f"  Real hidden-state path: {'PASS' if real_hs else 'FAIL'}")

    # Mock B5 pipeline
    mock_b5 = _check_mock_b5()
    print(f"  Mock B5 pipeline: {'PASS' if mock_b5['passed'] else 'FAIL'} "
          f"(state={mock_b5['final_state']})")

    # Run mandatory CPU tests (excluding mock_pipeline which is slow)
    print("  Running mandatory CPU tests...")
    test_results = _run_pytest(marker="not mock_pipeline")
    print(f"  Tests: {test_results['passed']} passed, {test_results['failed']} failed, "
          f"{test_results['skipped']} skipped")

    # Determine code_complete
    code_complete = (
        version_ok
        and artifact_scan_ok
        and b4_no_legacy
        and real_b5
        and real_hs
        and mock_b5["passed"]
        and test_results["failed"] == 0
    )

    # These reflect actual evidence — never infer completion
    b4_reproduction_executed = False
    b5_real_executed = False
    scientifically_qualified = False

    summary = {
        "schema_version": "1.0",
        "build_name": "Executive Execution Integrity Release",
        "package_version": "0.4.0a3",
        "date": __import__("time").strftime("%Y-%m-%d"),
        "status": "CODE_COMPLETE" if code_complete else "INCOMPLETE",
        "code_complete": code_complete,
        "mock_b5_pipeline_passed": mock_b5["passed"],
        "b4_reproduction_executed": b4_reproduction_executed,
        "b5_real_executed": b5_real_executed,
        "scientifically_qualified": scientifically_qualified,
        "validation": {
            "version_consistency": version_ok,
            "artifact_scan": artifact_scan_ok,
            "b4_no_legacy_sham": b4_no_legacy,
            "real_b5_execution_path": real_b5,
            "real_hidden_state_path": real_hs,
            "mock_b5_pipeline": mock_b5["passed"],
            "mock_b5_final_state": mock_b5["final_state"],
            "mandatory_tests": {
                "passed": test_results["passed"],
                "failed": test_results["failed"],
                "skipped": test_results["skipped"],
            },
        },
        "readiness": {
            "B4_REPRODUCTION_CODE_READY": b4_no_legacy,
            "B4_REPRODUCTION_EXECUTED": b4_reproduction_executed,
            "B4_REPRODUCTION_QUALIFIED": False,
            "B5_CODE_READY": code_complete,
            "B5_MOCK_VALIDATED": mock_b5["passed"],
            "B5_REAL_EXECUTED": b5_real_executed,
            "B5_QUALIFIED": scientifically_qualified,
        },
        "modules": {
            "new": [
                "atomic_io.py",
                "final_access.py",
            ],
            "modified": [
                "lifecycle.py — TRAIN_RUNNING/TRAIN_COMPLETE states, FinalAccessViolation, freeze accepts FrozenConfig",
                "artifact_integrity.py — B5_REQUIRED_ARTIFACTS updated to spec",
                "reproduce.py — B5 config hash and metric recomputation fixed",
                "b5_policies.py — SurfaceEnsemblePolicy save/load methods",
                "run_b5_staged.py — complete rewrite with phase isolation",
                "run_b4_staged.py — legacy sham fallback removed",
            ],
        },
        "tests": {
            "new": [
                "test_b5_execution_integrity.py — lifecycle, final isolation, B5 runner, B4 regression",
            ],
        },
    }

    output_path = ROOT / "IMPLEMENTATION_SUMMARY.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Written to {output_path}")
    print(f"  Status: {summary['status']}")


if __name__ == "__main__":
    main()
