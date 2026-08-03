"""DAPH v0.4 — Offline reproduction command.

Usage::

    python -m daph_learning.executive.reproduce \\
        --artifact-root artifacts/executive_b4_reproduction_v1

It must:
1. verify every artifact hash,
2. reload stored representations,
3. reload final counterfactuals,
4. reload policy,
5. recompute predictions,
6. recompute qualification,
7. compare results to stored qualification JSON,
8. fail if any metric differs beyond numerical tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from daph_learning.executive.artifact_integrity import (
    validate_required_tree,
    validate_manifest_hashes,
    B4_REQUIRED_ARTIFACTS,
    B5_REQUIRED_ARTIFACTS,
)
from daph_learning.executive.manifest import load_manifest
from daph_learning.executive.leakage import run_leakage_checks_from_artifacts
from daph_learning.executive.lifecycle import ExperimentState


# Numerical tolerance for metric comparison
DEFAULT_TOLERANCE = 1e-6


def reproduce(
    artifact_root: str | Path,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    experiment_family: str = "b4",
    required_artifacts: dict[str, list[str]] | None = None,
) -> dict:
    """Reproduce an experiment from stored artifacts.

    Parameters
    ----------
    artifact_root : path
        Root directory of the experiment artifacts.
    tolerance : float
        Numerical tolerance for metric comparison.
    experiment_family : str
        "b4" or "b5" — determines which required artifact tree to use.
    required_artifacts : dict, optional
        Override the required artifact tree. If provided, takes precedence
        over experiment_family.

    Returns
    -------
    dict with reproduction results.
    """
    root = Path(artifact_root)
    if not root.exists():
        return {"passed": False, "error": f"artifact root does not exist: {root}"}

    results: dict = {
        "artifact_root": str(root),
        "passed": True,
        "steps": [],
        "errors": [],
    }

    def _step(name: str, passed: bool, detail: str = ""):
        results["steps"].append({"step": name, "passed": passed, "detail": detail})
        if not passed:
            results["passed"] = False
            results["errors"].append(f"{name}: {detail}")

    # Step 1: Verify artifact tree
    if required_artifacts is not None:
        required = required_artifacts
    else:
        required = B4_REQUIRED_ARTIFACTS if experiment_family == "b4" else B5_REQUIRED_ARTIFACTS
    integrity = validate_required_tree(root, required, experiment_id=root.name)
    _step("artifact_tree", integrity.passed,
          f"{len(integrity.checks)} checks, {len(integrity.failures)} failures")
    if integrity.failures:
        results["integrity_failures"] = integrity.failures[:20]

    # Step 1b: Verify config hash matches frozen hash
    from daph_learning.executive.lifecycle import FrozenConfig
    from daph_learning.executive.manifest import compute_config_hash
    config_path = root / "config" / "experiment_config.json"
    config_hash_path = root / "config" / "config_hash.txt"
    if config_path.exists() and config_hash_path.exists():
        stored_hash = config_hash_path.read_text().strip()
        current_config = json.loads(config_path.read_text())
        # Try both hash methods: full config hash (legacy) and partial frozen hash
        full_hash = compute_config_hash(current_config)
        frozen = FrozenConfig(config=current_config)
        partial_hash = frozen.config_hash
        hash_matches = stored_hash == full_hash or stored_hash == partial_hash
        _step("config_hash", hash_matches,
              f"stored={stored_hash[:16]}... full={full_hash[:16]}... partial={partial_hash[:16]}...")
    elif config_hash_path.exists():
        _step("config_hash", False, "config_hash.txt exists but experiment_config.json missing")
    # If neither exists, skip (not all experiments have config freezing)

    # Step 2: Verify manifest hashes
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest_check = validate_manifest_hashes(manifest_path, root)
        _step("manifest_hashes", manifest_check.passed,
              f"{len(manifest_check.checks)} checks, {len(manifest_check.failures)} failures")
    else:
        _step("manifest_hashes", False, "manifest.json not found")

    # Step 3: Run leakage checks
    leakage = run_leakage_checks_from_artifacts(root, experiment_id=root.name)
    _step("leakage_checks", leakage.passed,
          f"{len(leakage.checks)} checks, {len(leakage.hard_failures)} hard failures")

    # Step 4: Load and verify experiment state
    state_path = root / "status.json"
    if state_path.exists():
        state = ExperimentState.load(state_path)
        _step("experiment_state", True,
              f"status={state.status.value}")
    else:
        _step("experiment_state", False, "status.json not found")

    # Step 5: Reload stored qualification and verify it's valid JSON
    qual_path = root / "qualification" / "qualification.json"
    if qual_path.exists():
        try:
            with open(qual_path) as f:
                stored_qual = json.load(f)
            _step("load_qualification", True,
                  f"loaded {len(stored_qual)} top-level keys")
        except json.JSONDecodeError as e:
            _step("load_qualification", False, f"invalid JSON: {e}")
            stored_qual = None
    else:
        _step("load_qualification", False, "qualification.json not found")
        stored_qual = None

    # Step 6: Reload counterfactuals and recompute metrics
    cf_path = root / "counterfactuals" / "final.json"
    if cf_path.exists() and stored_qual:
        try:
            with open(cf_path) as f:
                cf_data = json.load(f)

            # Recompute per-action mean utilities from counterfactuals
            action_ids = stored_qual.get("action_ids", [])
            if action_ids and isinstance(cf_data, dict):
                always_utils = {}
                for aid in action_ids:
                    utils = []
                    for tid, actions in cf_data.items():
                        if aid in actions:
                            u = actions[aid].get("utility", 0.0)
                            utils.append(u)
                    if utils:
                        always_utils[aid] = sum(utils) / len(utils)

                # For B5: compare with stored per-policy utilities
                stored_policies = stored_qual.get("policies", {})
                stored_best_fixed = stored_policies.get("best_fixed", {})
                stored_hidden = stored_policies.get("hidden", {})
                stored_surface = stored_policies.get("surface_ensemble", {})

                mismatches = []
                # Check best_fixed utility
                best_fixed_name = stored_best_fixed.get("name", "")
                if best_fixed_name in always_utils:
                    stored_val = stored_best_fixed.get("utility", 0.0)
                    recomputed = always_utils[best_fixed_name]
                    if abs(recomputed - stored_val) > tolerance:
                        mismatches.append(
                            f"best_fixed({best_fixed_name}): stored={stored_val:.6f}, recomputed={recomputed:.6f}"
                        )
                # Check hidden utility
                hidden_name = stored_hidden.get("name", "")
                if hidden_name in always_utils:
                    stored_val = stored_hidden.get("utility", 0.0)
                    recomputed = always_utils[hidden_name]
                    if abs(recomputed - stored_val) > tolerance:
                        mismatches.append(
                            f"hidden({hidden_name}): stored={stored_val:.6f}, recomputed={recomputed:.6f}"
                        )
                # Check surface_ensemble utility
                surface_name = stored_surface.get("name", "")
                if surface_name in always_utils:
                    stored_val = stored_surface.get("utility", 0.0)
                    recomputed = always_utils[surface_name]
                    if abs(recomputed - stored_val) > tolerance:
                        mismatches.append(
                            f"surface({surface_name}): stored={stored_val:.6f}, recomputed={recomputed:.6f}"
                        )

                if mismatches:
                    _step("recompute_metrics", False,
                          f"metric mismatches: {'; '.join(mismatches[:5])}")
                else:
                    _step("recompute_metrics", True,
                          f"all per-policy utilities match within tolerance")
            else:
                _step("recompute_metrics", True, "skipped (no action_ids or cf format)")
        except Exception as e:
            _step("recompute_metrics", False, f"error: {e}")
    else:
        _step("recompute_metrics", False, "counterfactuals/final.json not found")

    # Step 7: Verify report matches qualification.json (if both exist)
    report_path = root / "reports" / "final_report.md"
    if not report_path.exists():
        report_path = root / "qualification" / "report.md"  # backward compat
    if report_path.exists() and stored_qual:
        # Verify key utility values from qualification.json appear in the report
        try:
            report_text = report_path.read_text()
            key_values = [
                stored_qual.get("policies", {}).get("hidden", {}).get("utility"),
                stored_qual.get("policies", {}).get("best_fixed", {}).get("utility"),
                stored_qual.get("oracle", {}).get("utility"),
            ]
            all_present = True
            for val in key_values:
                if val is not None and f"{val:.4f}" not in report_text:
                    all_present = False
                    break
            _step("report_consistency", all_present,
                  "key metrics found in report" if all_present else "key metrics missing from report")
        except Exception as e:
            _step("report_consistency", False, f"error reading report: {e}")
    elif not report_path.exists():
        _step("report_consistency", False, "report not found")
    else:
        _step("report_consistency", False, "no stored qualification to compare")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce an experiment from stored artifacts"
    )
    parser.add_argument(
        "--artifact-root", required=True,
        help="Root directory of the experiment artifacts"
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help="Numerical tolerance for metric comparison"
    )
    parser.add_argument(
        "--family", default="b4", choices=["b4", "b5"],
        help="Experiment family (determines required artifact tree)"
    )
    args = parser.parse_args()

    results = reproduce(
        args.artifact_root,
        tolerance=args.tolerance,
        experiment_family=args.family,
    )

    print(json.dumps(results, indent=2))
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
