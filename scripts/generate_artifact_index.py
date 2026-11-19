#!/usr/bin/env python
"""Section 8.4 — Generate ARTIFACT_INDEX.json for an experiment bundle.

Enumerates all files, sizes, SHA-256 hashes, and semantic roles.

Usage::

    python scripts/generate_artifact_index.py --bundle artifacts/gate_a_qualified/daph_gate_a_real_005_requal
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


# Semantic roles for known artifact files.
FILE_ROLES = {
    "experiment_results.json": "experiment_results",
    "gate_decision.json": "gate_decision",
    "final_predictions.json": "final_policy_predictions",
    "final_task_metrics.json": "final_task_metrics",
    "final_experiences.json": "final_counterfactual_experiences",
    "final_features.npy": "final_policy_features",
    "final_access_ledger.json": "final_access_ledger",
    "freeze_manifest.json": "freeze_manifest",
    "sham_predictions.json": "sham_predictions",
    "bootstrap_p1_minus_p0.npy": "bootstrap_samples_p1_minus_p0",
    "bootstrap_p1_minus_sham.npy": "bootstrap_samples_p1_minus_sham",
    "GATE_A_RESULTS.md": "gate_a_report",
    "trained_baselines.json": "trained_baseline_results",
    "feature_manifest.json": "feature_manifest",
    "environment_manifest.json": "environment_manifest",
    "representation_selection.json": "representation_selection",
    "policy_artifact.json": "frozen_policy_artifact",
    "calibration.json": "calibration_artifact",
    "best_fixed_selection.json": "best_fixed_comparator_selection",
    "surface_only_policy.json": "surface_only_policy",
    "hidden_only_policy.json": "hidden_only_policy",
    "tfidf_policy.json": "tfidf_policy",
    "heuristic_policy.json": "heuristic_policy",
    "ARTIFACT_INDEX.json": "artifact_index",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ARTIFACT_INDEX.json")
    parser.add_argument("--bundle", required=True, help="Path to experiment bundle directory")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    if not bundle.exists():
        print(f"ERROR: bundle not found: {bundle}", file=sys.stderr)
        return 1

    files = []
    for path in sorted(bundle.iterdir()):
        if path.is_file():
            files.append({
                "path": path.name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "role": FILE_ROLES.get(path.name, "unknown"),
            })

    index = {
        "artifact_type": "artifact_index",
        "bundle": str(bundle.relative_to(Path.cwd()) if bundle.is_relative_to(Path.cwd()) else bundle),
        "n_files": len(files),
        "files": files,
    }

    out_path = bundle / "ARTIFACT_INDEX.json"
    out_path.write_text(json.dumps(index, indent=2))
    print(f"Wrote {out_path} ({len(files)} files indexed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
