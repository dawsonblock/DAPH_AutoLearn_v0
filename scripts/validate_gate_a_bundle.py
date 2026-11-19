#!/usr/bin/env python
"""Section 23 — validate a Gate A artifact bundle.

Wraps :func:`validate_artifact_bundle` with Gate A-specific requirements:
``require_real_model=True``, ``require_final_access_ledger=True``.
Exits nonzero on any failure (Section 23 CLI failure behavior).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Gate A artifact bundle")
    parser.add_argument("bundle_dir",
                        help="Path to the artifact bundle directory")
    parser.add_argument("--repo-root", default=str(REPO_ROOT),
                        help="Repository root for source hash verification")
    parser.add_argument("--require-final-access-ledger", action="store_true",
                        default=True,
                        help="Require final_access_ledger.json (default: true)")
    args = parser.parse_args()

    from daph_learning.evaluation.artifact_integrity import (
        validate_artifact_bundle)
    from daph_learning.provenance import compute_canonical_source_hash

    bundle = Path(args.bundle_dir)
    if not bundle.exists():
        print(f"[validate] ERROR: bundle directory does not exist: {bundle}",
              file=sys.stderr)
        return 1

    result = validate_artifact_bundle(bundle, Path(args.repo_root))
    print(f"[validate] bundle: {bundle}")
    print(f"[validate] valid: {result.valid}")
    print(f"[validate] errors: {len(result.errors)}")
    print(f"[validate] warnings: {len(result.warnings)}")

    # Gate A-specific checks.
    if args.require_final_access_ledger:
        ledger = bundle / "final_access_ledger.json"
        if not ledger.exists():
            result.errors.append(
                "final_access_ledger.json missing (required for Gate A)")
            result.valid = False

    # Check for real model evidence.
    result_json = bundle / "experiment_results.json"
    if result_json.exists():
        data = json.loads(result_json.read_text())
        if not data.get("used_real_model", False):
            result.errors.append(
                "experiment_results.json does not confirm real model usage")
            result.valid = False

    if result.errors:
        for e in result.errors:
            print(f"[validate]   ERROR: {e}", file=sys.stderr)
    if result.warnings:
        for w in result.warnings:
            print(f"[validate]   WARNING: {w}")

    if not result.valid:
        print("[validate] FAILED — bundle is not valid", file=sys.stderr)
        return 1
    print("[validate] OK — bundle is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
