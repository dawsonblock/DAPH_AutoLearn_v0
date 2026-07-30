#!/usr/bin/env python
"""Section 22 — freeze the Gate A protocol and policy.

Freezes source hash, config hash, policy hash, calibration hash, and
writes ``freeze_manifest.json`` + ``final_access_ledger.json`` (reserved
state). After this, no changes to source/dataset/utility/model/
representation/policy/calibration/gate-criteria are allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Gate A protocol")
    parser.add_argument("--config", required=True,
                        help="Path to gate criteria YAML")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for freeze manifest")
    args = parser.parse_args()

    from daph_learning.provenance import compute_canonical_source_hash
    from daph_learning.evaluation.gate_criteria import load_gate_criteria
    from daph_learning.policy.stage import (
        FreezeManifest, FinalAccessLedger, StageGuard, ExperimentStage)

    criteria = load_gate_criteria(args.config)
    source_hash = compute_canonical_source_hash(REPO_ROOT)
    config_hash = criteria.criteria_hash

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "artifacts" / "gate_a_qualified" / criteria.experiment_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = FreezeManifest(
        release_version=criteria.release_version,
        source_tree_sha256=source_hash,
        config_sha256=config_hash,
        policy_sha256="",  # filled by the experiment script
        calibration_sha256="",  # filled by the experiment script
        frozen_at=time.time(),
        stage="frozen",
    )
    manifest.save(out_dir / "freeze_manifest.json")

    ledger = FinalAccessLedger(
        max_accesses=criteria.gates.maximum_final_access_count,
        release_version=criteria.release_version,
        source_hash=source_hash,
        config_hash=config_hash,
    )
    ledger.save(out_dir / "final_access_ledger.json")

    print(f"[freeze] experiment_id: {criteria.experiment_id}")
    print(f"[freeze] source_hash: {source_hash[:16]}...")
    print(f"[freeze] config_hash: {config_hash[:16]}...")
    print(f"[freeze] manifest: {out_dir / 'freeze_manifest.json'}")
    print(f"[freeze] ledger: {out_dir / 'final_access_ledger.json'}")
    print(f"[freeze] Protocol frozen. No changes allowed after this point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
