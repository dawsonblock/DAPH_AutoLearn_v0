#!/usr/bin/env python
"""Section 22 — freeze the Gate A protocol and policy.

Freezes ALL identity-bearing inputs: source hash, config hash, dataset
hashes, utility config hash, model id, representation config hash,
policy hash, calibration hash, and gate criteria hash. Writes
``freeze_manifest.json`` + ``final_access_ledger.json`` (reserved
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


def _hash_tasks(tasks: list[dict]) -> str:
    payload = json.dumps(tasks, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    from daph_learning.policy.utility import get_protocol

    criteria = load_gate_criteria(args.config)
    source_hash = compute_canonical_source_hash(REPO_ROOT)
    config_hash = criteria.criteria_hash
    utility_config = get_protocol(criteria.utility_protocol)

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "artifacts" / "gate_a_qualified" / criteria.experiment_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset hashes from collect stage.
    if "smoke" in criteria.experiment_id:
        collect_dir = REPO_ROOT / "artifacts" / "real_model_smoke" / criteria.experiment_id
    else:
        collect_dir = REPO_ROOT / "artifacts" / "synthetic_ci" / criteria.experiment_id
    dataset_hashes_path = collect_dir / "dataset_hashes.json"
    if dataset_hashes_path.exists():
        dataset_hashes = json.loads(dataset_hashes_path.read_text())
    else:
        # Try loading tasks directly.
        dataset_hashes = {}
        for name in ("train", "development", "calibration", "final"):
            task_path = collect_dir / f"{name}_tasks.json"
            if task_path.exists():
                tasks = json.loads(task_path.read_text())
                dataset_hashes[name] = _hash_tasks(tasks)
            else:
                dataset_hashes[name] = ""

    # Load policy artifact from develop stage.
    if "smoke" in criteria.experiment_id:
        develop_dir = REPO_ROOT / "artifacts" / "real_model_smoke" / criteria.experiment_id
    else:
        develop_dir = REPO_ROOT / "artifacts" / "synthetic_ci" / criteria.experiment_id
    policy_artifact_path = develop_dir / "policy_artifact.json"
    policy_hash = ""
    if policy_artifact_path.exists():
        policy_artifact = json.loads(policy_artifact_path.read_text())
        policy_hash = hashlib.sha256(
            json.dumps(policy_artifact, sort_keys=True).encode()
        ).hexdigest()

    # Load calibration artifact.
    if "smoke" in criteria.experiment_id:
        cal_dir = REPO_ROOT / "artifacts" / "real_model_smoke" / criteria.experiment_id
    else:
        cal_dir = REPO_ROOT / "artifacts" / "synthetic_ci" / criteria.experiment_id
    cal_path = cal_dir / "calibration.json"
    calibration_hash = ""
    if cal_path.exists():
        cal_data = json.loads(cal_path.read_text())
        calibration_hash = hashlib.sha256(
            json.dumps(cal_data, sort_keys=True).encode()
        ).hexdigest()

    # Model info from criteria.
    model_info = criteria.raw.get("model", {})
    model_id = model_info.get("model_id", "")
    model_revision = model_info.get("revision", "")
    tokenizer_id = model_id  # same as model for HF
    tokenizer_revision = model_revision

    # Representation config hash.
    rep_info = criteria.raw.get("representation", {})
    representation_hash = hashlib.sha256(
        json.dumps(rep_info, sort_keys=True).encode()
    ).hexdigest()

    manifest = FreezeManifest(
        experiment_id=criteria.experiment_id,
        release_version=criteria.release_version,
        source_tree_sha256=source_hash,
        config_sha256=config_hash,
        train_dataset_sha256=dataset_hashes.get("train", ""),
        development_dataset_sha256=dataset_hashes.get("development", ""),
        calibration_dataset_sha256=dataset_hashes.get("calibration", ""),
        final_dataset_sha256=dataset_hashes.get("final", ""),
        utility_config_sha256=utility_config.utility_config_hash,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        representation_config_sha256=representation_hash,
        policy_sha256=policy_hash,
        calibration_sha256=calibration_hash,
        gate_criteria_sha256=criteria.criteria_hash,
        frozen_at=time.time(),
        stage="frozen",
    )
    manifest.assert_complete()
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
    print(f"[freeze] utility_config_hash: {utility_config.utility_config_hash[:16]}...")
    print(f"[freeze] model_id: {model_id}")
    print(f"[freeze] policy_hash: {policy_hash[:16] if policy_hash else '(empty)'}...")
    print(f"[freeze] calibration_hash: {calibration_hash[:16] if calibration_hash else '(empty)'}...")
    print(f"[freeze] manifest: {out_dir / 'freeze_manifest.json'}")
    print(f"[freeze] ledger: {out_dir / 'final_access_ledger.json'}")
    print(f"[freeze] Protocol frozen. No changes allowed after this point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
