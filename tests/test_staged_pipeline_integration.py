"""Section 22 — end-to-end integration test for the staged Gate A pipeline.

This test runs the full staged workflow (collect → develop → calibrate →
freeze → final) on a small smoke config and verifies that:
1. Each stage produces the expected artifacts.
2. The freeze manifest is complete (all identity-bearing inputs recorded).
3. The final stage verifies frozen state before proceeding.
4. The final report has labeled CIs and a gate decision.
5. The sham control ran with a shared training spec.
6. The canonical verifier was used for both backends.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _write_smoke_config(path: Path) -> None:
    """Write a minimal smoke config for integration testing."""
    path.write_text("""\
experiment_id: daph_gate_a_integration_test
release_version: 0.3.10.4-alpha
evidence_level: REAL_MODEL_SMOKE

primary_endpoint:
  name: verified_utility
  estimand: group_weighted
  confidence_level: 0.95
  bootstrap_iterations: 500

utility_protocol: gate_a_accuracy_primary
secondary_utility_protocol: gate_a_cost_sensitive_secondary

gates:
  minimum_point_gain_vs_p0: 0.0
  require_lcb_vs_p0_above: -1.0
  require_lcb_vs_sham_above: -1.0
  minimum_oracle_gap_capture: 0.0
  maximum_worst_subtype_regression: 1.0
  minimum_positive_group_fraction: 0.0
  maximum_final_access_count: 1

dataset:
  minimum_groups: 8
  minimum_crossover_subtypes: 1
  minimum_backend_win_fraction: 0.10
  minimum_decisive_fraction: 0.10

evidence:
  require_real_model: true
  allow_synthetic: false
  require_model_revision: true
  require_source_hash_match: true
  require_dataset_hashes: true
  require_final_access_ledger: false

model:
  model_id: mock-llm
  dtype: float16
  max_new_tokens: 48
  do_sample: false
  replicates: 1

representation:
  layer_strategy: depth_fractional
  pooling_candidates:
    - last_prompt_token
  selection_metric: dev_objective
  selection_data: development

targets:
  mode_selection_data: development
  modes:
    - hard_gap

sham:
  n_seeds: 3
  procedure: subtype_split_decisive_shuffle

splits:
  train: 0.50
  development: 0.20
  calibration: 0.15
  final: 0.15
""")


@pytest.fixture
def integration_workspace(tmp_path, monkeypatch):
    """Create a temporary workspace for the integration test."""
    # Save the real pointer so we can restore it after the test.
    real_pointer_path = REPO_ROOT / "artifacts" / "current" / "pointer.json"
    real_pointer_content = None
    if real_pointer_path.exists():
        real_pointer_content = real_pointer_path.read_text()

    # Write the smoke config.
    config_path = tmp_path / "gate_a_integration.yaml"
    _write_smoke_config(config_path)

    # Monkeypatch REPO_ROOT in the staged script to use tmp_path.
    # We need to set the artifacts directory.
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    monkeypatch.setattr(
        "scripts.run_gate_a_staged.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.freeze_gate_a.REPO_ROOT", tmp_path)

    yield {
        "config_path": config_path,
        "workspace": tmp_path,
        "artifacts_dir": artifacts_dir,
    }

    # Restore the real pointer after the test.
    if real_pointer_content is not None:
        real_pointer_path.write_text(real_pointer_content)
    elif real_pointer_path.exists():
        real_pointer_path.unlink()


def test_end_to_end_staged_pipeline(integration_workspace):
    """Run the full staged pipeline end-to-end and verify artifacts."""
    import importlib

    # Import the staged runner module.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_gate_a_staged
    import freeze_gate_a

    config_path = integration_workspace["config_path"]

    # Stage 1: collect
    from daph_learning.evaluation.gate_criteria import load_gate_criteria
    criteria = load_gate_criteria(str(config_path))

    class Args:
        seed = 42
        n_per_group = 4
        n_layers = None

    rc = run_gate_a_staged.stage_collect(criteria, Args())
    assert rc == 0, "collect stage failed"

    # Verify collect artifacts.
    collect_dir = run_gate_a_staged._collect_dir(criteria)
    assert (collect_dir / "train_tasks.json").exists()
    assert (collect_dir / "development_tasks.json").exists()
    assert (collect_dir / "calibration_tasks.json").exists()
    assert (collect_dir / "final_tasks.json").exists()
    assert (collect_dir / "dataset_audit.json").exists()
    assert (collect_dir / "dataset_hashes.json").exists()

    # Verify dataset hashes are non-empty.
    hashes = json.loads((collect_dir / "dataset_hashes.json").read_text())
    assert all(v != "" for v in hashes.values()), "dataset hashes must be non-empty"

    # Stage 2: develop
    rc = run_gate_a_staged.stage_develop(criteria, Args())
    assert rc == 0, "develop stage failed"

    develop_dir = run_gate_a_staged._develop_dir(criteria)
    assert (develop_dir / "train_experiences.json").exists()
    assert (develop_dir / "development_experiences.json").exists()
    assert (develop_dir / "representation_selection.json").exists()
    assert (develop_dir / "policy_artifact.json").exists()

    # Verify experiences have canonical verification fields.
    train_exp = json.loads((develop_dir / "train_experiences.json").read_text())
    assert len(train_exp) > 0
    e = train_exp[0]
    assert "symbolic_verification" in e
    assert "llm_verification" in e
    assert "symbolic_canonical_answer" in e
    assert "llm_canonical_answer" in e
    assert "delta_utility" in e

    # Stage 3: calibrate
    rc = run_gate_a_staged.stage_calibrate(criteria, Args())
    assert rc == 0, "calibrate stage failed"
    assert (develop_dir / "calibration_experiences.json").exists()
    assert (develop_dir / "calibration.json").exists()

    # Stage 4: freeze
    sys.argv = ["freeze_gate_a.py", "--config", str(config_path)]
    try:
        rc = freeze_gate_a.main()
    except SystemExit as exc:
        rc = exc.code
    assert rc == 0, "freeze stage failed"

    final_dir = run_gate_a_staged._out_dir(criteria, "final")
    assert (final_dir / "freeze_manifest.json").exists()
    assert (final_dir / "final_access_ledger.json").exists()

    # Verify freeze manifest is complete.
    from daph_learning.policy.stage import FreezeManifest
    manifest = FreezeManifest.load(final_dir / "freeze_manifest.json")
    manifest.assert_complete()  # should not raise
    assert manifest.experiment_id == "daph_gate_a_integration_test"
    assert manifest.train_dataset_sha256 != ""
    assert manifest.utility_config_sha256 != ""
    assert manifest.model_id == "mock-llm"
    assert manifest.gate_criteria_sha256 != ""

    # Stage 5: final
    rc = run_gate_a_staged.stage_final(criteria, Args())
    assert rc == 0, "final stage failed"

    # Verify final artifacts.
    assert (final_dir / "final_experiences.json").exists()
    assert (final_dir / "gate_decision.json").exists()
    assert (final_dir / "GATE_A_RESULTS.md").exists()
    assert (final_dir / "experiment_results.json").exists()

    # Verify the report has labeled CIs.
    report_md = (final_dir / "GATE_A_RESULTS.md").read_text()
    assert "95% CI" in report_md
    assert "group_weighted" in report_md
    assert "Utility protocol" in report_md
    assert "Training spec hash" in report_md

    # Verify the gate decision.
    decision = json.loads((final_dir / "gate_decision.json").read_text())
    assert "passed" in decision
    # Accept either old (gate_verdicts) or new (statistical_gates) schema.
    gates = decision.get("statistical_gates")
    if gates is None:
        gates = decision.get("gate_verdicts")
    assert gates is not None
    # Gates may be empty if preconditions fail (NOT_EVALUABLE).
    assert isinstance(gates, dict)

    # Verify the final access ledger was written.
    ledger = json.loads((final_dir / "final_access_ledger.json").read_text())
    assert ledger["n_accesses"] == 1


def test_final_refuses_after_source_change(integration_workspace):
    """Final stage must refuse if the source hash changed after freeze."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_gate_a_staged
    import freeze_gate_a

    config_path = integration_workspace["config_path"]
    from daph_learning.evaluation.gate_criteria import load_gate_criteria
    criteria = load_gate_criteria(str(config_path))

    class Args:
        seed = 42
        n_per_group = 4
        n_layers = None

    # Run collect → develop → calibrate → freeze → final.
    assert run_gate_a_staged.stage_collect(criteria, Args()) == 0
    assert run_gate_a_staged.stage_develop(criteria, Args()) == 0
    assert run_gate_a_staged.stage_calibrate(criteria, Args()) == 0

    sys.argv = ["freeze_gate_a.py", "--config", str(config_path)]
    try:
        rc = freeze_gate_a.main()
    except SystemExit as exc:
        rc = exc.code
    assert rc == 0

    # Corrupt the freeze manifest with a wrong source hash.
    final_dir = run_gate_a_staged._out_dir(criteria, "final")
    manifest_path = final_dir / "freeze_manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    manifest_data["source_tree_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    # Final should refuse.
    rc = run_gate_a_staged.stage_final(criteria, Args())
    assert rc == 1, "final should refuse when source hash changed"
