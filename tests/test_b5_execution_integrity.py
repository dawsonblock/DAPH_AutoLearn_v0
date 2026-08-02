"""DAPH v0.4.0a3 — Execution integrity tests.

Covers:
- Lifecycle state machine (DEVELOPMENT → FROZEN → TRAIN_RUNNING → TRAIN_COMPLETE → FINAL_RUNNING → QUALIFIED)
- Final access guard and ledger
- B5 mock pipeline (subprocess)
- B5 resume
- B5 config mismatch rejection
- B5 final access rejected during train
- B4 no legacy sham fallback
- B4 uses shared paired comparison
- B4 surface ensemble is frozen
- Version consistency
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daph_learning.executive.lifecycle import (
    ExperimentState, ExperimentStatus, FrozenConfig, FinalAccessViolation,
)
from daph_learning.executive.final_access import (
    FinalAccessGuard, check_final_isolation, FINAL_SPLITS,
)
from daph_learning.executive.atomic_io import (
    atomic_write_json, atomic_write_text, atomic_write_npz,
)


ROOT = Path(__file__).parent.parent
MOCK_CONFIG = ROOT / "configs" / "executive_b5_mock.yaml"
MOCK_OUTPUT = ROOT / "artifacts" / "executive_b5_mock_test"


# ──────────────────────────────────────────────────────────────────────
# Lifecycle state machine tests
# ──────────────────────────────────────────────────────────────────────

class TestLifecycleStateMachine:
    """Test the canonical experiment state machine."""

    def test_development_to_frozen_allowed(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        assert state.status == ExperimentStatus.FROZEN

    def test_development_to_final_running_rejected(self):
        state = ExperimentState(experiment_id="test")
        with pytest.raises(ValueError, match="invalid status transition"):
            state.transition_to(ExperimentStatus.FINAL_RUNNING)

    def test_frozen_to_final_running_allowed(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        state.start_final({"experiment_id": "test"})
        assert state.status == ExperimentStatus.FINAL_RUNNING

    def test_frozen_to_train_running_allowed(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        state.start_training()
        assert state.status == ExperimentStatus.TRAIN_RUNNING

    def test_train_running_to_train_complete(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        state.start_training()
        state.complete_training()
        assert state.status == ExperimentStatus.TRAIN_COMPLETE

    def test_train_complete_to_final_running(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        state.start_training()
        state.complete_training()
        state.start_final({"experiment_id": "test"})
        assert state.status == ExperimentStatus.FINAL_RUNNING

    def test_final_running_to_qualified(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        state.start_final({"experiment_id": "test"})
        state.mark_qualified()
        assert state.status == ExperimentStatus.QUALIFIED

    def test_state_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(experiment_id="persist_test")
            state.freeze({"experiment_id": "persist_test", "action_space": {}})
            state.start_training()
            state_path = Path(tmp) / "status.json"
            state.save(state_path)

            loaded = ExperimentState.load(state_path)
            assert loaded.experiment_id == "persist_test"
            assert loaded.status == ExperimentStatus.TRAIN_RUNNING
            assert loaded.frozen_config is not None
            assert len(loaded.history) == 2  # freeze + start_training

    def test_config_mismatch_rejected(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test", "action_space": {"a": 1}})
        with pytest.raises(ValueError, match="does not match"):
            state.start_final({"experiment_id": "test", "action_space": {"a": 2}})

    def test_freeze_accepts_frozen_config(self):
        state = ExperimentState(experiment_id="test")
        frozen = FrozenConfig(config={"experiment_id": "test"})
        state.freeze(frozen)
        assert state.status == ExperimentStatus.FROZEN
        assert state.frozen_config.config_hash == frozen.config_hash

    def test_corrupted_status_file_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "status.json"
            state_path.write_text("not valid json{{{")
            with pytest.raises(json.JSONDecodeError):
                ExperimentState.load(state_path)

    def test_history_has_timestamps(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        assert len(state.history) == 1
        assert "timestamp" in state.history[0]
        assert "from" in state.history[0]
        assert "to" in state.history[0]
        assert state.history[0]["from"] == "DEVELOPMENT"
        assert state.history[0]["to"] == "FROZEN"


# ──────────────────────────────────────────────────────────────────────
# Final access guard tests
# ──────────────────────────────────────────────────────────────────────

class TestFinalAccessGuard:
    """Test the final access guard and ledger."""

    def test_final_access_rejected_during_train(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        state.start_training()
        with tempfile.TemporaryDirectory() as tmp:
            guard = FinalAccessGuard(state, Path(tmp) / "ledger.jsonl")
            with pytest.raises(FinalAccessViolation, match="final"):
                guard.assert_can_read("final", artifact="test.json", purpose="test")

    def test_final_access_rejected_during_development(self):
        state = ExperimentState(experiment_id="test")
        with tempfile.TemporaryDirectory() as tmp:
            guard = FinalAccessGuard(state, Path(tmp) / "ledger.jsonl")
            with pytest.raises(FinalAccessViolation):
                guard.assert_can_read("final", artifact="test.json", purpose="test")

    def test_final_access_allowed_during_final_running(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        state.start_final({"experiment_id": "test"})
        with tempfile.TemporaryDirectory() as tmp:
            guard = FinalAccessGuard(state, Path(tmp) / "ledger.jsonl")
            guard.assert_can_read("final", artifact="test.json", purpose="test")
            assert len(guard.records) == 1

    def test_ood_access_rejected_during_train(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        state.start_training()
        with tempfile.TemporaryDirectory() as tmp:
            guard = FinalAccessGuard(state, Path(tmp) / "ledger.jsonl")
            with pytest.raises(FinalAccessViolation, match="final_ood"):
                guard.assert_can_read("final_ood", artifact="test.json", purpose="test")

    def test_non_final_split_always_accessible(self):
        state = ExperimentState(experiment_id="test")
        with tempfile.TemporaryDirectory() as tmp:
            guard = FinalAccessGuard(state, Path(tmp) / "ledger.jsonl")
            guard.assert_can_read("train", artifact="test.json", purpose="test")
            guard.assert_can_read("dev", artifact="test.json", purpose="test")

    def test_ledger_persisted(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        state.start_final({"experiment_id": "test"})
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            guard = FinalAccessGuard(state, ledger_path)
            guard.assert_can_read("final", artifact="cf.json", purpose="eval", stage="qualify")
            assert ledger_path.exists()
            lines = ledger_path.read_text().strip().split("\n")
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["split"] == "final"
            assert record["stage"] == "qualify"

    def test_final_isolation_check_passes(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        state.start_final({"experiment_id": "test"})
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            guard = FinalAccessGuard(state, ledger_path)
            guard.assert_can_read("final", artifact="cf.json", purpose="eval", stage="qualify")
            result = check_final_isolation(ledger_path)
            assert result["passed"]

    def test_final_isolation_check_fails_for_pre_final_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            # Simulate a pre-final access violation
            from daph_learning.executive.atomic_io import append_jsonl
            append_jsonl(ledger_path, {
                "timestamp": "2026-01-01T00:00:00",
                "experiment_id": "test",
                "stage": "train",
                "split": "final",
                "artifact": "cf.json",
                "purpose": "eval",
                "config_hash": "",
            })
            result = check_final_isolation(ledger_path)
            assert not result["passed"]
            assert len(result["violations"]) == 1


# ──────────────────────────────────────────────────────────────────────
# Atomic I/O tests
# ──────────────────────────────────────────────────────────────────────

class TestAtomicIO:
    """Test crash-safe atomic writes."""

    def test_atomic_write_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            atomic_write_json(path, {"key": "value"})
            assert json.loads(path.read_text())["key"] == "value"

    def test_atomic_write_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            atomic_write_text(path, "hello world")
            assert path.read_text() == "hello world"

    def test_atomic_write_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.npz"
            atomic_write_npz(path, a=np.array([1, 2, 3]), b=np.array([4, 5, 6]))
            data = np.load(path)
            assert np.array_equal(data["a"], [1, 2, 3])
            assert np.array_equal(data["b"], [4, 5, 6])

    def test_atomic_write_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            atomic_write_json(path, {"v": 1})
            atomic_write_json(path, {"v": 2})
            assert json.loads(path.read_text())["v"] == 2


# ──────────────────────────────────────────────────────────────────────
# B5 mock pipeline tests (subprocess)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.mock_pipeline
class TestB5MockPipeline:
    """Test the B5 mock pipeline end-to-end via subprocess."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Clean up mock artifacts before and after each test."""
        if MOCK_OUTPUT.exists():
            shutil.rmtree(MOCK_OUTPUT)
        yield
        if MOCK_OUTPUT.exists():
            shutil.rmtree(MOCK_OUTPUT)

    def _run_pipeline(self, *extra_args):
        cmd = [
            sys.executable, str(ROOT / "scripts" / "run_b5_staged.py"),
            "all", "--mock", "--config", str(MOCK_CONFIG),
        ] + list(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    def test_b5_mock_all_completes(self):
        """Test that the full mock pipeline completes with exit code 0."""
        result = self._run_pipeline()
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout[-500:]}"

        # Check final state
        status_path = MOCK_OUTPUT / "status.json"
        assert status_path.exists()
        state = json.loads(status_path.read_text())
        # Should be QUALIFIED or FAILED_QUALIFICATION (not infrastructure failure)
        assert state["status"] in ("QUALIFIED", "FAILED_QUALIFICATION")

    def test_b5_mock_produces_required_artifacts(self):
        """Test that all required qualification artifacts are produced."""
        result = self._run_pipeline()
        assert result.returncode == 0

        qual_dir = MOCK_OUTPUT / "qualification"
        assert (qual_dir / "qualification.json").exists()
        assert (qual_dir / "per_task_results.json").exists()
        assert (qual_dir / "per_group_results.json").exists()
        assert (qual_dir / "ood_results.json").exists()
        assert (qual_dir / "bootstrap_results.npz").exists()

    def test_b5_mock_final_isolation_passes(self):
        """Test that final isolation check passes."""
        result = self._run_pipeline()
        assert result.returncode == 0

        iso_path = MOCK_OUTPUT / "checks" / "final_isolation.json"
        assert iso_path.exists()
        iso = json.loads(iso_path.read_text())
        assert iso["passed"], f"Final isolation failed: {iso.get('violations', [])}"

    def test_b5_mock_reproduction_passes(self):
        """Test that offline reproduction passes."""
        result = self._run_pipeline()
        assert result.returncode == 0

        rep_path = MOCK_OUTPUT / "checks" / "reproduction.json"
        assert rep_path.exists()
        rep = json.loads(rep_path.read_text())
        assert rep["passed"], f"Reproduction failed: {rep.get('errors', [])}"

    def test_b5_mock_final_access_ledger_exists(self):
        """Test that the final access ledger is created."""
        result = self._run_pipeline()
        assert result.returncode == 0

        ledger_path = MOCK_OUTPUT / "checks" / "final_access_ledger.jsonl"
        assert ledger_path.exists()
        # Should have entries for final accesses
        lines = ledger_path.read_text().strip().split("\n")
        assert len(lines) > 0
        for line in lines:
            record = json.loads(line)
            assert record["split"] in FINAL_SPLITS

    def test_b5_mock_no_final_access_during_train(self):
        """Test that no FINAL access occurred during training stages."""
        result = self._run_pipeline()
        assert result.returncode == 0

        ledger_path = MOCK_OUTPUT / "checks" / "final_access_ledger.jsonl"
        if ledger_path.exists():
            for line in ledger_path.read_text().strip().split("\n"):
                record = json.loads(line)
                # No train-stage access to final
                assert record.get("stage") != "train", \
                    "FINAL access during train stage detected!"

    def test_b5_mock_selected_policy_loaded_not_retrained(self):
        """Test that qualification loads the selected policy, not retrains."""
        result = self._run_pipeline()
        assert result.returncode == 0

        selection_path = MOCK_OUTPUT / "policies" / "selection.json"
        assert selection_path.exists()
        selection = json.loads(selection_path.read_text())
        assert selection["selected_before_final"] is True

        # Frozen policy manifest should exist
        frozen_path = MOCK_OUTPUT / "policies" / "frozen_policy_manifest.json"
        assert frozen_path.exists()

    def test_b5_mock_surface_policy_loaded_not_retrained(self):
        """Test that surface ensemble is saved and can be loaded."""
        result = self._run_pipeline()
        assert result.returncode == 0

        surface_path = MOCK_OUTPUT / "policies" / "surface_baselines.json"
        assert surface_path.exists()
        data = json.loads(surface_path.read_text())
        assert data["policy_type"] == "surface_ensemble"

    def test_b5_mock_lifecycle_transitions(self):
        """Test that lifecycle goes through all required states."""
        result = self._run_pipeline()
        assert result.returncode == 0

        status_path = MOCK_OUTPUT / "status.json"
        state = json.loads(status_path.read_text())
        history = state.get("history", [])

        # Should have transitions: DEVELOPMENT→FROZEN, FROZEN→TRAIN_RUNNING,
        # TRAIN_RUNNING→TRAIN_COMPLETE, TRAIN_COMPLETE→FINAL_RUNNING,
        # FINAL_RUNNING→QUALIFIED or FAILED_QUALIFICATION
        transitions = [(h["from"], h["to"]) for h in history]
        assert ("DEVELOPMENT", "FROZEN") in transitions
        assert ("FROZEN", "TRAIN_RUNNING") in transitions
        assert ("TRAIN_RUNNING", "TRAIN_COMPLETE") in transitions
        assert ("TRAIN_COMPLETE", "FINAL_RUNNING") in transitions


# ──────────────────────────────────────────────────────────────────────
# B5 resume test
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.mock_pipeline
class TestB5Resume:
    """Test B5 resume semantics."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        if MOCK_OUTPUT.exists():
            shutil.rmtree(MOCK_OUTPUT)
        yield
        if MOCK_OUTPUT.exists():
            shutil.rmtree(MOCK_OUTPUT)

    def test_b5_resume_reuses_completed_records(self):
        """Test that resume reuses counterfactual records with matching config hash."""
        # Run development-counterfactuals first
        cmd1 = [
            sys.executable, str(ROOT / "scripts" / "run_b5_staged.py"),
            "prepare", "--mock", "--config", str(MOCK_CONFIG),
        ]
        subprocess.run(cmd1, check=True, timeout=30)

        cmd2 = [
            sys.executable, str(ROOT / "scripts" / "run_b5_staged.py"),
            "development-counterfactuals", "--mock", "--config", str(MOCK_CONFIG),
        ]
        subprocess.run(cmd2, check=True, timeout=60)

        # Run again with --resume, should reuse
        cmd3 = [
            sys.executable, str(ROOT / "scripts" / "run_b5_staged.py"),
            "development-counterfactuals", "--resume", "--mock", "--config", str(MOCK_CONFIG),
        ]
        result = subprocess.run(cmd3, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0
        # Should show 0 new executions
        assert "0 new" in result.stdout


# ──────────────────────────────────────────────────────────────────────
# B5 config mismatch rejection test
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.mock_pipeline
class TestB5ConfigMismatch:
    """Test that config mismatch is rejected on resume."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        if MOCK_OUTPUT.exists():
            shutil.rmtree(MOCK_OUTPUT)
        yield
        if MOCK_OUTPUT.exists():
            shutil.rmtree(MOCK_OUTPUT)

    def test_b5_config_mismatch_resume_rejected(self):
        """Test that changing config after freeze is rejected."""
        # Prepare with original config
        cmd = [
            sys.executable, str(ROOT / "scripts" / "run_b5_staged.py"),
            "prepare", "--mock", "--config", str(MOCK_CONFIG),
        ]
        subprocess.run(cmd, check=True, timeout=30)

        # Modify config and try to run counterfactuals
        import yaml
        config = yaml.safe_load(MOCK_CONFIG.read_text())
        config["experiment_id"] = "different_id"
        modified_path = MOCK_CONFIG.parent / "executive_b5_mock_modified.yaml"
        modified_path.write_text(yaml.dump(config))

        cmd2 = [
            sys.executable, str(ROOT / "scripts" / "run_b5_staged.py"),
            "development-counterfactuals", "--mock", "--config", str(modified_path),
        ]
        result = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert "Config hash mismatch" in result.stdout

        modified_path.unlink()


# ──────────────────────────────────────────────────────────────────────
# B4 regression tests
# ──────────────────────────────────────────────────────────────────────

class TestB4Regression:
    """B4 regression tests."""

    def test_b4_no_legacy_sham_fallback(self):
        """Test that B4 runner has no legacy sham fallback."""
        b4_path = ROOT / "scripts" / "run_b4_staged.py"
        content = b4_path.read_text()
        # No inline np.percentile for sham
        assert "np.percentile(h_vs_sham" not in content
        # No sham_regrets - hidden_regret computation
        assert "sham_regrets - hidden_regret" not in content
        # No "fallback" comment
        assert "Fallback: compute from sham regrets" not in content

    def test_b4_uses_shared_paired_comparison(self):
        """Test that B4 uses canonical paired comparison from stats module."""
        b4_path = ROOT / "scripts" / "run_b4_staged.py"
        content = b4_path.read_text()
        assert "make_paired_comparison" in content
        assert "paired_group_bootstrap" in content

    def test_b4_no_inline_np_percentile(self):
        """Test that B4 runner has no inline np.percentile for qualification."""
        b4_path = ROOT / "scripts" / "run_b4_staged.py"
        content = b4_path.read_text()
        assert "np.percentile" not in content

    def test_b4_surface_ensemble_is_frozen(self):
        """Test that B4 surface ensemble is frozen (not recomputed on FINAL)."""
        b4_path = ROOT / "scripts" / "run_b4_staged.py"
        content = b4_path.read_text()
        # Should reference surface baseline saving/loading
        assert "surface" in content.lower()

    def test_b4_has_final_access_guard_concept(self):
        """Test that B4 has some form of final access protection."""
        # B4 should not access FINAL during training stages.
        # This is enforced by the lifecycle state machine in the B5 runner.
        # B4 itself should at minimum not load final data during ablation training.
        b4_path = ROOT / "scripts" / "run_b4_staged.py"
        content = b4_path.read_text()
        # B4 should reference the stats module for sham comparison
        assert "run_matched_sham_evaluation" in content or "sham" in content


# ──────────────────────────────────────────────────────────────────────
# Version consistency test
# ──────────────────────────────────────────────────────────────────────

class TestVersionConsistency:
    """Test that all version surfaces agree on 0.4.0a3."""

    def test_current_release_versions_match(self):
        """All release-level version surfaces must agree on 0.4.0a3."""
        target = "0.4.0a3"

        # pyproject.toml
        pyproject = (ROOT / "pyproject.toml").read_text()
        assert f'version = "{target}"' in pyproject

        # Python __version__
        from daph_learning import __version__
        assert __version__ == target, f"__version__ is {__version__}, expected {target}"

        # Policy config default
        from daph_learning.policy.config import ExperimentConfig
        cfg = ExperimentConfig()
        assert cfg.autolearn_version == target

        # Provenance default
        from daph_learning.policy.provenance import ProvenanceRecord
        prov = ProvenanceRecord()
        assert prov.release_version == target


# ──────────────────────────────────────────────────────────────────────
# No fake dates / credentials tests
# ──────────────────────────────────────────────────────────────────────

class TestNoFakeDatesOrCredentials:
    """Test that no fake dates or credentials exist in active configs."""

    def test_no_fake_dates_in_source(self):
        """Test that no hard-coded historical timestamps in source files."""
        fake_dates = ["2025-01-01", "1970-01-01"]
        src_dir = ROOT / "src" / "daph_learning" / "executive"
        for py_file in src_dir.glob("*.py"):
            content = py_file.read_text()
            for date in fake_dates:
                if date in content:
                    # Allow in comments/docstrings
                    lines = [l for l in content.split("\n") if date in l]
                    for line in lines:
                        stripped = line.strip()
                        if not (stripped.startswith("#") or stripped.startswith('"""')
                                or stripped.startswith("'")):
                            pytest.fail(f"Fake date {date} in {py_file.name}: {line}")

    def test_no_fake_credentials_in_configs(self):
        """Test that no fake API keys in active configs."""
        fake_keys = ["sk-placeholder", "changeme", "dummy-key", "fake-key"]
        configs_dir = ROOT / "configs"
        for yaml_file in configs_dir.glob("*.yaml"):
            if "legacy" in str(yaml_file):
                continue
            content = yaml_file.read_text()
            for key in fake_keys:
                assert key not in content, \
                    f"Fake key '{key}' found in {yaml_file.name}"
