"""DAPH v0.4.0a2 — Reproduction hardening tests.

Tests for:
* Corrupted reproduction artifact
* Config change after freeze
* Missing files
* Hash mismatch
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from daph_learning.executive.synthetic_pipeline import run_synthetic_experiment


@pytest.fixture(scope="class")
def base_experiment(tmp_path_factory):
    """Run a complete synthetic experiment to use as base for reproduction tests."""
    root = tmp_path_factory.mktemp("repro_base")
    result = run_synthetic_experiment(
        root, n_train=120, n_dev=60, n_final=80, n_ood=40,
        n_shams=5, bootstrap_replicates=200, control_mode="positive",
    )
    return root, result


class TestReproductionCorruption:
    """Tests that reproduction detects corrupted artifacts."""

    def test_clean_reproduction_passes(self, base_experiment):
        from daph_learning.executive.reproduce import reproduce
        root, _ = base_experiment
        result = reproduce(root)
        assert result["passed"], f"Clean reproduction should pass: {result.get('errors', [])}"

    def test_corrupted_qualification_detected(self, base_experiment):
        from daph_learning.executive.reproduce import reproduce
        root, _ = base_experiment
        qual_path = root / "qualification" / "qualification.json"
        original = qual_path.read_text()
        # Corrupt the file
        qual_path.write_text('{"corrupted": true, "broken json')
        try:
            result = reproduce(root)
            assert not result["passed"], "Corrupted qualification should fail reproduction"
        finally:
            qual_path.write_text(original)

    def test_missing_counterfactuals_detected(self, base_experiment):
        from daph_learning.executive.reproduce import reproduce
        root, _ = base_experiment
        cf_path = root / "counterfactuals" / "final.json"
        if cf_path.exists():
            cf_path.unlink()
            try:
                result = reproduce(root)
                assert not result["passed"], "Missing counterfactuals should fail reproduction"
            finally:
                # Restore by re-running (skip for now, fixture is class-scoped)
                pass

    def test_corrupted_manifest_detected(self, base_experiment):
        from daph_learning.executive.reproduce import reproduce
        root, _ = base_experiment
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            original = manifest_path.read_text()
            manifest_path.write_text('{"broken": true')
            try:
                result = reproduce(root)
                assert not result["passed"], "Corrupted manifest should fail reproduction"
            finally:
                manifest_path.write_text(original)


class TestReproductionConfigChange:
    """Tests that reproduction detects config changes after freeze."""

    def test_config_hash_mismatch_detected(self, base_experiment):
        from daph_learning.executive.reproduce import reproduce
        root, _ = base_experiment
        config_path = root / "config" / "experiment_config.json"
        if config_path.exists():
            original = config_path.read_text()
            config = json.loads(original)
            config["experiment_id"] = "modified_after_freeze"
            config_path.write_text(json.dumps(config, indent=2))
            try:
                result = reproduce(root)
                # Reproduction should detect the config hash mismatch
                assert not result["passed"] or "config" in str(result.get("errors", "")).lower(), \
                    "Config hash mismatch should be detected"
            finally:
                config_path.write_text(original)


class TestReproductionMissingFiles:
    """Tests that reproduction detects missing required files."""

    def test_missing_dataset_detected(self, base_experiment):
        from daph_learning.executive.reproduce import reproduce
        root, _ = base_experiment
        ds_path = root / "dataset" / "final.json"
        if ds_path.exists():
            ds_path.unlink()
            try:
                result = reproduce(root)
                assert not result["passed"], "Missing dataset should fail reproduction"
            finally:
                pass  # fixture cleanup will handle it

    def test_missing_representations_detected(self, base_experiment):
        from daph_learning.executive.reproduce import reproduce
        root, _ = base_experiment
        rep_path = root / "representations" / "final.npz"
        if rep_path.exists():
            rep_path.unlink()
            try:
                result = reproduce(root)
                assert not result["passed"], "Missing representations should fail reproduction"
            finally:
                pass


class TestReproductionHashMismatch:
    """Tests that reproduction detects hash mismatches in artifacts."""

    def test_modified_counterfactual_hash_detected(self, base_experiment):
        from daph_learning.executive.reproduce import reproduce
        root, _ = base_experiment
        cf_path = root / "counterfactuals" / "final.json"
        if cf_path.exists():
            original = cf_path.read_text()
            # Modify the content (changes the hash)
            data = json.loads(original)
            first_key = list(data.keys())[0]
            data[first_key][list(data[first_key].keys())[0]]["utility"] = 999.0
            cf_path.write_text(json.dumps(data, indent=2))
            try:
                result = reproduce(root)
                # Hash mismatch should be detected
                assert not result["passed"] or "hash" in str(result.get("errors", "")).lower(), \
                    "Hash mismatch should be detected"
            finally:
                cf_path.write_text(original)
