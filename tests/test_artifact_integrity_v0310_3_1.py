"""v0.3.10.3.1-alpha — Section 32-35: artifact directory discipline,
source hash enforcement, test collection hash, re-run test report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daph_learning.evaluation.artifact_integrity import (
    POLICY_ARTIFACT_FILES,
    RESULT_FILES,
    assert_source_tree_matches,
    compute_source_tree_hash,
    compute_test_collection_hash,
    rerun_tests,
    validate_policy_artifact_dir,
    validate_result_dir,
)


REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"


# --- Section 32: artifact directory discipline ---

def test_policy_artifact_dir_validates_missing(tmp_path):
    """An empty dir is missing all expected files."""
    missing = validate_policy_artifact_dir(tmp_path / "policy_x")
    assert set(missing) == set(POLICY_ARTIFACT_FILES)


def test_policy_artifact_dir_complete(tmp_path):
    d = tmp_path / "policy_y"
    d.mkdir()
    for f in POLICY_ARTIFACT_FILES:
        (d / f).write_text("{}")
    assert validate_policy_artifact_dir(d) == []


def test_result_dir_validates_missing(tmp_path):
    missing = validate_result_dir(tmp_path / "exp_z")
    assert set(missing) == set(RESULT_FILES)


def test_result_dir_complete(tmp_path):
    d = tmp_path / "exp_w"
    d.mkdir()
    for f in RESULT_FILES:
        (d / f).write_text("{}")
    assert validate_result_dir(d) == []


# --- Section 33: source tree hash ---

def test_source_tree_hash_deterministic():
    h1 = compute_source_tree_hash(SRC)
    h2 = compute_source_tree_hash(SRC)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_source_tree_hash_changes_on_edit(tmp_path):
    """A different source tree produces a different hash."""
    src_a = tmp_path / "a" / "src"
    src_a.mkdir(parents=True)
    (src_a / "mod.py").write_text("x = 1\n")
    src_b = tmp_path / "b" / "src"
    src_b.mkdir(parents=True)
    (src_b / "mod.py").write_text("x = 2\n")
    assert compute_source_tree_hash(src_a) != compute_source_tree_hash(src_b)


def test_assert_source_tree_matches_strict_raises():
    h = compute_source_tree_hash(SRC)
    with pytest.raises(RuntimeError, match="mismatch"):
        assert_source_tree_matches("0" * 64, SRC, strict=True)


def test_assert_source_tree_matches_non_strict_warns(capsys):
    h = compute_source_tree_hash(SRC)
    assert_source_tree_matches("0" * 64, SRC, strict=False)  # no raise
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_assert_source_tree_matches_self_ok():
    h = compute_source_tree_hash(SRC)
    assert_source_tree_matches(h, SRC, strict=True)  # no raise


# --- Section 34: test collection hash ---

def test_test_collection_hash_deterministic():
    h1 = compute_test_collection_hash(REPO)
    h2 = compute_test_collection_hash(REPO)
    assert h1 == h2
    assert len(h1) == 64


# --- Section 35: re-run test report ---

def test_rerun_tests_runs_and_reports():
    """Section 35: rerun_tests executes pytest and reports counts.

    Uses a small, fast test file (NOT this one — that would recurse)
    to keep the test fast; the function supports the full suite via
    the default ``test_paths=None``.
    """
    report = rerun_tests(
        REPO, SRC,
        test_paths=["tests/test_stage_access_v0310_3_1.py"],
        timeout=120)
    assert report.all_passed or report.n_passed > 0
    assert report.n_passed > 0
    assert report.collection_hash != ""
    assert report.source_tree_hash != ""
