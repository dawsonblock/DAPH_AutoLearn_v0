"""v0.3.10.3.1-alpha — Section 16-17: evaluate_routes CLI enforces
stage access control for final split."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "daph_learning.cli.commands.evaluate_routes",
         *args],
        cwd=REPO, capture_output=True, text=True, timeout=30,
        env={"PYTHONPATH": str(SRC), "PATH": ""})


def test_evaluate_routes_rejects_final_before_frozen(tmp_path):
    """Section 14 / G10: final split rejected when stage != frozen.

    The existing assert_split_access rejects final_evaluation without
    --configuration-frozen. With --configuration-frozen set, the NEW
    StageGuard rejects when --stage is not 'frozen'.
    """
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps({"task_id": "t1"}) + "\n")
    routes_path = tmp_path / "routes.jsonl"
    routes_path.write_text(json.dumps({"task_id": "t1"}) + "\n")
    # First: without --configuration-frozen, the existing guard rejects.
    result = _run_cli(
        "--tasks", str(tasks_path),
        "--routes", str(routes_path),
        "--dataset-split", "final_test",
        "--protocol-purpose", "final_evaluation",
        "--final-test-access-sha256", "a" * 64,
        "--stage", "train",
        "--no-manifest",
    )
    assert result.returncode != 0
    # Now with --configuration-frozen but stage=train, the NEW StageGuard
    # should reject because stage != frozen.
    result2 = _run_cli(
        "--tasks", str(tasks_path),
        "--routes", str(routes_path),
        "--dataset-split", "final_test",
        "--protocol-purpose", "final_evaluation",
        "--final-test-access-sha256", "a" * 64,
        "--configuration-frozen",
        "--stage", "train",
        "--no-manifest",
    )
    assert result2.returncode != 0
    assert "FROZEN" in result2.stderr or "frozen" in result2.stderr.lower()


def test_evaluate_routes_unknown_stage_rejected(tmp_path):
    """Section 14: unknown stage value is rejected."""
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps({"task_id": "t1"}) + "\n")
    routes_path = tmp_path / "routes.jsonl"
    routes_path.write_text(json.dumps({"task_id": "t1"}) + "\n")
    result = _run_cli(
        "--tasks", str(tasks_path),
        "--routes", str(routes_path),
        "--dataset-split", "final_test",
        "--protocol-purpose", "final_evaluation",
        "--final-test-access-sha256", "a" * 64,
        "--configuration-frozen",
        "--stage", "bogus",
        "--no-manifest",
    )
    assert result.returncode != 0
