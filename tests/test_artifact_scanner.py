"""Repository-wide artifact scanner tests.

Rejects:
* zero hashes in active artifacts
* placeholder hashes
* malformed manifest references
* active artifacts marked invalid
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"

# Directories that are explicitly invalid/historical and should not be scanned
EXEMPT_DIRS = {
    "invalidated",
    "invalid_fixtures",
    "legacy",
    "gate_a_failed",
    "gate_a_historical",
    # Test fixture created by test_staged_pipeline_integration.py
    "daph_gate_a_integration_test",
    # Mock test artifacts
    "executive_b5_mock_test",
}

ZERO_HASH = "0" * 64
PLACEHOLDER_HASHES = {ZERO_HASH, "", "placeholder", "tbd", "todo", "n/a"}


def _is_exempt(path: Path) -> bool:
    """Check if a path is under an exempt directory."""
    try:
        rel = path.relative_to(ARTIFACTS)
    except ValueError:
        return True
    parts = rel.parts
    return any(p in EXEMPT_DIRS for p in parts)


def _find_json_files(root: Path) -> list[Path]:
    """Find all JSON files in active artifact directories."""
    results = []
    if not root.exists():
        return results
    for p in root.rglob("*.json"):
        if not _is_exempt(p):
            results.append(p)
    return results


class TestArtifactScanner:
    """Repository-wide artifact scanner."""

    def test_no_zero_hashes_in_active_artifacts(self):
        """No active artifact may contain a zero hash."""
        violations = []
        for jf in _find_json_files(ARTIFACTS):
            try:
                text = jf.read_text()
            except Exception:
                continue
            if ZERO_HASH in text:
                violations.append(str(jf.relative_to(REPO_ROOT)))
        assert not violations, (
            f"Zero hashes found in active artifacts: {violations}"
        )

    def test_no_placeholder_hashes_in_active_artifacts(self):
        """No active artifact may contain placeholder hash values."""
        violations = []
        for jf in _find_json_files(ARTIFACTS):
            try:
                data = json.loads(jf.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            _check_placeholder_recursive(data, str(jf.relative_to(REPO_ROOT)), violations)
        assert not violations, (
            f"Placeholder hashes found in active artifacts: {violations}"
        )

    def test_no_invalidated_in_active_tree(self):
        """No active artifact directory should contain an invalidation.json."""
        # invalidation.json should only exist under invalidated/
        for jf in _find_json_files(ARTIFACTS):
            if jf.name == "invalidation.json":
                rel = jf.relative_to(ARTIFACTS)
                assert "invalidated" in rel.parts, (
                    f"invalidation.json found in active tree: {jf}"
                )

    def test_no_corrupted_artifacts_in_active_tree(self):
        """No active artifact should contain SSH/PTY corruption patterns."""
        corruption_patterns = [
            "Your SSH client doesn't support PTY",
            "PTY allocation request failed",
            "Permission denied (publickey)",
        ]
        violations = []
        for jf in _find_json_files(ARTIFACTS):
            try:
                text = jf.read_text()
            except Exception:
                continue
            for pattern in corruption_patterns:
                if pattern in text:
                    violations.append((str(jf.relative_to(REPO_ROOT)), pattern))
        assert not violations, (
            f"Corrupted artifacts found in active tree: {violations}"
        )

    def test_no_fake_api_keys_in_configs(self):
        """No config file should contain fake API keys."""
        from daph_learning.executive import check_api_key_placeholder
        configs_dir = REPO_ROOT / "configs"
        violations = []
        if configs_dir.exists():
            for cf in configs_dir.rglob("*.yaml"):
                text = cf.read_text()
                check = check_api_key_placeholder(text)
                if not check.passed:
                    violations.append((str(cf.relative_to(REPO_ROOT)), check.detail))
        assert not violations, (
            f"Fake API keys found in configs: {violations}"
        )


def _check_placeholder_recursive(obj, path: str, violations: list, key_prefix: str = ""):
    """Recursively check for placeholder hash values."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{key_prefix}.{k}" if key_prefix else k
            if "hash" in k.lower() or "sha" in k.lower():
                if isinstance(v, str) and v.lower() in PLACEHOLDER_HASHES:
                    violations.append(f"{path}:{kp}={v!r}")
            else:
                _check_placeholder_recursive(v, path, violations, kp)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_placeholder_recursive(v, path, violations, f"{key_prefix}[{i}]")
