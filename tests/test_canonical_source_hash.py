"""v0.3.10.3.2-alpha — Section 2 / G02: canonical source-tree hash.

Verifies that all provenance modules return identical source hashes
and that the canonical hash is deterministic, full-length (64 chars),
and changes when source files change.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_canonical_hash_is_64_chars():
    from daph_learning.provenance import compute_source_tree_sha256
    h = compute_source_tree_sha256(REPO_ROOT)
    assert len(h) == 64, f"canonical hash is {len(h)} chars, expected 64"
    assert all(c in "0123456789abcdef" for c in h), (
        f"canonical hash contains non-hex characters: {h!r}")


def test_canonical_hash_is_deterministic():
    from daph_learning.provenance import compute_source_tree_sha256
    h1 = compute_source_tree_sha256(REPO_ROOT)
    h2 = compute_source_tree_sha256(REPO_ROOT)
    assert h1 == h2, "canonical hash is not deterministic"


def test_legacy_provenance_delegates_to_canonical():
    """policy.provenance.source_tree_sha256 must return the same value
    as the canonical implementation (Section 2)."""
    from daph_learning.provenance import compute_source_tree_sha256
    from daph_learning.policy.provenance import source_tree_sha256
    canonical = compute_source_tree_sha256(REPO_ROOT)
    legacy = source_tree_sha256(REPO_ROOT)
    assert legacy == canonical, (
        f"legacy source_tree_sha256={legacy!r} != canonical={canonical!r}")


def test_artifact_integrity_delegates_to_canonical():
    """evaluation.artifact_integrity.compute_source_tree_hash must
    return the same value as the canonical implementation (Section 2)."""
    from daph_learning.provenance import compute_source_tree_sha256
    from daph_learning.evaluation.artifact_integrity import (
        compute_source_tree_hash,
    )
    canonical = compute_source_tree_sha256(REPO_ROOT)
    legacy = compute_source_tree_hash(REPO_ROOT / "src")
    assert legacy == canonical, (
        f"artifact_integrity hash={legacy!r} != canonical={canonical!r}")


def test_canonical_hash_excludes_pycache():
    """__pycache__ directories must not affect the hash."""
    from daph_learning.provenance import compute_source_tree_sha256, should_exclude
    # Create a temp tree with a __pycache__ file.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "mod.py").write_text("x = 1\n")
        (root / "src" / "__pycache__").mkdir()
        (root / "src" / "__pycache__" / "mod.cpython-310.pyc").write_text("bytecode")
        h1 = compute_source_tree_sha256(root)
        # Modify the pyc file — hash should not change.
        (root / "src" / "__pycache__" / "mod.cpython-310.pyc").write_text("changed")
        h2 = compute_source_tree_sha256(root)
        assert h1 == h2, "hash changed when only __pycache__ content changed"
        # Verify should_exclude works.
        pyc = root / "src" / "__pycache__" / "mod.cpython-310.pyc"
        assert should_exclude(pyc, root), f"should_exclude({pyc}) returned False"


def test_canonical_hash_changes_with_source():
    """Changing a source file must change the hash."""
    from daph_learning.provenance import compute_source_tree_sha256
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "mod.py").write_text("x = 1\n")
        h1 = compute_source_tree_sha256(root)
        (root / "src" / "mod.py").write_text("x = 2\n")
        h2 = compute_source_tree_sha256(root)
        assert h1 != h2, "hash did not change when source file changed"


def test_canonical_hash_excludes_artifacts_dir():
    """artifacts/ directory must not affect the source hash."""
    from daph_learning.provenance import compute_source_tree_sha256
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "mod.py").write_text("x = 1\n")
        (root / "artifacts").mkdir()
        (root / "artifacts" / "result.json").write_text('{"x": 1}')
        h1 = compute_source_tree_sha256(root)
        (root / "artifacts" / "result.json").write_text('{"x": 2}')
        h2 = compute_source_tree_sha256(root)
        assert h1 == h2, "hash changed when only artifacts/ content changed"


def test_deterministic_seed_is_deterministic():
    """Section 9: deterministic_seed must return the same value for
    the same inputs across calls."""
    from daph_learning.provenance import deterministic_seed
    s1 = deterministic_seed("train", "A", "0")
    s2 = deterministic_seed("train", "A", "0")
    assert s1 == s2, "deterministic_seed is not deterministic"
    assert s1 >= 0, "deterministic_seed returned negative value"
    assert isinstance(s1, int), "deterministic_seed must return int"


def test_deterministic_seed_differs_for_different_inputs():
    from daph_learning.provenance import deterministic_seed
    s1 = deterministic_seed("train", "A", "0")
    s2 = deterministic_seed("train", "B", "0")
    s3 = deterministic_seed("dev", "A", "0")
    assert s1 != s2, "seed same for different subtypes"
    assert s1 != s3, "seed same for different splits"
