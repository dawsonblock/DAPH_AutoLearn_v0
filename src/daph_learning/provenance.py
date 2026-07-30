"""v0.3.10.3.2-alpha — canonical provenance: source-tree hash + deterministic
seed derivation (Sections 2, 9).

This module provides the ONE canonical implementation of:

* :func:`compute_source_tree_sha256` — deterministic SHA-256 over the
  entire relevant source tree (src, tests, scripts, config files).
* :func:`deterministic_seed` — SHA-derived integer seed replacing
  Python's process-randomized ``hash()``.

All other modules that need a source-tree hash MUST delegate to
:func:`compute_source_tree_sha256` so that every artifact records the
same hash value for the same tree.

Hash rules (Section 2):

- deterministic ordering (sorted by normalized POSIX relative path)
- normalized POSIX relative paths (``as_posix()``)
- SHA-256, UTF-8 path encoding
- include relevant source, tests, scripts, config files
- exclude: .git, __pycache__, .pytest_cache, generated artifacts,
  temporary files, compiled bytecode

The canonical hash is the **full 64-character** SHA-256. UI/report
summaries may display the first 16 characters, but artifacts must store
the full hash.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


# ------------------------------------------------------------------
# Section 2: exclusion rules
# ------------------------------------------------------------------

# Directory names that are always excluded from the source-tree hash.
_EXCLUDED_DIR_NAMES: frozenset[str] = frozenset({
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "artifacts",
    "experiments",
    "extensions",
    ".venv",
    "venv",
    "env",
})

# File extensions that are always excluded (compiled bytecode, caches).
_EXCLUDED_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dylib",
    ".dll",
    ".egg-info",
})

# File name patterns that indicate generated/temporary artifacts.
_GENERATED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^release_gates\.json$"),
    re.compile(r"^experiment_results\.json$"),
    re.compile(r"^experiment_results_.*\.json$"),
    re.compile(r"^test_report.*\.(json|xml|txt)$"),
    re.compile(r"^junit.*\.xml$"),
    re.compile(r"^current_source_manifest\.json$"),
    re.compile(r"^freeze_manifest\.json$"),
    re.compile(r".*\.tmp$"),
    re.compile(r".*\.bak$"),
    re.compile(r".*\.swp$"),
    re.compile(r"^__results__.*$"),
)

# File extensions that ARE included in the hash (source, tests, scripts,
# config files).
_INCLUDED_EXTENSIONS: frozenset[str] = frozenset({
    ".py",
    ".toml",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".txt",
    ".sh",
    ".bash",
    ".rst",
})


def should_exclude(path: Path, root: Path) -> bool:
    """Section 2: determine whether a path should be excluded from the
    source-tree hash.

    Excludes:
    - .git, __pycache__, .pytest_cache, artifacts/, experiments/,
      extensions/, and other non-source directories
    - compiled bytecode (.pyc, .pyo, .so, .dll, .egg-info)
    - generated artifacts (release_gates.json, experiment_results.json,
      test reports, junit XML, freeze manifests, temp files)
    - files inside excluded directories
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True

    parts = rel.parts

    # Exclude if any path component is an excluded directory name.
    for part in parts:
        if part in _EXCLUDED_DIR_NAMES:
            return True

    # Exclude by extension.
    suffix = path.suffix.lower()
    if suffix in _EXCLUDED_EXTENSIONS:
        return True
    # .egg-info is a directory suffix, check parts.
    for part in parts:
        if part.endswith(".egg-info"):
            return True

    # Exclude generated/temporary file patterns.
    name = path.name
    for pattern in _GENERATED_PATTERNS:
        if pattern.match(name):
            return True

    # Only include known source/config file extensions.
    if suffix not in _INCLUDED_EXTENSIONS:
        return True

    return False


# ------------------------------------------------------------------
# Section 2: canonical source-tree hash
# ------------------------------------------------------------------

def compute_source_tree_sha256(root: Path) -> str:
    """Section 2: the ONE canonical source-tree SHA-256.

    Computes a deterministic SHA-256 hash over all relevant source,
    test, script, and config files in the repository rooted at
    ``root``.

    Hash rules:
    - deterministic ordering: files sorted by normalized POSIX relative
      path
    - normalized POSIX relative paths (``Path.as_posix()``)
    - SHA-256, UTF-8 path encoding
    - each file contributes: ``rel_path + \\0 + file_bytes + \\0``
    - excludes .git, __pycache__, .pytest_cache, generated artifacts,
      temporary files, compiled bytecode

    Returns the **full 64-character** SHA-256 hex digest.
    """
    repo = Path(root)
    if not repo.is_dir():
        raise FileNotFoundError(f"source tree root not found: {repo}")

    digest = hashlib.sha256()

    files = sorted(
        (p for p in repo.rglob("*")
         if p.is_file() and not should_exclude(p, repo)),
        key=lambda p: p.relative_to(repo).as_posix(),
    )

    for path in files:
        rel = path.relative_to(repo).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


# ------------------------------------------------------------------
# Section 9: deterministic seed derivation
# ------------------------------------------------------------------

def deterministic_seed(*parts: str) -> int:
    """Section 9: deterministic SHA-derived integer seed.

    Replaces Python's process-randomized ``hash()`` in experiment and
    benchmark provenance. The same inputs always produce the same seed
    across processes and Python invocations.

    Parameters
    ----------
    *parts : str
        Components that identify the seed context (e.g. split name,
        subtype, index). All parts are joined with ``"::"``.

    Returns
    -------
    int
        A non-negative integer derived from the first 8 bytes of the
        SHA-256 digest of the joined parts.
    """
    text = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


__all__ = [
    "compute_source_tree_sha256",
    "deterministic_seed",
    "should_exclude",
]
