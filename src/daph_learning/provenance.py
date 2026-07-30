"""v0.3.10.4-alpha — canonical provenance: source-tree hash + deterministic
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

import fnmatch
import hashlib
import json
import re
import sys
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
# Section 4.1: canonical source hash with explicit include/exclude globs
# ------------------------------------------------------------------

# Frozen default include/exclude globs for the canonical source hash.
# These are the ONE accepted configuration; all artifacts must record a
# source hash computed with these rules (or a strict superset documented
# at freeze time).
DEFAULT_INCLUDE_GLOBS: tuple[str, ...] = (
    "src/**/*.py",
    "scripts/**/*.py",
    "tests/**/*.py",
)
DEFAULT_EXCLUDE_GLOBS: tuple[str, ...] = (
    "**/__pycache__/**",
    "**/*.pyc",
    "artifacts/**",
    ".git/**",
    ".pytest_cache/**",
    "build/**",
    "dist/**",
)


def _matches_any(rel_posix: str, patterns: tuple[str, ...]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(rel_posix, pat):
            return True
    return False


def _normalize_line_endings(data: bytes) -> bytes:
    """Section 4.1: normalize CRLF/CR to LF so the hash is independent of
    the operating system that produced the file."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def compute_canonical_source_hash(
    repo_root: Path,
    *,
    include_globs: tuple[str, ...] = DEFAULT_INCLUDE_GLOBS,
    exclude_globs: tuple[str, ...] = DEFAULT_EXCLUDE_GLOBS,
) -> str:
    """Section 4.1: the ONE canonical source hash.

    Deterministic SHA-256 over the repository source tree with:

    * deterministic file ordering (sorted by normalized POSIX relative path);
    * normalized relative paths (``as_posix()``);
    * normalized line endings (CRLF/CR -> LF);
    * file path included in the digest;
    * file content included in the digest;
    * explicit include/exclude glob rules;
    * no timestamps;
    * no environment-dependent metadata.

    Each file contributes::

        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\\0")
        digest.update(normalized_bytes)
        digest.update(b"\\0")

    Returns the full 64-character SHA-256 hex digest.
    """
    repo = Path(repo_root)
    if not repo.is_dir():
        raise FileNotFoundError(f"source tree root not found: {repo}")

    selected: list[Path] = []
    for glob in include_globs:
        for path in repo.glob(glob):
            if path.is_file():
                selected.append(path)

    # Deduplicate and sort by normalized POSIX relative path.
    seen: set[str] = set()
    files: list[Path] = []
    for path in selected:
        rel = path.relative_to(repo).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        files.append(path)
    files.sort(key=lambda p: p.relative_to(repo).as_posix())

    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(repo).as_posix()
        if _matches_any(rel, exclude_globs):
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_normalize_line_endings(path.read_bytes()))
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
    "compute_canonical_source_hash",
    "deterministic_seed",
    "should_exclude",
    "DEFAULT_INCLUDE_GLOBS",
    "DEFAULT_EXCLUDE_GLOBS",
]


# ------------------------------------------------------------------
# Section 4.1: CLI -- `python -m daph_learning.provenance source-hash`
# ------------------------------------------------------------------

def _cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point. Prints only the canonical hash by default.

    Usage::

        python -m daph_learning.provenance source-hash
        python -m daph_learning.provenance source-hash --json
        python -m daph_learning.provenance source-hash --repo-root /path
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print("usage: python -m daph_learning.provenance source-hash [--json] [--repo-root PATH]")
        return 0

    command = args[0]
    if command != "source-hash":
        print(f"unknown command: {command!r}", file=sys.stderr)
        print("usage: python -m daph_learning.provenance source-hash [--json] [--repo-root PATH]", file=sys.stderr)
        return 2

    as_json = False
    repo_root = Path.cwd()
    rest = args[1:]
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--json":
            as_json = True
            i += 1
        elif tok == "--repo-root":
            if i + 1 >= len(rest):
                print("--repo-root requires a path argument", file=sys.stderr)
                return 2
            repo_root = Path(rest[i + 1])
            i += 2
        else:
            print(f"unknown option: {tok!r}", file=sys.stderr)
            return 2

    try:
        h = compute_canonical_source_hash(repo_root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps({
            "source_hash": h,
            "repo_root": str(repo_root),
            "include_globs": list(DEFAULT_INCLUDE_GLOBS),
            "exclude_globs": list(DEFAULT_EXCLUDE_GLOBS),
            "hash_algorithm": "SHA-256",
            "hash_length": 64,
        }, indent=2))
    else:
        print(h)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
