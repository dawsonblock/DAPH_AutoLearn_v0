"""Pytest configuration — ensures tests can collect without PYTHONPATH=.

Adds the repository root to sys.path so that tests importing from
``scripts.*`` can collect under a plain ``pytest`` invocation (without
needing ``PYTHONPATH=src:.``). This is a transitional measure until
all script logic is migrated into ``src/daph_learning/`` (v0.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _REPO_ROOT / "src"

for _p in (str(_SRC_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
