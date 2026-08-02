"""DAPH v0.4.0a3 — Crash-safe atomic I/O.

All critical JSON/NPZ artifacts must be written using atomic operations:
temporary file → fsync → atomic rename.

This prevents partial writes from corrupting authoritative evidence.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write JSON atomically: temp file → fsync → rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write to temp file in the same directory (required for atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(p.parent), prefix=f".{p.stem}_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write text atomically: temp file → fsync → rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(p.parent), prefix=f".{p.stem}_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_npz(path: str | Path, **arrays: Any) -> None:
    """Write NPZ atomically: temp file → fsync → rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Use a temp file name that doesn't conflict
    tmp_path = str(p.parent / f".{p.stem}_tmp_{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp_path, **arrays)
        # fsync the temp file
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write JSONL atomically: temp file → fsync → rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(p.parent), prefix=f".{p.stem}_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append a single record to a JSONL file.

    This is NOT atomic for the append itself, but ensures the file
    exists and the record is flushed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


__all__ = [
    "atomic_write_json",
    "atomic_write_text",
    "atomic_write_npz",
    "atomic_write_jsonl",
    "append_jsonl",
]
