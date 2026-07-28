"""Scientific split-access policy and one-shot final-test ledger."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import re
from typing import Any


class ProtocolPurpose(str, Enum):
    VECTOR_CAPTURE = "vector_capture"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    CALIBRATION = "calibration"
    MODEL_SELECTION = "model_selection"
    ENGINEERING_EVAL = "engineering_eval"
    FINAL_EVALUATION = "final_evaluation"


FINAL_SPLITS = frozenset({"test", "final_test"})
_FORBIDDEN_FINAL_PURPOSES = frozenset(
    {
        ProtocolPurpose.VECTOR_CAPTURE,
        ProtocolPurpose.HYPERPARAMETER_TUNING,
        ProtocolPurpose.CALIBRATION,
        ProtocolPurpose.MODEL_SELECTION,
        ProtocolPurpose.ENGINEERING_EVAL,
    }
)


class ProtocolViolation(RuntimeError):
    pass


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require_sha256(name: str, value: Any) -> str:
    normalized = str(value).lower() if isinstance(value, str) else ""
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return normalized


def assert_split_access(
    split: str,
    purpose: ProtocolPurpose | str,
    *,
    configuration_frozen: bool = False,
) -> None:
    """Reject test data in every adaptive or exploratory workflow."""
    try:
        parsed = purpose if isinstance(purpose, ProtocolPurpose) else ProtocolPurpose(purpose)
    except ValueError as exc:
        raise ProtocolViolation(f"unknown protocol purpose: {purpose!r}") from exc
    if split in FINAL_SPLITS:
        if parsed in _FORBIDDEN_FINAL_PURPOSES:
            raise ProtocolViolation(
                f"split {split!r} cannot be used for {parsed.value!r}"
            )
        if parsed is ProtocolPurpose.FINAL_EVALUATION and not configuration_frozen:
            raise ProtocolViolation(
                "final evaluation requires a frozen configuration"
            )


@dataclass(frozen=True)
class FinalTestAccess:
    schema: str
    run_id: str
    dataset_sha256: str
    configuration_sha256: str
    created_at: str
    purpose: str = ProtocolPurpose.FINAL_EVALUATION.value


def reserve_final_test(
    ledger_path: str | Path,
    *,
    run_id: str,
    dataset_sha256: str,
    configuration_sha256: str,
) -> FinalTestAccess:
    """Atomically burn a final-test split for one frozen configuration.

    The ledger is created with ``O_EXCL``.  A crash after reservation still
    counts as a test touch; operators must create a new final split rather than
    deleting the ledger and pretending the observation never happened.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id is missing or malformed")
    dataset_sha256 = _require_sha256("dataset_sha256", dataset_sha256)
    configuration_sha256 = _require_sha256(
        "configuration_sha256",
        configuration_sha256,
    )
    assert_split_access(
        "final_test",
        ProtocolPurpose.FINAL_EVALUATION,
        configuration_frozen=True,
    )

    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    access = FinalTestAccess(
        schema="daph.final_test_access.v1",
        run_id=run_id,
        dataset_sha256=dataset_sha256,
        configuration_sha256=configuration_sha256,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    payload = json.dumps(
        asdict(access),
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ProtocolViolation(
            f"final-test ledger already exists at {path}; this split is retired"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The reservation remains burned even on a partial write.
        raise
    return access


def load_final_test_access(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "daph.final_test_access.v1":
        raise ValueError("unsupported final-test ledger schema")
    if payload.get("purpose") != ProtocolPurpose.FINAL_EVALUATION.value:
        raise ValueError("final-test ledger has an invalid purpose")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"].strip():
        raise ValueError("final-test ledger is missing run_id")
    _require_sha256("dataset_sha256", payload.get("dataset_sha256"))
    _require_sha256(
        "configuration_sha256",
        payload.get("configuration_sha256"),
    )
    return payload


__all__ = [
    "FINAL_SPLITS",
    "FinalTestAccess",
    "ProtocolPurpose",
    "ProtocolViolation",
    "assert_split_access",
    "load_final_test_access",
    "reserve_final_test",
]
