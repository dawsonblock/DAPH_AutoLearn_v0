from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any


class DataOrigin(str, Enum):
    EMPIRICAL = "empirical"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class ExperimentProvenance:
    data_origin: DataOrigin
    experiment_name: str
    git_commit: str | None = None
    checkpoint: str | None = None
    checkpoint_revision: str | None = None
    dataset_hash: str | None = None
    seed: int | None = None
    notes: str = ""

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if self.data_origin is DataOrigin.EMPIRICAL:
            if not self.checkpoint or not self.checkpoint.strip():
                raise ValueError("empirical provenance requires checkpoint")
            if not self.dataset_hash or not self.dataset_hash.strip():
                raise ValueError("empirical provenance requires dataset_hash")
            if self.seed is None or not isinstance(self.seed, int) or isinstance(self.seed, bool):
                raise ValueError("empirical provenance requires integer seed")

    def write_json(self, path: str | Path, payload: dict[str, Any] | None = None) -> None:
        self.validate()
        out = asdict(self)
        out["data_origin"] = self.data_origin.value
        if payload is not None:
            out["results"] = payload
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
