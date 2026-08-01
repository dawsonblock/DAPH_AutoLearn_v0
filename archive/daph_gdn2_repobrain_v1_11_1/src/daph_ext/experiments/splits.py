from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence


def _normalize_ids(values: tuple[str, ...], split_name: str) -> tuple[str, ...]:
    normalized = tuple(v.strip() for v in values)
    if any(not v for v in normalized):
        raise ValueError(f"{split_name} contains an empty repository identifier")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{split_name} contains duplicate repository identifiers")
    return normalized


@dataclass(frozen=True)
class ImmutableRepoSplits:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def normalized(self) -> "ImmutableRepoSplits":
        return ImmutableRepoSplits(
            _normalize_ids(self.train, "train"),
            _normalize_ids(self.validation, "validation"),
            _normalize_ids(self.test, "test"),
        )

    def validate(self) -> None:
        normalized = self.normalized()
        groups = {
            "train": set(normalized.train),
            "validation": set(normalized.validation),
            "test": set(normalized.test),
        }
        if not all(groups.values()):
            raise ValueError("train/validation/test splits must all be non-empty")
        if groups["train"] & groups["validation"]:
            raise ValueError("train and validation repositories overlap")
        if groups["train"] & groups["test"]:
            raise ValueError("train and test repositories overlap")
        if groups["validation"] & groups["test"]:
            raise ValueError("validation and test repositories overlap")

    def canonical_json(self) -> str:
        normalized = self.normalized()
        groups = {
            "train": set(normalized.train),
            "validation": set(normalized.validation),
            "test": set(normalized.test),
        }
        if not all(groups.values()):
            raise ValueError("train/validation/test splits must all be non-empty")
        if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
            raise ValueError("repository splits overlap")
        payload = {
            "train": sorted(normalized.train),
            "validation": sorted(normalized.validation),
            "test": sorted(normalized.test),
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def write(self, path: str | Path) -> str:
        text = self.canonical_json()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def from_sequences(
        cls,
        train: Sequence[str],
        validation: Sequence[str],
        test: Sequence[str],
    ) -> "ImmutableRepoSplits":
        for name, values in (("train", train), ("validation", validation), ("test", test)):
            if isinstance(values, (str, bytes)):
                raise TypeError(f"{name} must be a sequence of repository identifiers, not a string")
            if any(not isinstance(v, str) for v in values):
                raise TypeError(f"{name} repository identifiers must all be strings")
        return cls(tuple(train), tuple(validation), tuple(test)).normalized()


@dataclass(frozen=True)
class RepositoryIdentity:
    repo_id: str
    split: str
    content_hash: str | None = None
    fork_root: str | None = None
    family: str | None = None


def validate_repository_lineage(records: Sequence[RepositoryIdentity]) -> None:
    """Reject obvious cross-split leakage through forks/families/duplicate code."""
    allowed = {"train", "validation", "test"}
    seen_repo: dict[str, str] = {}
    for r in records:
        if r.split not in allowed or not r.repo_id.strip():
            raise ValueError("invalid repository identity record")
        prev = seen_repo.setdefault(r.repo_id, r.split)
        if prev != r.split:
            raise ValueError(f"repository {r.repo_id} appears in multiple splits")
    for field in ("content_hash", "fork_root", "family"):
        owners: dict[str, str] = {}
        for r in records:
            value = getattr(r, field)
            if not value:
                continue
            prev = owners.setdefault(value, r.split)
            if prev != r.split:
                raise ValueError(f"cross-split leakage detected via {field}={value!r}")
