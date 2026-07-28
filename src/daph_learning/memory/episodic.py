"""v0.3.9 (V039-010) — retrievable, revisable episodic memory.

Item 6 of the recommended repair sequence: build memory as a **separate
retrievable, revisable system**. This module implements an episodic memory
that:

* **retrieves** records by key, by metadata filter, or by cosine similarity
  over optional embeddings (the retrieval surface for the learning loop and
  the experience-replay buffer);
* **revises** records by appending a new revision while retaining prior
  revisions for auditability — a record is never silently overwritten;
* **hashes** every revision so memory contents are tamper-evident and can be
  coupled into the experience-ledger provenance.

This is deliberately a self-contained, dependency-light store (stdlib +
numpy). It is the memory substrate; the experience-replay buffer
(V039-010) and the RISER-style vector library live alongside it in
:mod:`daph_learning.memory.vector_library`.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_record_payload(payload: Mapping[str, Any]) -> str:
    return _sha256_hex(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    )


@dataclass(frozen=True)
class EpisodicMemoryRecord:
    """One revision of one memory entry.

    ``revision`` starts at 1 and increments on each revise. ``record_hash``
    couples the key, content, embedding hash, metadata, revision, and
    parent hash so any change is detectable. Prior revisions are retained
    by :class:`EpisodicMemory` so the revision history is auditable.
    """

    key: str
    revision: int
    content: Any
    embedding: tuple[float, ...] | None
    embedding_hash: str | None
    metadata: Mapping[str, Any]
    parent_hash: str | None
    record_hash: str
    created_at: str

    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["embedding"] = list(self.embedding) if self.embedding is not None else None
        d["metadata"] = dict(self.metadata)
        return d


def _embedding_hash(embedding: Sequence[float] | None) -> str | None:
    if embedding is None:
        return None
    arr = np.asarray(embedding, dtype=np.float32)
    return _sha256_hex(arr.tobytes())


def build_record(
    *,
    key: str,
    content: Any,
    embedding: Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
    revision: int = 1,
    parent_hash: str | None = None,
    created_at: str | None = None,
) -> EpisodicMemoryRecord:
    """Construct a record with a computed coupling hash."""
    if created_at is None:
        created_at = str(int(time.time()))
    emb_hash = _embedding_hash(embedding)
    payload = {
        "key": key,
        "revision": revision,
        "content": content,
        "embedding_hash": emb_hash,
        "metadata": dict(metadata or {}),
        "parent_hash": parent_hash,
        "created_at": created_at,
    }
    record_hash = _hash_record_payload(payload)
    return EpisodicMemoryRecord(
        key=key, revision=revision, content=content,
        embedding=tuple(embedding) if embedding is not None else None,
        embedding_hash=emb_hash, metadata=dict(metadata or {}),
        parent_hash=parent_hash, record_hash=record_hash, created_at=created_at,
    )


class EpisodicMemory:
    """A retrievable, revisable episodic memory store.

    Records are keyed by ``key``. ``store`` inserts a new record (revision 1
    or the next revision if the key exists). ``revise`` appends a new
    revision, retaining the old one. ``retrieve`` returns the latest
    revision; ``history`` returns the full revision chain. ``retrieve_similar``
    returns the top-k records by cosine similarity over embeddings.
    """

    def __init__(self) -> None:
        self._records: dict[str, list[EpisodicMemoryRecord]] = {}

    def store(
        self,
        key: str,
        content: Any,
        *,
        embedding: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EpisodicMemoryRecord:
        if key in self._records:
            return self.revise(key, content, embedding=embedding, metadata=metadata)
        record = build_record(key=key, content=content, embedding=embedding,
                              metadata=metadata, revision=1, parent_hash=None)
        self._records[key] = [record]
        return record

    def revise(
        self,
        key: str,
        content: Any,
        *,
        embedding: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EpisodicMemoryRecord:
        if key not in self._records:
            raise KeyError(f"no memory record for key {key!r}; use store() first")
        prior = self._records[key][-1]
        # carry forward metadata that the caller did not override
        merged_meta = dict(prior.metadata)
        merged_meta.update(metadata or {})
        # carry forward the prior embedding when the caller does not provide
        # a new one, so retrieve_similar continues to find revised records.
        effective_embedding = embedding if embedding is not None else prior.embedding
        record = build_record(
            key=key, content=content, embedding=effective_embedding, metadata=merged_meta,
            revision=prior.revision + 1, parent_hash=prior.record_hash,
        )
        self._records[key].append(record)
        return record

    def retrieve(self, key: str) -> EpisodicMemoryRecord | None:
        revs = self._records.get(key)
        return revs[-1] if revs else None

    def history(self, key: str) -> list[EpisodicMemoryRecord]:
        return list(self._records.get(key, []))

    def keys(self) -> list[str]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: object) -> bool:
        return key in self._records

    def retrieve_by_metadata(self, **filters: Any) -> list[EpisodicMemoryRecord]:
        """Return latest revisions whose metadata matches all filters."""
        out: list[EpisodicMemoryRecord] = []
        for revs in self._records.values():
            rec = revs[-1]
            if all(rec.metadata.get(k) == v for k, v in filters.items()):
                out.append(rec)
        return out

    def retrieve_similar(
        self,
        query: Sequence[float],
        *,
        limit: int = 5,
        min_cosine: float = -1.0,
    ) -> list[tuple[float, EpisodicMemoryRecord]]:
        """Cosine-similarity retrieval over records that carry embeddings.

        Returns ``(cosine, record)`` pairs, descending, for records whose
        cosine is >= ``min_cosine``. Records without embeddings are skipped.
        """
        q = np.asarray(query, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        scored: list[tuple[float, EpisodicMemoryRecord]] = []
        for revs in self._records.values():
            rec = revs[-1]
            if rec.embedding is None:
                continue
            v = np.asarray(rec.embedding, dtype=np.float32)
            v_norm = float(np.linalg.norm(v))
            if v_norm == 0.0:
                continue
            cos = float(np.dot(q, v) / (q_norm * v_norm))
            if cos >= min_cosine:
                scored.append((cos, rec))
        scored.sort(key=lambda x: (-x[0], x[1].key))
        return scored[:limit]

    def verify_integrity(self) -> None:
        """Recompute every record hash and assert it matches; assert the
        parent-hash chain is consistent across revisions."""
        for key, revs in self._records.items():
            parent: str | None = None
            for i, rec in enumerate(revs):
                payload = {
                    "key": rec.key,
                    "revision": rec.revision,
                    "content": rec.content,
                    "embedding_hash": rec.embedding_hash,
                    "metadata": dict(rec.metadata),
                    "parent_hash": rec.parent_hash,
                    "created_at": rec.created_at,
                }
                expected = _hash_record_payload(payload)
                if expected != rec.record_hash:
                    raise ValueError(
                        f"memory record {key!r} rev {rec.revision} hash mismatch"
                    )
                if i == 0:
                    if rec.parent_hash is not None:
                        raise ValueError(
                            f"first revision of {key!r} must have parent_hash None"
                        )
                else:
                    if rec.parent_hash != parent:
                        raise ValueError(
                            f"memory record {key!r} rev {rec.revision} parent_hash "
                            f"chain broken"
                        )
                parent = rec.record_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "daph.memory.episodic.v1",
            "records": {
                key: [r.to_dict() for r in revs]
                for key, revs in self._records.items()
            },
        }


__all__ = [
    "EpisodicMemory",
    "EpisodicMemoryRecord",
    "build_record",
]
