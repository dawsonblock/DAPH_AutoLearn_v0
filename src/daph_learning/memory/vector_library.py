"""v0.3.9 (V039-010) — RISER-style vector library and SAE feature selection.

Item 6 of the recommended repair sequence: move toward a **RISER-style
vector library** or **SAE-selected features** for steering, with memory as a
separate retrievable, revisable system (see :mod:`daph_learning.memory.episodic`).

This module provides:

* :class:`VectorLibrary` — a retrievable library of steering vectors
  indexed by behavior / family / layer, ranked by recorded metrics. This is
  the RISER-style retrieval surface: given a target behavior, retrieve the
  best-performing stored direction rather than re-extracting from scratch.
* :class:`FeatureSelector` — an extension-point protocol for selecting
  steering-relevant features from a representation (e.g. sparse-autoencoder
  features).
* :class:`TopKFeatureSelector` — a concrete, dependency-light SAE-style
  selector that keeps the top-k features by activation magnitude and
  projects activations onto the selected feature subspace. This is the
  minimal working implementation of the SAE-selected-features direction;
  richer SAE trainers can be plugged in by implementing the protocol.

The library and selector are deliberately self-contained (stdlib + numpy)
so they import anywhere ``daph_learning`` runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _vector_hash(values: Sequence[float]) -> str:
    arr = np.asarray(values, dtype=np.float32)
    return _sha256_hex(arr.tobytes())


# --- Vector library ---

@dataclass(frozen=True)
class VectorLibraryEntry:
    """One steering vector in the library."""

    vector_id: str
    behavior: str            # e.g. "invoke_symbolic_tool"
    family: str              # "tool_policy" | "reasoning_policy" | ...
    layer: int
    alpha: float
    vector_hash: str         # sha256 of the values
    hidden_size: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    parent_vector_id: str | None = None
    promoted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VectorLibrary:
    """A retrievable, ranked library of steering vectors (RISER-style).

    Vectors are stored by id with their metadata and metrics. Retrieval
    ranks by a metric (default ``"held_out_utility"``) descending, filtered
    by behavior / family / layer. The library does not store the dense
    values themselves (those live in ``.npz`` artifacts referenced by id);
    it stores the hash so retrieval is tamper-evident.
    """

    def __init__(self) -> None:
        self._entries: dict[str, VectorLibraryEntry] = {}

    def add(self, entry: VectorLibraryEntry) -> None:
        self._entries[entry.vector_id] = entry

    def get(self, vector_id: str) -> VectorLibraryEntry | None:
        return self._entries.get(vector_id)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, vector_id: object) -> bool:
        return vector_id in self._entries

    def retrieve(
        self,
        *,
        behavior: str | None = None,
        family: str | None = None,
        layer: int | None = None,
        rank_by: str = "held_out_utility",
        limit: int = 5,
    ) -> list[VectorLibraryEntry]:
        """Retrieve the best vectors matching the filters, ranked by metric."""
        entries = list(self._entries.values())
        if behavior is not None:
            entries = [e for e in entries if e.behavior == behavior]
        if family is not None:
            entries = [e for e in entries if e.family == family]
        if layer is not None:
            entries = [e for e in entries if e.layer == layer]
        entries.sort(key=lambda e: (-float(e.metrics.get(rank_by, 0.0)), e.vector_id))
        return entries[:limit]

    def best_for_behavior(self, behavior: str, *, rank_by: str = "held_out_utility") -> VectorLibraryEntry | None:
        hits = self.retrieve(behavior=behavior, rank_by=rank_by, limit=1)
        return hits[0] if hits else None

    def all_entries(self) -> list[VectorLibraryEntry]:
        return list(self._entries.values())


def make_entry(
    *,
    vector_id: str,
    behavior: str,
    family: str,
    layer: int,
    alpha: float,
    values: Sequence[float],
    metrics: Mapping[str, float] | None = None,
    parent_vector_id: str | None = None,
    promoted: bool = False,
) -> VectorLibraryEntry:
    """Build a :class:`VectorLibraryEntry` from raw values (hashes them)."""
    arr = np.asarray(values, dtype=np.float32)
    return VectorLibraryEntry(
        vector_id=vector_id, behavior=behavior, family=family, layer=layer,
        alpha=alpha, vector_hash=_vector_hash(values), hidden_size=int(arr.size),
        metrics=dict(metrics or {}), parent_vector_id=parent_vector_id,
        promoted=promoted,
    )


# --- SAE-selected features extension point ---

@runtime_checkable
class FeatureSelector(Protocol):
    """Extension-point protocol for selecting steering-relevant features.

    A concrete selector chooses a subset of feature dimensions (e.g. SAE
    latents) that are most relevant to a target behavior, and projects
    activations onto that subspace. Implementations may wrap a trained
    sparse autoencoder, a probe-based selector, or any other method.
    """

    def fit(self, activations: np.ndarray, labels: Sequence[int]) -> None:
        """Fit the selector on a matrix of activations [N, D] with labels."""
        ...

    def select(self, k: int) -> np.ndarray:
        """Return the indices of the top-k selected features (sorted)."""
        ...

    def project(self, activations: np.ndarray) -> np.ndarray:
        """Project activations onto the selected feature subspace."""
        ...


@dataclass
class TopKFeatureSelector:
    """SAE-style top-k feature selector.

    Selects the k features whose per-class mean activation difference is
    largest in absolute value (a contrastive SAE-style score). This is a
    dependency-light stand-in for a trained sparse autoencoder: it captures
    the "SAE-selected features" direction of the roadmap without requiring
    an external SAE trainer. A real SAE can be plugged in by implementing
    :class:`FeatureSelector`.
    """

    k: int = 8
    _scores: np.ndarray | None = field(default=None, init=False, repr=False)
    _selected: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, activations: np.ndarray, labels: Sequence[int]) -> None:
        acts = np.asarray(activations, dtype=np.float32)
        if acts.ndim != 2:
            raise ValueError("activations must be [N, D]")
        labels_arr = np.asarray(labels)
        if labels_arr.shape[0] != acts.shape[0]:
            raise ValueError("labels must have one entry per activation row")
        if set(np.unique(labels_arr).tolist()) - {0, 1}:
            raise ValueError("labels must be binary {0, 1}")
        pos = acts[labels_arr == 1]
        neg = acts[labels_arr == 0]
        if pos.shape[0] == 0 or neg.shape[0] == 0:
            # fall back to variance across all samples if a class is empty
            self._scores = acts.var(axis=0)
        else:
            self._scores = np.abs(pos.mean(axis=0) - neg.mean(axis=0))
        self._selected = None  # invalidate prior selection

    def select(self, k: int | None = None) -> np.ndarray:
        if self._scores is None:
            raise ValueError("selector not fitted; call fit() first")
        kk = k if k is not None else self.k
        if kk <= 0:
            raise ValueError("k must be > 0")
        kk = min(kk, int(self._scores.shape[0]))
        # argpartition for top-k, then sort for a stable, ascending index list
        idx = np.argpartition(-self._scores, kk - 1)[:kk]
        self._selected = np.sort(idx)
        return self._selected

    def project(self, activations: np.ndarray) -> np.ndarray:
        if self._selected is None:
            raise ValueError("no selection; call select() first")
        acts = np.asarray(activations, dtype=np.float32)
        return acts[..., self._selected]

    @property
    def scores(self) -> np.ndarray | None:
        return self._scores

    @property
    def selected(self) -> np.ndarray | None:
        return self._selected


__all__ = [
    "FeatureSelector",
    "TopKFeatureSelector",
    "VectorLibrary",
    "VectorLibraryEntry",
    "make_entry",
]
