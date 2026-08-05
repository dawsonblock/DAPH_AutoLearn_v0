"""v0.3.10.1-alpha — strict task-ID alignment (Sections 5, 6).

The previous policy APIs accepted ``experiences`` and an
``activations`` ndarray and inferred alignment from array order. This
is unsafe: a shuffled feature order would silently misalign features
from experiences, and a missing capture would silently truncate the
loss/eval set without any warning.

This module replaces naked row arrays at cross-module boundaries with
task-bound :class:`FeatureRecord` objects and provides an explicit
``join_by_task_id`` helper that:

* asserts no duplicate ``task_id`` in either input
* returns aligned experiences and feature matrices
* reports missing feature records explicitly (no silent truncation)
* is order-independent: a shuffled feature order produces an identical
  result after the ID join

Audit rule (Section 6): no scientific evaluation may use plain
``zip(experiences, activations)`` without either asserting equal
lengths first or joining by ``task_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .types import CounterfactualExperience, FeatureRecord


@dataclass
class AlignmentResult:
    """Result of joining experiences and feature records by ``task_id``.

    Attributes
    ----------
    experiences : list[CounterfactualExperience]
        Experiences with a matching feature record, in the order of
        ``experiences`` input.
    features : np.ndarray
        Shape ``[N_aligned, D]`` float32 feature matrix, row-aligned
        with ``experiences``.
    aligned_task_ids : list[str]
        The task_ids that were successfully joined.
    missing_feature_task_ids : list[str]
        Experiences whose task_id had no matching feature record.
        These are *not* silently dropped — the caller is expected to
        log or fail on this.
    duplicate_experience_task_ids : list[str]
        Task_ids that appeared more than once in ``experiences`` (and
        were therefore rejected before joining).
    duplicate_feature_task_ids : list[str]
        Task_ids that appeared more than once in ``feature_records``
        (and were therefore rejected before joining).
    """

    experiences: list[CounterfactualExperience] = field(default_factory=list)
    features: np.ndarray | None = None
    aligned_task_ids: list[str] = field(default_factory=list)
    missing_feature_task_ids: list[str] = field(default_factory=list)
    duplicate_experience_task_ids: list[str] = field(default_factory=list)
    duplicate_feature_task_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if the alignment was clean (no duplicates, no missing)."""
        return (
            not self.duplicate_experience_task_ids
            and not self.duplicate_feature_task_ids
            and not self.missing_feature_task_ids
        )

    def raise_if_not_ok(self) -> "AlignmentResult":
        """Raise ``ValueError`` if the alignment was not clean."""
        if self.duplicate_experience_task_ids:
            raise ValueError(
                f"duplicate task_id in experiences: "
                f"{self.duplicate_experience_task_ids}")
        if self.duplicate_feature_task_ids:
            raise ValueError(
                f"duplicate task_id in feature_records: "
                f"{self.duplicate_feature_task_ids}")
        if self.missing_feature_task_ids:
            raise ValueError(
                f"missing feature records for task_ids: "
                f"{self.missing_feature_task_ids}")
        return self


def _find_duplicates(task_ids: Iterable[str]) -> list[str]:
    """Return the list of task_ids that appear more than once (in order
    of first duplication)."""
    seen: set[str] = set()
    dups: list[str] = []
    dup_seen: set[str] = set()
    for tid in task_ids:
        if tid in seen and tid not in dup_seen:
            dups.append(tid)
            dup_seen.add(tid)
        seen.add(tid)
    return dups


def assert_unique_task_ids(
    task_ids: Iterable[str],
    *,
    source: str = "records",
) -> None:
    """Raise ``ValueError`` if ``task_ids`` contains duplicates.

    Parameters
    ----------
    task_ids : iterable of str
    source : str
        Label used in the error message (e.g. ``"experiences"``).
    """
    dups = _find_duplicates(task_ids)
    if dups:
        raise ValueError(
            f"duplicate task_id in {source}: {dups}")


def join_by_task_id(
    experiences: Sequence[CounterfactualExperience],
    feature_records: Sequence[FeatureRecord],
    *,
    strict: bool = True,
) -> AlignmentResult:
    """Join experiences and feature records by ``task_id`` (Sections 5, 6).

    Parameters
    ----------
    experiences : sequence of CounterfactualExperience
    feature_records : sequence of FeatureRecord
    strict : bool
        If True (default), missing feature records raise
        ``ValueError``. If False, missing records are reported in
        ``AlignmentResult.missing_feature_task_ids`` and the caller is
        expected to handle them. Duplicate task_ids in *either* input
        always raise regardless of ``strict``.

    Returns
    -------
    AlignmentResult
        With ``experiences`` and ``features`` row-aligned by task_id,
        in the order of ``experiences``.

    Raises
    ------
    ValueError
        On duplicate task_ids in either input, or on missing feature
        records when ``strict=True``.
    """
    exp_dups = _find_duplicates(e.task_id for e in experiences)
    feat_dups = _find_duplicates(r.task_id for r in feature_records)
    result = AlignmentResult(
        duplicate_experience_task_ids=exp_dups,
        duplicate_feature_task_ids=feat_dups,
    )
    if exp_dups or feat_dups:
        # Always raise on duplicates — silent dedup would be unsafe.
        return result.raise_if_not_ok()

    feature_by_id = {r.task_id: r.features for r in feature_records}
    aligned_exps: list[CounterfactualExperience] = []
    aligned_feats: list[np.ndarray] = []
    aligned_ids: list[str] = []
    missing: list[str] = []
    for exp in experiences:
        feats = feature_by_id.get(exp.task_id)
        if feats is None:
            missing.append(exp.task_id)
            continue
        aligned_exps.append(exp)
        aligned_feats.append(np.asarray(feats, dtype=np.float32))
        aligned_ids.append(exp.task_id)

    result.experiences = aligned_exps
    result.aligned_task_ids = aligned_ids
    result.missing_feature_task_ids = missing
    if aligned_feats:
        # Stack along axis 0; all feature rows must have the same D.
        d0 = aligned_feats[0].shape[0]
        for i, f in enumerate(aligned_feats):
            if f.shape[0] != d0:
                raise ValueError(
                    f"feature dimension mismatch: task "
                    f"{aligned_ids[i]} has dim {f.shape[0]}, expected {d0}")
        result.features = np.stack(aligned_feats, axis=0)
    else:
        result.features = np.zeros((0, 0), dtype=np.float32)

    if strict:
        return result.raise_if_not_ok()
    return result


__all__ = [
    "AlignmentResult",
    "assert_unique_task_ids",
    "join_by_task_id",
]
