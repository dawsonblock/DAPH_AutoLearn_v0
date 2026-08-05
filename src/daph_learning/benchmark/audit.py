"""Section 10 — dataset leakage and benchmark audit.

Implements :class:`DatasetAudit` which checks a generated dataset for:

* exact prompt duplicates (within and across splits);
* normalized prompt duplicates (whitespace/Unicode/punctuation normalized);
* cross-split group leaks (a group_id appearing in multiple splits);
* template-family leaks (a template_family appearing in multiple splits);
* MinHash near-duplicate detection (Jaccard similarity above threshold);
* answer/subtype distributions;
* crossover metrics (per-subtype symbolic/LLM win fractions);
* decisive fraction (fraction of tasks with |ΔU| > threshold);
* minimum group count per split.

The audit ABORTS before model execution on any leakage, insufficient
crossover, or low decisive fraction. It never silently passes a leaky
dataset.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from ..data.integrity import normalize_prompt


@dataclass
class DatasetAudit:
    """Section 10 — result of auditing a dataset for leakage and quality.

    Attributes
    ----------
    valid : bool
        ``True`` only if no leakage and all quality thresholds met.
    errors : list[str]
        Blocking errors (audit fails if non-empty).
    warnings : list[str]
        Non-blocking warnings.
    n_tasks : int
        Total tasks across all splits.
    n_splits : int
        Number of splits audited.
    n_groups : dict[str, int]
        Unique group_ids per split.
    exact_duplicates : list[tuple[str, str]]
        Pairs of task_ids with identical raw prompts.
    normalized_duplicates : list[tuple[str, str]]
        Pairs of task_ids with identical normalized prompts.
    cross_split_group_leaks : list[tuple[str, str, str]]
        (group_id, split_a, split_b) for groups in multiple splits.
    template_family_leaks : list[tuple[str, str, str]]
        (template_family, split_a, split_b).
    near_duplicates : list[tuple[str, str, float]]
        (task_id_a, task_id_b, jaccard_similarity) above threshold.
    per_subtype_counts : dict[str, dict[str, int]]
        Per-subtype task counts per split.
    crossover_subtypes : list[str]
        Subtypes with within-subtype crossover.
    decisive_fraction : float
        Fraction of tasks with |ΔU| > decisive_gap_threshold.
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_tasks: int = 0
    n_splits: int = 0
    n_groups: dict[str, int] = field(default_factory=dict)
    exact_duplicates: list[tuple[str, str]] = field(default_factory=list)
    normalized_duplicates: list[tuple[str, str]] = field(default_factory=list)
    cross_split_group_leaks: list[tuple[str, str, str]] = field(default_factory=list)
    template_family_leaks: list[tuple[str, str, str]] = field(default_factory=list)
    near_duplicates: list[tuple[str, str, float]] = field(default_factory=list)
    per_subtype_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    crossover_subtypes: list[str] = field(default_factory=list)
    decisive_fraction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "n_tasks": self.n_tasks,
            "n_splits": self.n_splits,
            "n_groups": dict(self.n_groups),
            "exact_duplicates": [list(p) for p in self.exact_duplicates],
            "normalized_duplicates": [list(p) for p in self.normalized_duplicates],
            "cross_split_group_leaks": [list(p) for p in self.cross_split_group_leaks],
            "template_family_leaks": [list(p) for p in self.template_family_leaks],
            "near_duplicates": [list(p) for p in self.near_duplicates],
            "per_subtype_counts": dict(self.per_subtype_counts),
            "crossover_subtypes": list(self.crossover_subtypes),
            "decisive_fraction": self.decisive_fraction,
        }


# ------------------------------------------------------------------
# MinHash near-duplicate detection
# ------------------------------------------------------------------

def _shingles(text: str, k: int = 3) -> set[str]:
    """Character k-shingles for MinHash."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < k:
        return {text}
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _minhash_signature(shingles: set[str], n_hashes: int = 64) -> tuple[int, ...]:
    """Simple MinHash signature using hash functions h_i(x) = (a*x + b) mod p."""
    import random
    rng = random.Random(42)
    p = 2**61 - 1  # Mersenne prime
    coeffs = [(rng.randrange(1, p), rng.randrange(0, p)) for _ in range(n_hashes)]
    sig = []
    for a, b in coeffs:
        min_h = p
        for s in shingles:
            h = (a * int(hashlib.md5(s.encode()).hexdigest(), 16) + b) % p
            if h < min_h:
                min_h = h
        sig.append(min_h)
    return tuple(sig)


def _minhash_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    if len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ------------------------------------------------------------------
# Main audit function
# ------------------------------------------------------------------

def audit_dataset(
    splits: dict[str, list[dict[str, Any]]],
    *,
    min_groups_per_split: dict[str, int] | None = None,
    min_crossover_subtypes: int = 3,
    min_backend_win_fraction: float = 0.20,
    min_decisive_fraction: float = 0.35,
    near_duplicate_threshold: float = 0.99,
    crossover_distribution: dict[str, Any] | None = None,
) -> DatasetAudit:
    """Section 10 — audit a dataset for leakage and quality.

    Parameters
    ----------
    splits : dict
        ``{split_name: [task, ...]}``.
    min_groups_per_split : dict | None
        Minimum unique groups required per split.
    min_crossover_subtypes : int
        Minimum subtypes with within-subtype crossover.
    min_backend_win_fraction : float
        Minimum win fraction for the minority backend per crossover subtype.
    min_decisive_fraction : float
        Minimum fraction of tasks with |ΔU| > 0.
    near_duplicate_threshold : float
        Jaccard similarity above which two prompts are near-duplicates.
    crossover_distribution : dict | None
        Output of ``crossover_optimal_distribution`` for crossover checks.
        If None, crossover checks are skipped (warning only).
    """
    audit = DatasetAudit()
    audit.n_splits = len(splits)

    # Flatten tasks with split labels.
    all_tasks: list[tuple[str, dict[str, Any]]] = []
    for split_name, tasks in splits.items():
        for t in tasks:
            all_tasks.append((split_name, t))
    audit.n_tasks = len(all_tasks)

    # --- Group counts per split ---
    for split_name, tasks in splits.items():
        gids = {t.get("metadata", {}).get("group_id", "") for t in tasks}
        audit.n_groups[split_name] = len(gids)

    # --- Minimum groups per split ---
    if min_groups_per_split:
        for split_name, min_g in min_groups_per_split.items():
            if split_name in splits:
                actual = audit.n_groups.get(split_name, 0)
                if actual < min_g:
                    audit.errors.append(
                        f"split {split_name!r} has {actual} groups, "
                        f"minimum required is {min_g}")

    # --- Exact duplicates ---
    exact_seen: dict[str, str] = {}  # prompt → task_id
    for split_name, t in all_tasks:
        spec = str(t.get("specification", t.get("prompt", "")))
        tid = str(t.get("task_id", ""))
        if spec in exact_seen:
            audit.exact_duplicates.append((exact_seen[spec], tid))
            audit.errors.append(
                f"exact prompt duplicate: {exact_seen[spec]} == {tid}")
        else:
            exact_seen[spec] = tid

    # --- Normalized duplicates ---
    norm_seen: dict[str, str] = {}
    for split_name, t in all_tasks:
        spec = str(t.get("specification", t.get("prompt", "")))
        tid = str(t.get("task_id", ""))
        nspec = normalize_prompt(spec)
        if nspec in norm_seen:
            audit.normalized_duplicates.append((norm_seen[nspec], tid))
            audit.errors.append(
                f"normalized prompt duplicate: {norm_seen[nspec]} == {tid}")
        else:
            norm_seen[nspec] = tid

    # --- Cross-split group leaks ---
    group_to_splits: dict[str, set[str]] = {}
    for split_name, t in all_tasks:
        gid = t.get("metadata", {}).get("group_id", "")
        group_to_splits.setdefault(gid, set()).add(split_name)
    for gid, slist in group_to_splits.items():
        if len(slist) > 1:
            s_sorted = sorted(slist)
            audit.cross_split_group_leaks.append(
                (gid, s_sorted[0], s_sorted[1]))
            audit.errors.append(
                f"group {gid!r} crosses splits: {s_sorted}")

    # --- Template-family leaks ---
    fam_to_splits: dict[str, set[str]] = {}
    for split_name, t in all_tasks:
        fam = t.get("metadata", {}).get("template_family", "")
        fam_to_splits.setdefault(fam, set()).add(split_name)
    for fam, slist in fam_to_splits.items():
        if len(slist) > 1:
            s_sorted = sorted(slist)
            audit.template_family_leaks.append(
                (fam, s_sorted[0], s_sorted[1]))
            audit.errors.append(
                f"template_family {fam!r} crosses splits: {s_sorted}")

    # --- Near-duplicate detection (MinHash) ---
    if near_duplicate_threshold > 0 and audit.n_tasks > 1:
        specs = [(str(t.get("task_id", "")),
                  str(t.get("specification", t.get("prompt", ""))))
                 for _, t in all_tasks]
        sigs = [(tid, _minhash_signature(_shingles(s))) for tid, s in specs]
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                sim = _minhash_jaccard(sigs[i][1], sigs[j][1])
                if sim >= near_duplicate_threshold:
                    audit.near_duplicates.append(
                        (sigs[i][0], sigs[j][0], round(sim, 4)))
                    audit.errors.append(
                        f"near-duplicate prompts (Jaccard={sim:.2f}): "
                        f"{sigs[i][0]} ~ {sigs[j][0]}")

    # --- Per-subtype counts ---
    for split_name, tasks in splits.items():
        for t in tasks:
            subtype = t.get("metadata", {}).get("subtype", "?")
            audit.per_subtype_counts.setdefault(subtype, {})
            audit.per_subtype_counts[subtype].setdefault(split_name, 0)
            audit.per_subtype_counts[subtype][split_name] += 1

    # --- Crossover checks ---
    if crossover_distribution is not None:
        audit.crossover_subtypes = list(
            crossover_distribution.get("crossover_subtypes", []))
        if len(audit.crossover_subtypes) < min_crossover_subtypes:
            audit.errors.append(
                f"only {len(audit.crossover_subtypes)} crossover subtypes, "
                f"minimum required is {min_crossover_subtypes}")
        # Check minority backend win fraction per crossover subtype.
        per_sub = crossover_distribution.get("per_subtype_summary", {})
        for subtype, info in per_sub.items():
            if info.get("has_crossover"):
                p_sym = info.get("p_symbolic", 0.5)
                minority = min(p_sym, 1.0 - p_sym)
                if minority < min_backend_win_fraction:
                    audit.errors.append(
                        f"subtype {subtype!r} minority backend win fraction "
                        f"{minority:.2f} < minimum {min_backend_win_fraction}")
        # Decisive fraction.
        n_decisive = crossover_distribution.get("n_decisive", 0)
        n_total = crossover_distribution.get("n", 0)
        audit.decisive_fraction = (
            n_decisive / n_total if n_total > 0 else 0.0)
        if audit.decisive_fraction < min_decisive_fraction:
            audit.errors.append(
                f"decisive fraction {audit.decisive_fraction:.2f} < "
                f"minimum {min_decisive_fraction}")
    else:
        audit.warnings.append(
            "crossover_distribution not provided; crossover checks skipped")

    audit.valid = len(audit.errors) == 0
    return audit


def assert_dataset_clean(audit: DatasetAudit) -> None:
    """Raise if the audit found any blocking errors."""
    if not audit.valid:
        raise DatasetAuditError(audit.errors)


# ------------------------------------------------------------------
# Section 10: two-phase audit — empirical crossover after execution
# ------------------------------------------------------------------

@dataclass
class EmpiricalCrossoverAudit:
    """Section 10 — empirical crossover audit (post-execution).

    Computed AFTER both backends execute and the verifier runs. This
    audit checks that the EMPIRICAL crossover distribution meets the
    gate criteria, not just the structural distribution.
    """
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_tasks: int = 0
    n_decisive: int = 0
    decisive_fraction: float = 0.0
    crossover_subtypes: list[str] = field(default_factory=list)
    per_subtype_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    backend_win_fraction: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "n_tasks": self.n_tasks,
            "n_decisive": self.n_decisive,
            "decisive_fraction": self.decisive_fraction,
            "crossover_subtypes": list(self.crossover_subtypes),
            "per_subtype_summary": dict(self.per_subtype_summary),
            "backend_win_fraction": dict(self.backend_win_fraction),
        }


def audit_empirical_crossover(
    experiences: list[dict[str, Any]],
    *,
    min_crossover_subtypes: int = 3,
    min_backend_win_fraction: float = 0.20,
    min_decisive_fraction: float = 0.35,
    decisive_threshold: float = 0.02,
) -> EmpiricalCrossoverAudit:
    """Section 10 — audit EMPIRICAL crossover after backend execution.

    Parameters
    ----------
    experiences : list of experience dicts
        Each must have ``delta_utility``, ``preferred_action``, and
        task metadata with ``subtype``.
    min_crossover_subtypes : int
        Minimum subtypes with within-subtype crossover (both backends
        win at least ``min_backend_win_fraction``).
    min_backend_win_fraction : float
        Minimum win fraction for the minority backend.
    min_decisive_fraction : float
        Minimum fraction of tasks with |ΔU| > decisive_threshold.
    decisive_threshold : float
        Threshold for a task to be "decisive".
    """
    audit = EmpiricalCrossoverAudit()
    audit.n_tasks = len(experiences)

    if not experiences:
        audit.errors.append("no experiences to audit")
        audit.valid = False
        return audit

    # Compute decisive fraction.
    delta_us = [abs(float(e.get("delta_utility", 0.0))) for e in experiences]
    audit.n_decisive = sum(1 for d in delta_us if d > decisive_threshold)
    audit.decisive_fraction = audit.n_decisive / audit.n_tasks
    if audit.decisive_fraction < min_decisive_fraction:
        audit.errors.append(
            f"empirical decisive fraction {audit.decisive_fraction:.2f} < "
            f"minimum {min_decisive_fraction}")

    # Per-subtype crossover.
    subtype_results: dict[str, dict[str, int]] = {}
    for e in experiences:
        # Subtype is in the task metadata, but experiences may not carry it.
        # Try to extract from the experience or skip.
        subtype = e.get("subtype", e.get("metadata", {}).get("subtype", "?"))
        preferred = e.get("preferred_action", "abstain")
        subtype_results.setdefault(subtype, {"symbolic": 0, "llm": 0, "abstain": 0})
        if preferred in subtype_results[subtype]:
            subtype_results[subtype][preferred] += 1

    crossover_subtypes = []
    for subtype, counts in subtype_results.items():
        total = sum(counts.values())
        if total == 0:
            continue
        p_sym = counts["symbolic"] / total
        p_llm = counts["llm"] / total
        minority = min(p_sym, p_llm)
        audit.per_subtype_summary[subtype] = {
            "n": total,
            "p_symbolic": p_sym,
            "p_llm": p_llm,
            "minority_win_fraction": minority,
            "has_crossover": minority >= min_backend_win_fraction,
        }
        audit.backend_win_fraction[subtype] = minority
        if minority >= min_backend_win_fraction:
            crossover_subtypes.append(subtype)
        else:
            audit.warnings.append(
                f"subtype {subtype!r} minority win fraction {minority:.2f} < "
                f"minimum {min_backend_win_fraction}")

    audit.crossover_subtypes = sorted(crossover_subtypes)
    if len(crossover_subtypes) < min_crossover_subtypes:
        audit.errors.append(
            f"only {len(crossover_subtypes)} empirical crossover subtypes, "
            f"minimum required is {min_crossover_subtypes}")

    audit.valid = len(audit.errors) == 0
    return audit


class DatasetAuditError(Exception):
    """Raised when a dataset audit fails (Section 10)."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(
            f"dataset audit failed with {len(errors)} error(s):\n  "
            + "\n  ".join(errors))


__all__ = [
    "DatasetAudit",
    "DatasetAuditError",
    "EmpiricalCrossoverAudit",
    "assert_dataset_clean",
    "audit_dataset",
    "audit_empirical_crossover",
]
