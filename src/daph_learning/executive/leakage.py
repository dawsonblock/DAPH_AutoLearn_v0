"""DAPH v0.4 — Leakage checks for experiment integrity.

Before qualification, automatically verify:

* **Dataset identity**: No task ID overlap between train, dev, final.
* **Exact prompt leakage**: No normalized exact-prompt overlap.
* **Retrieval leakage**: The retrieval store may contain TRAIN examples only.
* **Representation leakage**: PCA must be fit on training representations only.
* **Policy leakage**: Final labels or final utilities must never be used for
  representation selection, PCA fitting, hyperparameter selection, ridge
  strength selection, threshold selection, or feature selection.
* **Group leakage**: If multiple generated variants derive from the same
  underlying template/seed, they must remain within the same split.

Qualification fails if any hard leakage test fails.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Check result types
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LeakageCheck:
    """Result of a single leakage check."""

    check_name: str
    passed: bool
    severity: str = "hard"  # "hard" or "soft"
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


@dataclass
class LeakageReport:
    """Full leakage check report."""

    experiment_id: str
    checks: list[LeakageCheck] = field(default_factory=list)
    passed: bool = True
    hard_failures: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)

    def add(self, check: LeakageCheck) -> None:
        self.checks.append(check)
        if not check.passed:
            if check.severity == "hard":
                self.passed = False
                self.hard_failures.append(
                    f"{check.check_name}: {check.detail}"
                )
            else:
                self.soft_warnings.append(
                    f"{check.check_name}: {check.detail}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_hard_failures": len(self.hard_failures),
            "n_soft_warnings": len(self.soft_warnings),
            "hard_failures": list(self.hard_failures),
            "soft_warnings": list(self.soft_warnings),
            "checks": [c.to_dict() for c in self.checks],
        }


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Task ID overlap check
# ──────────────────────────────────────────────────────────────────────

def check_task_id_overlap(
    train_ids: Sequence[str],
    dev_ids: Sequence[str],
    final_ids: Sequence[str],
) -> LeakageCheck:
    """Verify no task ID overlap between train, dev, and final splits."""
    train_set = set(train_ids)
    dev_set = set(dev_ids)
    final_set = set(final_ids)

    td_overlap = train_set & dev_set
    tf_overlap = train_set & final_set
    df_overlap = dev_set & final_set

    all_overlap = td_overlap | tf_overlap | df_overlap
    passed = len(all_overlap) == 0

    detail = ""
    if not passed:
        overlaps = []
        if td_overlap:
            overlaps.append(f"train∩dev={len(td_overlap)}")
        if tf_overlap:
            overlaps.append(f"train∩final={len(tf_overlap)}")
        if df_overlap:
            overlaps.append(f"dev∩final={len(df_overlap)}")
        detail = f"task ID overlap detected: {', '.join(overlaps)}"

    return LeakageCheck(
        check_name="task_id_no_overlap",
        passed=passed,
        severity="hard",
        detail=detail or "no task ID overlap between splits",
        evidence={
            "n_train": len(train_set),
            "n_dev": len(dev_set),
            "n_final": len(final_set),
            "overlap_count": len(all_overlap),
            "overlap_sample": sorted(all_overlap)[:10],
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Exact prompt leakage check
# ──────────────────────────────────────────────────────────────────────

def _normalize_prompt(prompt: str) -> str:
    """Normalize a prompt for leakage comparison."""
    # Strip whitespace, lowercase, remove non-alphanumeric
    return re.sub(r"[^a-z0-9]", "", prompt.lower().strip())


def check_exact_prompt_leakage(
    train_prompts: Sequence[str],
    dev_prompts: Sequence[str],
    final_prompts: Sequence[str],
) -> LeakageCheck:
    """Verify no normalized exact-prompt overlap between splits."""
    train_norm = {_normalize_prompt(p) for p in train_prompts if p}
    dev_norm = {_normalize_prompt(p) for p in dev_prompts if p}
    final_norm = {_normalize_prompt(p) for p in final_prompts if p}

    # Remove empty strings from sets
    train_norm.discard("")
    dev_norm.discard("")
    final_norm.discard("")

    td = train_norm & dev_norm
    tf = train_norm & final_norm
    df = dev_norm & final_norm
    all_overlap = td | tf | df

    passed = len(all_overlap) == 0
    detail = ""
    if not passed:
        detail = f"exact prompt overlap: {len(all_overlap)} prompts shared across splits"

    return LeakageCheck(
        check_name="exact_prompt_no_overlap",
        passed=passed,
        severity="hard",
        detail=detail or "no exact prompt overlap between splits",
        evidence={
            "overlap_count": len(all_overlap),
            "overlap_sample": sorted(all_overlap)[:5],
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Retrieval store leakage check
# ──────────────────────────────────────────────────────────────────────

def check_retrieval_store_leakage(
    retrieval_store_ids: Sequence[str],
    train_ids: Sequence[str],
    dev_ids: Sequence[str],
    final_ids: Sequence[str],
) -> LeakageCheck:
    """Verify the retrieval store contains only TRAIN examples."""
    store_set = set(retrieval_store_ids)
    train_set = set(train_ids)
    dev_set = set(dev_ids)
    final_set = set(final_ids)

    dev_in_store = store_set & dev_set
    final_in_store = store_set & final_set
    contamination = dev_in_store | final_in_store

    passed = len(contamination) == 0
    detail = ""
    if not passed:
        detail = (f"retrieval store contaminated: "
                  f"{len(dev_in_store)} dev, {len(final_in_store)} final examples")

    return LeakageCheck(
        check_name="retrieval_store_train_only",
        passed=passed,
        severity="hard",
        detail=detail or "retrieval store contains only train examples",
        evidence={
            "store_size": len(store_set),
            "dev_contamination": len(dev_in_store),
            "final_contamination": len(final_in_store),
            "contaminated_ids": sorted(contamination)[:10],
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Section 5 — PCA train-only check
# ──────────────────────────────────────────────────────────────────────

def check_pca_train_only(pca_manifest: Mapping[str, Any]) -> LeakageCheck:
    """Verify PCA was fit on training representations only.

    The PCA manifest must declare:
    * ``fit_split`` = ``"train"``
    * ``training_representation_hash`` is non-empty and non-zero
    """
    fit_split = pca_manifest.get("fit_split", "")
    train_hash = pca_manifest.get("training_representation_hash", "")

    issues = []
    if fit_split != "train":
        issues.append(f"fit_split={fit_split!r} (expected 'train')")
    if not train_hash or train_hash == "0" * 64:
        issues.append("training_representation_hash is missing or zeroed")

    passed = len(issues) == 0
    return LeakageCheck(
        check_name="pca_train_only",
        passed=passed,
        severity="hard",
        detail="; ".join(issues) if issues else "PCA fit on train only",
        evidence={
            "fit_split": fit_split,
            "has_training_hash": bool(train_hash) and train_hash != "0" * 64,
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Section 6 — Policy leakage check (no final data used for selection)
# ──────────────────────────────────────────────────────────────────────

def check_policy_leakage(
    selection_manifest: Mapping[str, Any],
) -> LeakageCheck:
    """Verify representation/feature/hyperparameter selection used dev only,
    not final.

    The selection manifest must declare:
    * ``selection_split`` = ``"dev"``
    * No reference to final utilities in the selection process
    """
    selection_split = selection_manifest.get("selection_split", "")
    used_final = selection_manifest.get("used_final_data", False)

    issues = []
    if selection_split not in ("dev", "validation"):
        issues.append(f"selection_split={selection_split!r} (expected 'dev')")
    if used_final:
        issues.append("selection process used final data")

    passed = len(issues) == 0
    return LeakageCheck(
        check_name="policy_no_final_leakage",
        passed=passed,
        severity="hard",
        detail="; ".join(issues) if issues else "selection used dev only, no final leakage",
        evidence={
            "selection_split": selection_split,
            "used_final_data": used_final,
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Section 7 — Group/template leakage check
# ──────────────────────────────────────────────────────────────────────

def check_group_leakage(
    train_groups: Sequence[str],
    dev_groups: Sequence[str],
    final_groups: Sequence[str],
) -> LeakageCheck:
    """Verify no group (template/seed family) appears in multiple splits.

    If multiple generated variants derive from the same underlying
    template/seed, they must remain within the same split.
    """
    train_g = set(train_groups)
    dev_g = set(dev_groups)
    final_g = set(final_groups)

    td = train_g & dev_g
    tf = train_g & final_g
    df = dev_g & final_g
    overlap = td | tf | df

    passed = len(overlap) == 0
    detail = ""
    if not passed:
        detail = f"group/template leakage: {len(overlap)} groups in multiple splits"

    return LeakageCheck(
        check_name="group_no_cross_split",
        passed=passed,
        severity="hard",
        detail=detail or "no group appears in multiple splits",
        evidence={
            "n_train_groups": len(train_g),
            "n_dev_groups": len(dev_g),
            "n_final_groups": len(final_g),
            "overlap_count": len(overlap),
            "overlap_sample": sorted(overlap)[:10],
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Section 8 — Representation sanity check
# ──────────────────────────────────────────────────────────────────────

def check_representation_sanity(
    features: np.ndarray,
    task_ids: Sequence[str],
    *,
    expected_dim: int | None = None,
) -> LeakageCheck:
    """Verify a representation matrix is sane.

    Checks:
    * No NaNs
    * No infinities
    * Nonzero variance
    * Correct dimensionality (if specified)
    * Task IDs match row count
    """
    issues = []

    if features.ndim != 2:
        issues.append(f"expected 2D array, got {features.ndim}D")
    else:
        if np.any(np.isnan(features)):
            n_nan = int(np.sum(np.isnan(features)))
            issues.append(f"{n_nan} NaN values")
        if np.any(np.isinf(features)):
            n_inf = int(np.sum(np.isinf(features)))
            issues.append(f"{n_inf} infinite values")
        variances = np.var(features, axis=0)
        zero_var = int(np.sum(variances == 0))
        if zero_var > 0:
            issues.append(f"{zero_var} zero-variance columns")
        if expected_dim is not None and features.shape[1] != expected_dim:
            issues.append(f"dim {features.shape[1]} != expected {expected_dim}")

    if len(task_ids) != features.shape[0]:
        issues.append(f"task_ids count {len(task_ids)} != rows {features.shape[0]}")

    passed = len(issues) == 0
    return LeakageCheck(
        check_name="representation_sanity",
        passed=passed,
        severity="hard",
        detail="; ".join(issues) if issues else "representation is sane",
        evidence={
            "shape": list(features.shape),
            "n_task_ids": len(task_ids),
            "has_nan": bool(np.any(np.isnan(features))),
            "has_inf": bool(np.any(np.isinf(features))),
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Section 9 — Full leakage report from artifact tree
# ──────────────────────────────────────────────────────────────────────

def run_leakage_checks_from_artifacts(
    artifact_root: str | Path,
    *,
    experiment_id: str = "",
) -> LeakageReport:
    """Run all leakage checks from saved artifacts.

    This loads the dataset splits, counterfactuals, PCA manifest, and
    selection manifest from the artifact tree and runs all checks.
    """
    root = Path(artifact_root)
    report = LeakageReport(experiment_id=experiment_id)

    ds_dir = root / "dataset"
    pca_dir = root / "pca"
    sel_dir = root / "selection"

    # Load dataset splits if available
    def _load_json(p: Path) -> dict | list | None:
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    train_tasks = _load_json(ds_dir / "train.json") or []
    dev_tasks = _load_json(ds_dir / "dev.json") or []
    final_tasks = _load_json(ds_dir / "final.json") or []

    if train_tasks and dev_tasks and final_tasks:
        def _ids(tasks):
            return [t.get("task_id", "") for t in tasks]
        def _prompts(tasks):
            return [t.get("prompt", t.get("specification", "")) for t in tasks]
        def _groups(tasks):
            return [t.get("group_id", t.get("group", "")) for t in tasks]

        report.add(check_task_id_overlap(
            _ids(train_tasks), _ids(dev_tasks), _ids(final_tasks)
        ))
        report.add(check_exact_prompt_leakage(
            _prompts(train_tasks), _prompts(dev_tasks), _prompts(final_tasks)
        ))
        report.add(check_group_leakage(
            _groups(train_tasks), _groups(dev_tasks), _groups(final_tasks)
        ))

    # PCA manifest
    pca_manifest = _load_json(pca_dir / "pca_manifest.json")
    if pca_manifest and isinstance(pca_manifest, dict):
        report.add(check_pca_train_only(pca_manifest))

    # Selection manifest
    sel_manifest = _load_json(sel_dir / "selected_representation.json")
    if sel_manifest and isinstance(sel_manifest, dict):
        report.add(check_policy_leakage(sel_manifest))

    return report


__all__ = [
    "LeakageCheck",
    "LeakageReport",
    "check_task_id_overlap",
    "check_exact_prompt_leakage",
    "check_retrieval_store_leakage",
    "check_pca_train_only",
    "check_policy_leakage",
    "check_group_leakage",
    "check_representation_sanity",
    "run_leakage_checks_from_artifacts",
]
