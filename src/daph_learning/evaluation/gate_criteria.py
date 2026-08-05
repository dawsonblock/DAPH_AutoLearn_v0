"""Section 18 — frozen Gate A criteria loader and validator.

Loads ``configs/gate_a_real_002.yaml`` (or any Gate A config), validates
it against the expected schema, and exposes typed access to the gate
criteria. The criteria must NOT change after final access.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PrimaryEndpoint:
    name: str
    estimand: str
    confidence_level: float
    bootstrap_iterations: int


@dataclass(frozen=True)
class Gates:
    minimum_point_gain_vs_p0: float
    require_lcb_vs_p0_above: float
    require_lcb_vs_sham_above: float
    minimum_oracle_gap_capture: float
    maximum_worst_subtype_regression: float
    minimum_positive_group_fraction: float
    maximum_final_access_count: int


@dataclass(frozen=True)
class DatasetCriteria:
    minimum_groups: int
    minimum_crossover_subtypes: int
    minimum_backend_win_fraction: float
    minimum_decisive_fraction: float


@dataclass(frozen=True)
class EvidenceCriteria:
    require_real_model: bool
    allow_synthetic: bool
    require_model_revision: bool
    require_source_hash_match: bool
    require_dataset_hashes: bool
    require_final_access_ledger: bool


@dataclass(frozen=True)
class GateCriteria:
    """Section 18 — frozen Gate A criteria."""

    experiment_id: str
    release_version: str
    primary_endpoint: PrimaryEndpoint
    utility_protocol: str
    secondary_utility_protocol: str
    gates: Gates
    dataset: DatasetCriteria
    evidence: EvidenceCriteria
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def criteria_hash(self) -> str:
        """SHA-256 of the canonical JSON serialization (excluding raw)."""
        payload = {
            "experiment_id": self.experiment_id,
            "release_version": self.release_version,
            "primary_endpoint": {
                "name": self.primary_endpoint.name,
                "estimand": self.primary_endpoint.estimand,
                "confidence_level": self.primary_endpoint.confidence_level,
                "bootstrap_iterations": self.primary_endpoint.bootstrap_iterations,
            },
            "utility_protocol": self.utility_protocol,
            "secondary_utility_protocol": self.secondary_utility_protocol,
            "gates": {
                "minimum_point_gain_vs_p0": self.gates.minimum_point_gain_vs_p0,
                "require_lcb_vs_p0_above": self.gates.require_lcb_vs_p0_above,
                "require_lcb_vs_sham_above": self.gates.require_lcb_vs_sham_above,
                "minimum_oracle_gap_capture": self.gates.minimum_oracle_gap_capture,
                "maximum_worst_subtype_regression": self.gates.maximum_worst_subtype_regression,
                "minimum_positive_group_fraction": self.gates.minimum_positive_group_fraction,
                "maximum_final_access_count": self.gates.maximum_final_access_count,
            },
            "dataset": {
                "minimum_groups": self.dataset.minimum_groups,
                "minimum_crossover_subtypes": self.dataset.minimum_crossover_subtypes,
                "minimum_backend_win_fraction": self.dataset.minimum_backend_win_fraction,
                "minimum_decisive_fraction": self.dataset.minimum_decisive_fraction,
            },
            "evidence": {
                "require_real_model": self.evidence.require_real_model,
                "allow_synthetic": self.evidence.allow_synthetic,
                "require_model_revision": self.evidence.require_model_revision,
                "require_source_hash_match": self.evidence.require_source_hash_match,
                "require_dataset_hashes": self.evidence.require_dataset_hashes,
                "require_final_access_ledger": self.evidence.require_final_access_ledger,
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return self.raw


# Known top-level keys; unknown keys are rejected to prevent silent
# criteria drift.
_REQUIRED_KEYS = {
    "experiment_id", "primary_endpoint", "gates", "dataset", "evidence",
}
_ALLOWED_KEYS = _REQUIRED_KEYS | {
    "release_version", "utility_protocol", "secondary_utility_protocol",
    "model", "representation", "targets", "sham", "splits",
    "evidence_level", "statistics", "preconditions",
}


def load_gate_criteria(path: str | Path) -> GateCriteria:
    """Section 18 — load and validate a Gate A criteria YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load gate criteria configs; "
            "install it with `pip install pyyaml`") from exc

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"gate criteria file is not a mapping: {path}")

    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown gate criteria keys: {sorted(unknown)}")
    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(f"missing required gate criteria keys: {sorted(missing)}")

    pe = raw["primary_endpoint"]
    primary = PrimaryEndpoint(
        name=str(pe["name"]),
        estimand=str(pe["estimand"]),
        confidence_level=float(pe["confidence_level"]),
        bootstrap_iterations=int(pe["bootstrap_iterations"]),
    )
    if primary.estimand not in ("group_weighted", "task_weighted"):
        raise ValueError(
            f"primary_endpoint.estimand must be 'group_weighted' or "
            f"'task_weighted', got {primary.estimand!r}")
    if not 0.0 < primary.confidence_level < 1.0:
        raise ValueError("primary_endpoint.confidence_level must be in (0, 1)")

    g = raw["gates"]

    def _gate_value(key: str, default_comparator: str = "gte") -> float:
        """Extract gate threshold value, supporting both legacy (float)
        and v0.3.10.5 (dict with threshold/comparator) formats."""
        v = g[key]
        if isinstance(v, dict):
            return float(v.get("threshold", 0.0))
        return float(v)

    gates = Gates(
        minimum_point_gain_vs_p0=_gate_value("minimum_point_gain_vs_p0"),
        require_lcb_vs_p0_above=_gate_value("require_lcb_vs_p0_above"),
        require_lcb_vs_sham_above=_gate_value("require_lcb_vs_sham_above"),
        minimum_oracle_gap_capture=_gate_value("minimum_oracle_gap_capture"),
        maximum_worst_subtype_regression=_gate_value("maximum_worst_subtype_regression"),
        minimum_positive_group_fraction=_gate_value("minimum_positive_group_fraction"),
        maximum_final_access_count=int(_gate_value("maximum_final_access_count")),
    )

    d = raw["dataset"]
    dataset = DatasetCriteria(
        minimum_groups=int(d["minimum_groups"]),
        minimum_crossover_subtypes=int(d["minimum_crossover_subtypes"]),
        minimum_backend_win_fraction=float(d["minimum_backend_win_fraction"]),
        minimum_decisive_fraction=float(d["minimum_decisive_fraction"]),
    )

    e = raw["evidence"]
    evidence = EvidenceCriteria(
        require_real_model=bool(e["require_real_model"]),
        allow_synthetic=bool(e["allow_synthetic"]),
        require_model_revision=bool(e["require_model_revision"]),
        require_source_hash_match=bool(e["require_source_hash_match"]),
        require_dataset_hashes=bool(e["require_dataset_hashes"]),
        require_final_access_ledger=bool(e["require_final_access_ledger"]),
    )

    if evidence.allow_synthetic and evidence.require_real_model:
        raise ValueError(
            "contradictory evidence criteria: allow_synthetic=true AND "
            "require_real_model=true")

    return GateCriteria(
        experiment_id=str(raw["experiment_id"]),
        release_version=str(raw.get("release_version", "")),
        primary_endpoint=primary,
        utility_protocol=str(raw.get("utility_protocol", "gate_a_accuracy_primary")),
        secondary_utility_protocol=str(raw.get(
            "secondary_utility_protocol", "gate_a_cost_sensitive_secondary")),
        gates=gates,
        dataset=dataset,
        evidence=evidence,
        raw=raw,
    )


__all__ = [
    "DatasetCriteria",
    "EvidenceCriteria",
    "GateCriteria",
    "Gates",
    "PrimaryEndpoint",
    "load_gate_criteria",
]
