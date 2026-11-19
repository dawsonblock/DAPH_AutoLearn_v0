"""Section 12 — representation-selection protocol (v0.3.10.5-alpha).

Development-only study over candidate layers and pooling methods.
Selects on a frozen development metric; records every candidate result;
freezes layer/pooling/PCA/policy class/regularization/thresholds after
selection. Never accesses the final split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RepresentationCandidate:
    """One candidate representation to evaluate on dev data."""
    layer: int
    pooling: str  # last_prompt_token | mean_prompt_tokens | mean_content_tokens
    pca_components: int | None = None


@dataclass
class CandidateResult:
    """Result of evaluating one candidate on dev data."""
    candidate: RepresentationCandidate
    dev_objective: float
    mean_utility: float
    mean_regret: float
    p_symbolic: float
    abstention_rate: float
    n_tasks: int


@dataclass
class RepresentationSelection:
    """Section 12 — frozen representation selection result."""
    selected: RepresentationCandidate
    all_results: list[CandidateResult] = field(default_factory=list)
    selection_metric: str = "dev_objective"
    selection_data: str = "development"
    n_layers: int = 0
    tie_breaking: str = "lowest_layer_then_alphabetical_pooling"

    @property
    def selection_hash(self) -> str:
        payload = {
            "selected": {
                "layer": self.selected.layer,
                "pooling": self.selected.pooling,
                "pca_components": self.selected.pca_components,
            },
            "selection_metric": self.selection_metric,
            "selection_data": self.selection_data,
            "n_layers": self.n_layers,
            "n_candidates": len(self.all_results),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": {
                "layer": self.selected.layer,
                "pooling": self.selected.pooling,
                "pca_components": self.selected.pca_components,
            },
            "selection_hash": self.selection_hash,
            "selection_metric": self.selection_metric,
            "selection_data": self.selection_data,
            "n_layers": self.n_layers,
            "all_results": [
                {
                    "candidate": {
                        "layer": r.candidate.layer,
                        "pooling": r.candidate.pooling,
                        "pca_components": r.candidate.pca_components,
                    },
                    "dev_objective": r.dev_objective,
                    "mean_utility": r.mean_utility,
                    "mean_regret": r.mean_regret,
                    "p_symbolic": r.p_symbolic,
                    "abstention_rate": r.abstention_rate,
                    "n_tasks": r.n_tasks,
                }
                for r in self.all_results
            ],
        }


def layer_candidates(n_layers: int) -> list[int]:
    """Section 12 — candidate layers at 0.25, 0.50, 0.75, and last."""
    if n_layers <= 0:
        return [0]
    candidates = set()
    for frac in (0.25, 0.50, 0.75):
        candidates.add(round(frac * (n_layers - 1)))
    candidates.add(n_layers - 1)  # last layer
    return sorted(candidates)


def pooling_candidates() -> list[str]:
    """Section 12 — candidate pooling methods (minimum set)."""
    return ["last_prompt_token", "mean_prompt_tokens", "mean_content_tokens"]


def all_candidates(n_layers: int) -> list[RepresentationCandidate]:
    """Enumerate all (layer, pooling) candidates."""
    return [
        RepresentationCandidate(layer=l, pooling=p)
        for l in layer_candidates(n_layers)
        for p in pooling_candidates()
    ]


def select_representation(
    results: list[CandidateResult],
    *,
    selection_metric: str = "dev_objective",
    n_layers: int = 0,
) -> RepresentationSelection:
    """Section 12 — select the best candidate on the frozen dev metric.

    Tie-breaking: highest dev_objective wins; ties broken by lowest layer
    then alphabetical pooling (deterministic).
    """
    if not results:
        raise ValueError("no candidate results provided")
    # Sort by dev_objective descending, then layer ascending, then pooling.
    sorted_results = sorted(
        results,
        key=lambda r: (-r.dev_objective, r.candidate.layer, r.candidate.pooling),
    )
    best = sorted_results[0]
    return RepresentationSelection(
        selected=best.candidate,
        all_results=results,
        selection_metric=selection_metric,
        selection_data="development",
        n_layers=n_layers,
    )
