"""v0.3.10.3.2-alpha — capture representation ablation (Section 28).

The capture location (which residual-stream layer we read activations
from) is a hyperparameter chosen on DEV, never on FINAL. This module
provides the ablation harness: evaluate routing utility across
candidate capture layers and select the best on DEV.

The ablation records:
  * per-layer mean utility, mean regret, P(symbolic), abstention rate
  * the selected layer (chosen on DEV)
  * the source tree hash and config hash for binding

The selected layer is then FROZEN and used on FINAL without further
tuning (Section 14).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CaptureLayerResult:
    """Section 28: one capture layer's evaluation."""
    layer: int
    mean_utility: float
    mean_regret: float
    p_symbolic: float
    abstention_rate: float
    n_tasks: int


@dataclass
class CaptureAblationReport:
    """Section 28: full capture-layer ablation."""
    layers: list[int]
    per_layer: list[CaptureLayerResult]
    best_layer: int
    best_layer_utility: float
    config_hash: str = ""
    source_tree_hash: str = ""
    split: str = "dev"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": self.layers,
            "per_layer": [
                {"layer": r.layer, "mean_utility": r.mean_utility,
                 "mean_regret": r.mean_regret, "p_symbolic": r.p_symbolic,
                 "abstention_rate": r.abstention_rate, "n_tasks": r.n_tasks}
                for r in self.per_layer],
            "best_layer": self.best_layer,
            "best_layer_utility": self.best_layer_utility,
            "config_hash": self.config_hash,
            "source_tree_hash": self.source_tree_hash,
            "split": self.split,
        }


def run_capture_ablation(
    tasks: list[dict[str, Any]],
    *,
    layers: Sequence[int],
    policy_proba_fn: Callable[[Mapping[str, Any], int], float],
    utility_fn: Callable[[Mapping[str, Any], str], float],
    config,
    split: str = "dev",
) -> CaptureAblationReport:
    """Run the capture-layer ablation (Section 28).

    Parameters
    ----------
    tasks : list of task mappings (DEV split only)
    layers : candidate capture layers
    policy_proba_fn : callable(task, layer) -> P(symbolic | captured at layer)
    utility_fn : callable(task, route) -> utility
    config : ExperimentConfig
    split : str (must be "dev" or "train"; never "final")
    """
    if split == "final":
        raise ValueError(
            "Section 28: capture ablation on FINAL is forbidden; "
            "the capture layer must be selected on DEV and frozen")
    per_layer: list[CaptureLayerResult] = []
    for layer in layers:
        utilities = []
        regrets = []
        p_syms = []
        n_abstain = 0
        for t in tasks:
            p = float(policy_proba_fn(t, layer))
            p_syms.append(p)
            if p >= config.confidence_threshold:
                route = "symbolic"
            elif p <= 1.0 - config.confidence_threshold:
                route = "llm"
            else:
                route = "abstain"
                n_abstain += 1
            u = utility_fn(t, route) if route != "abstain" else 0.0
            utilities.append(u)
            # Regret = max(u_sym, u_llm) - u_chosen (oracle uses both routes)
            u_sym = utility_fn(t, "symbolic")
            u_llm = utility_fn(t, "llm")
            u_opt = max(u_sym, u_llm)
            regrets.append(u_opt - u)
        n = len(tasks)
        per_layer.append(CaptureLayerResult(
            layer=layer,
            mean_utility=float(np.mean(utilities)) if utilities else 0.0,
            mean_regret=float(np.mean(regrets)) if regrets else 0.0,
            p_symbolic=float(np.mean(p_syms)) if p_syms else 0.0,
            abstention_rate=n_abstain / n if n else 0.0,
            n_tasks=n))
    best = max(per_layer, key=lambda r: r.mean_utility)
    return CaptureAblationReport(
        layers=list(layers), per_layer=per_layer,
        best_layer=best.layer, best_layer_utility=best.mean_utility,
        split=split)


__all__ = [
    "CaptureAblationReport",
    "CaptureLayerResult",
    "run_capture_ablation",
]
