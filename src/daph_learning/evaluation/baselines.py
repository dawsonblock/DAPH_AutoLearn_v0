"""Section 13 — surface and structured-feature baselines (v0.3.10.4-alpha).

Implements the 10 required baselines so the report can answer: does
hidden-state routing beat subtype-only / TF-IDF / structured features
on unseen groups?

Baselines:
  1. always_llm          — route everything to LLM
  2. always_symbolic     — route everything to symbolic
  3. hand_router         — rule-based (exists in routing.hand_router)
  4. subtype_only_logistic — logistic on one-hot subtype
  5. surface_tfidf_logistic — logistic on TF-IDF of prompt
  6. structured_feature_logistic — logistic on StructuredTaskFeatures
  7. hidden_state_centroid — utility-weighted centroid (exists)
  8. hidden_state_logistic — logistic on hidden states
  9. sham_hidden_state_logistic — same features, shuffled labels
 10. oracle              — always picks the higher-utility backend
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..policy.types import CounterfactualExperience, Route


@dataclass
class StructuredTaskFeatures:
    """Section 13 — structured features extracted from a task."""
    n_operands: int = 0
    max_operand_digits: int = 0
    has_modular: bool = False
    has_comparison: bool = False
    is_natural_language: bool = False
    n_steps: int = 1
    operand_magnitude_bucket: int = 0  # 0=small, 1=medium, 2=large

    def to_vector(self) -> list[float]:
        return [
            float(self.n_operands),
            float(self.max_operand_digits),
            float(self.has_modular),
            float(self.has_comparison),
            float(self.is_natural_language),
            float(self.n_steps),
            float(self.operand_magnitude_bucket),
        ]


def extract_structured_features(task: dict[str, Any]) -> StructuredTaskFeatures:
    """Extract structured features from a task dict."""
    inputs = task.get("inputs", {})
    spec = str(task.get("specification", task.get("prompt", "")))
    subtype = str(task.get("metadata", {}).get("subtype", ""))
    a = abs(int(inputs.get("a", 0)))
    b = abs(int(inputs.get("b", 0)))
    max_val = max(a, b)
    digits = len(str(max_val)) if max_val > 0 else 1
    op = str(inputs.get("op", ""))
    has_mod = op in ("%", "mod", "//")
    has_cmp = subtype == "E"
    is_nl = subtype in ("B", "C", "F")
    bucket = 0 if max_val < 100 else (1 if max_val < 10000 else 2)
    return StructuredTaskFeatures(
        n_operands=2 if "b" in inputs else 1,
        max_operand_digits=digits,
        has_modular=has_mod,
        has_comparison=has_cmp,
        is_natural_language=is_nl,
        n_steps=1 if subtype != "F" else 3,
        operand_magnitude_bucket=bucket,
    )


# ------------------------------------------------------------------
# Baseline routers
# ------------------------------------------------------------------

def always_llm(_task: dict[str, Any], _experience: CounterfactualExperience
               ) -> Route:
    return Route.LLM


def always_symbolic(_task: dict[str, Any], _experience: CounterfactualExperience
                    ) -> Route:
    return Route.SYMBOLIC


def oracle(_task: dict[str, Any], experience: CounterfactualExperience,
           *, utility_fn: Callable | None = None) -> Route:
    """Always pick the higher-utility backend (upper bound)."""
    if utility_fn is None:
        # Default: compare quality.
        u_sym = experience.symbolic.quality
        u_llm = experience.llm.quality
    else:
        u_sym = utility_fn(experience.symbolic)
        u_llm = utility_fn(experience.llm)
    if u_sym >= u_llm:
        return Route.SYMBOLIC
    return Route.LLM


def hand_router(task: dict[str, Any], _experience: CounterfactualExperience
                ) -> Route:
    """Rule-based: structured arithmetic → symbolic; NL/ambiguous → LLM."""
    from ..routing.hand_router import hand_router_route
    return hand_router_route(task)


def subtype_only_logistic(task: dict[str, Any], _experience: CounterfactualExperience,
                          *, weights: dict[str, dict[str, float]] | None = None
                          ) -> Route:
    """Logistic on one-hot subtype. Weights map subtype → {symbolic, llm}."""
    subtype = str(task.get("metadata", {}).get("subtype", ""))
    w = (weights or {}).get(subtype, {})
    score_sym = w.get("symbolic", 0.0)
    score_llm = w.get("llm", 0.0)
    return Route.SYMBOLIC if score_sym >= score_llm else Route.LLM


def surface_tfidf_logistic(task: dict[str, Any], _experience: CounterfactualExperience,
                           *, weights: dict[str, float] | None = None,
                           vocab: dict[str, int] | None = None
                           ) -> Route:
    """Logistic on TF-IDF of prompt. Simplified: bag-of-words dot product."""
    spec = str(task.get("specification", task.get("prompt", "")))
    words = spec.lower().split()
    w = weights or {}
    v = vocab or {}
    score = 0.0
    for word in words:
        if word in v:
            score += w.get(word, 0.0)
    return Route.SYMBOLIC if score >= 0 else Route.LLM


def structured_feature_logistic(
    task: dict[str, Any], _experience: CounterfactualExperience,
    *, weights: list[float] | None = None, bias: float = 0.0,
) -> Route:
    """Logistic on StructuredTaskFeatures."""
    feats = extract_structured_features(task).to_vector()
    w = weights or [0.0] * len(feats)
    score = sum(wi * fi for wi, fi in zip(w, feats)) + bias
    return Route.SYMBOLIC if score >= 0 else Route.LLM


def hidden_state_centroid(_task: dict[str, Any], _experience: CounterfactualExperience,
                          *, centroid_sym=None, centroid_llm=None, h=None
                          ) -> Route:
    """Utility-weighted centroid baseline."""
    if h is None or centroid_sym is None or centroid_llm is None:
        return Route.LLM  # fallback
    import numpy as np
    d_sym = float(np.linalg.norm(h - centroid_sym))
    d_llm = float(np.linalg.norm(h - centroid_llm))
    return Route.SYMBOLIC if d_sym <= d_llm else Route.LLM


def hidden_state_logistic(_task: dict[str, Any], _experience: CounterfactualExperience,
                          *, weights=None, bias: float = 0.0, h=None
                          ) -> Route:
    """Logistic on hidden states."""
    if h is None or weights is None:
        return Route.LLM
    import numpy as np
    score = float(np.dot(weights, h)) + bias
    return Route.SYMBOLIC if score >= 0 else Route.LLM


def sham_hidden_state_logistic(task, experience, *, weights=None, bias=0.0, h=None,
                               sham_labels=None) -> Route:
    """Same features/model as hidden_state_logistic but with shuffled labels.
    The sham destroys the feature→winner mapping."""
    if sham_labels is None:
        # If no sham labels provided, route randomly (signal destroyed).
        import random
        return Route.SYMBOLIC if random.random() < 0.5 else Route.LLM
    return hidden_state_logistic(task, experience, weights=weights, bias=bias, h=h)


BASELINE_NAMES = (
    "always_llm",
    "always_symbolic",
    "hand_router",
    "subtype_only_logistic",
    "surface_tfidf_logistic",
    "structured_feature_logistic",
    "hidden_state_centroid",
    "hidden_state_logistic",
    "sham_hidden_state_logistic",
    "oracle",
)


__all__ = [
    "BASELINE_NAMES",
    "StructuredTaskFeatures",
    "always_llm",
    "always_symbolic",
    "extract_structured_features",
    "hand_router",
    "hidden_state_centroid",
    "hidden_state_logistic",
    "oracle",
    "sham_hidden_state_logistic",
    "structured_feature_logistic",
    "subtype_only_logistic",
    "surface_tfidf_logistic",
]
