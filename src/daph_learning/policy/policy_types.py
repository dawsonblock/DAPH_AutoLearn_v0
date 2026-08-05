"""Section 2.1 — Formal policy identifiers for Gate A.

Every policy in the Gate A experiment must be identified by one of these
explicit enum values.  Legacy aliases (p0, p1, baseline) must NOT be
written into new artifacts; they are mapped to these identifiers at
production time and never persisted.

See: DAPH AutoLearn v0.3.10.5-alpha Repair Spec, Section 2.1.
"""

from __future__ import annotations

from enum import Enum


class PolicyId(str, Enum):
    """Canonical policy identifiers.

    Members
    -------
    ALWAYS_LLM
        Always select the LLM backend.  (P_LLM)
    ALWAYS_SYMBOLIC
        Always select the symbolic backend.  (P_SYM)
    BEST_FIXED
        Select the best fixed backend using development data only.
        The choice is frozen before final evaluation.  (P_FIXED)
    SHAM
        Train or derive a policy under a sham condition that destroys
        the feature→treatment-advantage relationship while preserving
        the training pipeline.  (P_SHAM)
    SURFACE_ONLY
        A learned policy using only explicitly engineered prompt
        features.  (P_SURFACE)
    HIDDEN_ONLY
        A learned policy using only hidden-state features.  (P_HIDDEN)
    HIDDEN_PLUS_SURFACE
        A learned policy using hidden-state features plus engineered
        prompt features.  (P_COMBINED)
    TFIDF
        A lexical baseline using prompt text only.  (P_TFIDF)
    HEURISTIC
        A simple deterministic benchmark-agnostic or minimally tuned
        routing heuristic.  (P_HEURISTIC)
    SUBTYPE_ONLY
        Use only task subtype metadata.  Diagnostic: reveals whether
        task-family labels alone solve routing.
    SHUFFLED_HIDDEN
        Randomly permute hidden vectors across training groups.
        Control for hidden-state signal.
    RANDOM_PROJECTION
        Replace hidden states with a deterministic random vector of
        equal dimensionality.  Control for dimensionality.
    HIDDEN_NORM_ONLY
        Use only coarse magnitude statistics of hidden states
        (L1 norm, L2 norm, mean, std, max, min).
    ORACLE
        Select the backend with the highest verified utility for each
        task after both counterfactual executions are known.
        Diagnostic upper bound, NOT a deployable policy.  (P_ORACLE)
    """

    ALWAYS_LLM = "always_llm"
    ALWAYS_SYMBOLIC = "always_symbolic"
    BEST_FIXED = "best_fixed"
    SHAM = "sham"
    SURFACE_ONLY = "surface_only"
    HIDDEN_ONLY = "hidden_only"
    HIDDEN_PLUS_SURFACE = "hidden_plus_surface"
    TFIDF = "tfidf"
    HEURISTIC = "heuristic"
    SUBTYPE_ONLY = "subtype_only"
    SHUFFLED_HIDDEN = "shuffled_hidden"
    RANDOM_PROJECTION = "random_projection"
    HIDDEN_NORM_ONLY = "hidden_norm_only"
    ORACLE = "oracle"

    @property
    def is_learned(self) -> bool:
        """True if this policy is trained from features."""
        return self in (
            PolicyId.SURFACE_ONLY,
            PolicyId.HIDDEN_ONLY,
            PolicyId.HIDDEN_PLUS_SURFACE,
            PolicyId.TFIDF,
            PolicyId.SUBTYPE_ONLY,
            PolicyId.SHUFFLED_HIDDEN,
            PolicyId.RANDOM_PROJECTION,
            PolicyId.HIDDEN_NORM_ONLY,
            PolicyId.SHAM,
        )

    @property
    def is_fixed(self) -> bool:
        """True if this policy is a fixed backend choice (no learning)."""
        return self in (PolicyId.ALWAYS_LLM, PolicyId.ALWAYS_SYMBOLIC, PolicyId.BEST_FIXED)

    @property
    def is_diagnostic(self) -> bool:
        """True if this policy is diagnostic (not deployable)."""
        return self in (PolicyId.ORACLE, PolicyId.SUBTYPE_ONLY)

    @property
    def uses_hidden_states(self) -> bool:
        """True if this policy uses hidden-state features."""
        return self in (
            PolicyId.HIDDEN_ONLY,
            PolicyId.HIDDEN_PLUS_SURFACE,
            PolicyId.SHUFFLED_HIDDEN,
            PolicyId.RANDOM_PROJECTION,
            PolicyId.HIDDEN_NORM_ONLY,
        )

    @property
    def uses_surface_features(self) -> bool:
        """True if this policy uses engineered prompt features."""
        return self in (PolicyId.SURFACE_ONLY, PolicyId.HIDDEN_PLUS_SURFACE)

    @property
    def uses_lexical_features(self) -> bool:
        """True if this policy uses raw text (TF-IDF / bag-of-words)."""
        return self in (PolicyId.TFIDF,)


# ---------------------------------------------------------------------
# Legacy alias mapping (read-only, never written to new artifacts)
# ---------------------------------------------------------------------

_LEGACY_ALIAS_MAP: dict[str, PolicyId] = {
    "p0": PolicyId.ALWAYS_LLM,        # legacy: P0 was always_llm
    "p1": PolicyId.HIDDEN_PLUS_SURFACE,  # legacy: P1 was combined
    "baseline": PolicyId.ALWAYS_LLM,
    "control": PolicyId.SHAM,
    "always_llm": PolicyId.ALWAYS_LLM,
    "always_symbolic": PolicyId.ALWAYS_SYMBOLIC,
    "oracle": PolicyId.ORACLE,
    "p1_policy": PolicyId.HIDDEN_PLUS_SURFACE,
    "subtype_majority": PolicyId.SUBTYPE_ONLY,
    "sham": PolicyId.SHAM,
}


def resolve_policy_id(name: str) -> PolicyId:
    """Resolve a legacy or canonical name to a PolicyId.

    Raises ValueError if the name is not recognised.
    """
    if name in _LEGACY_ALIAS_MAP:
        return _LEGACY_ALIAS_MAP[name]
    try:
        return PolicyId(name)
    except ValueError:
        raise ValueError(f"Unknown policy identifier: {name!r}") from None


def select_best_fixed_policy(
    dev_llm_utilities: list[float],
    dev_symbolic_utilities: list[float],
) -> PolicyId:
    """Section 2.2 — Select the best fixed backend using development data only.

    The choice must be frozen before final evaluation.

    Parameters
    ----------
    dev_llm_utilities
        Per-task LLM utilities on the development split.
    dev_symbolic_utilities
        Per-task symbolic utilities on the development split.

    Returns
    -------
    PolicyId.ALWAYS_SYMBOLIC or PolicyId.ALWAYS_LLM

    Tie-breaker: ALWAYS_SYMBOLIC (deterministic).
    """
    if not dev_llm_utilities or not dev_symbolic_utilities:
        raise ValueError("Development utility lists must be non-empty")
    if len(dev_llm_utilities) != len(dev_symbolic_utilities):
        raise ValueError("Development utility lists must have equal length")
    mean_llm = sum(dev_llm_utilities) / len(dev_llm_utilities)
    mean_symbolic = sum(dev_symbolic_utilities) / len(dev_symbolic_utilities)
    if mean_symbolic > mean_llm:
        return PolicyId.ALWAYS_SYMBOLIC
    if mean_llm > mean_symbolic:
        return PolicyId.ALWAYS_LLM
    # Deterministic tie-breaker.
    return PolicyId.ALWAYS_SYMBOLIC


__all__ = [
    "PolicyId",
    "resolve_policy_id",
    "select_best_fixed_policy",
]
