"""Section 11 — Tests for repaired policy semantics (v0.3.10.5-alpha).

Tests the new PolicyId enum, select_best_fixed_policy, and the
separation of compute-routing and hidden-state claims.
"""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from daph_learning.policy.policy_types import (
    PolicyId, resolve_policy_id, select_best_fixed_policy,
)


class TestPolicyId:
    def test_always_llm_value(self):
        assert PolicyId.ALWAYS_LLM.value == "always_llm"

    def test_always_symbolic_value(self):
        assert PolicyId.ALWAYS_SYMBOLIC.value == "always_symbolic"

    def test_best_fixed_value(self):
        assert PolicyId.BEST_FIXED.value == "best_fixed"

    def test_hidden_plus_surface_value(self):
        assert PolicyId.HIDDEN_PLUS_SURFACE.value == "hidden_plus_surface"

    def test_oracle_value(self):
        assert PolicyId.ORACLE.value == "oracle"

    def test_is_learned(self):
        assert PolicyId.SURFACE_ONLY.is_learned
        assert PolicyId.HIDDEN_ONLY.is_learned
        assert PolicyId.HIDDEN_PLUS_SURFACE.is_learned
        assert PolicyId.TFIDF.is_learned
        assert not PolicyId.ALWAYS_LLM.is_learned
        assert not PolicyId.ORACLE.is_learned

    def test_is_fixed(self):
        assert PolicyId.ALWAYS_LLM.is_fixed
        assert PolicyId.ALWAYS_SYMBOLIC.is_fixed
        assert PolicyId.BEST_FIXED.is_fixed
        assert not PolicyId.HIDDEN_ONLY.is_fixed

    def test_is_diagnostic(self):
        assert PolicyId.ORACLE.is_diagnostic
        assert not PolicyId.ALWAYS_LLM.is_diagnostic

    def test_uses_hidden_states(self):
        assert PolicyId.HIDDEN_ONLY.uses_hidden_states
        assert PolicyId.HIDDEN_PLUS_SURFACE.uses_hidden_states
        assert PolicyId.SHUFFLED_HIDDEN.uses_hidden_states
        assert not PolicyId.SURFACE_ONLY.uses_hidden_states
        assert not PolicyId.ALWAYS_LLM.uses_hidden_states

    def test_uses_surface_features(self):
        assert PolicyId.SURFACE_ONLY.uses_surface_features
        assert PolicyId.HIDDEN_PLUS_SURFACE.uses_surface_features
        assert not PolicyId.HIDDEN_ONLY.uses_surface_features


class TestResolvePolicyId:
    def test_resolve_canonical(self):
        assert resolve_policy_id("always_llm") == PolicyId.ALWAYS_LLM
        assert resolve_policy_id("always_symbolic") == PolicyId.ALWAYS_SYMBOLIC

    def test_resolve_legacy_p0(self):
        assert resolve_policy_id("p0") == PolicyId.ALWAYS_LLM

    def test_resolve_legacy_p1(self):
        assert resolve_policy_id("p1") == PolicyId.HIDDEN_PLUS_SURFACE

    def test_resolve_legacy_baseline(self):
        assert resolve_policy_id("baseline") == PolicyId.ALWAYS_LLM

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown policy"):
            resolve_policy_id("nonexistent_policy")


class TestSelectBestFixedPolicy:
    def test_symbolic_wins(self):
        """When symbolic has higher mean utility, select always_symbolic."""
        dev_llm = [0.3, 0.4, 0.2, 0.5]
        dev_sym = [0.8, 0.9, 0.7, 0.8]
        result = select_best_fixed_policy(dev_llm, dev_sym)
        assert result == PolicyId.ALWAYS_SYMBOLIC

    def test_llm_wins(self):
        """When LLM has higher mean utility, select always_llm."""
        dev_llm = [0.8, 0.9, 0.7, 0.8]
        dev_sym = [0.3, 0.4, 0.2, 0.5]
        result = select_best_fixed_policy(dev_llm, dev_sym)
        assert result == PolicyId.ALWAYS_LLM

    def test_tie_breaker_is_symbolic(self):
        """Tie-breaker is deterministic: always_symbolic."""
        dev_llm = [0.5, 0.5, 0.5]
        dev_sym = [0.5, 0.5, 0.5]
        result = select_best_fixed_policy(dev_llm, dev_sym)
        assert result == PolicyId.ALWAYS_SYMBOLIC

    def test_empty_lists_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            select_best_fixed_policy([], [])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="equal length"):
            select_best_fixed_policy([0.5, 0.6], [0.5])

    def test_selection_is_deterministic(self):
        """Same inputs always produce same output."""
        dev_llm = [0.3, 0.4, 0.2, 0.5]
        dev_sym = [0.8, 0.9, 0.7, 0.8]
        r1 = select_best_fixed_policy(dev_llm, dev_sym)
        r2 = select_best_fixed_policy(dev_llm, dev_sym)
        assert r1 == r2

    def test_does_not_use_final_data(self):
        """The selection function only takes dev data as input."""
        # This is enforced by the API: only dev_llm and dev_sym are passed.
        # No final data parameter exists.
        import inspect
        sig = inspect.signature(select_best_fixed_policy)
        params = list(sig.parameters.keys())
        assert "dev_llm_utilities" in params
        assert "dev_symbolic_utilities" in params
        assert "final" not in params
        assert "test" not in params


class TestAsymmetricFixture:
    """Section 11.1 — Create a deliberately asymmetric fixture where
    confusing LLM and symbolic baselines causes an obvious failure.
    """

    def test_confusing_baselines_fails(self):
        """If we swap LLM and symbolic utilities, best_fixed changes."""
        # Symbolic is clearly better.
        dev_llm = [0.1, 0.2, 0.1, 0.2]
        dev_sym = [0.9, 0.8, 0.9, 0.8]
        correct = select_best_fixed_policy(dev_llm, dev_sym)
        assert correct == PolicyId.ALWAYS_SYMBOLIC

        # If we confuse them, we get the wrong answer.
        confused = select_best_fixed_policy(dev_sym, dev_llm)
        assert confused == PolicyId.ALWAYS_LLM
        assert confused != correct
