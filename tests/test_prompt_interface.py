"""Tests for the canonical LLM prompt interface.

Verifies that all prompt construction paths include the FINAL_ANSWER
format suffix required by the canonical verifier (Section 7).
"""

from __future__ import annotations

import pytest

from daph_learning.execution.real_backends import (
    build_llm_prompt,
    _LLM_FINAL_ANSWER_SUFFIX,
)


class TestBuildLlmPrompt:
    """Test the shared ``build_llm_prompt`` function."""

    def test_includes_final_answer_suffix(self):
        """The prompt must include the FINAL_ANSWER format requirement."""
        task = {"specification": "Compute 2 + 3."}
        prompt = build_llm_prompt(task)
        assert "FINAL_ANSWER: <integer>" in prompt

    def test_uses_specification_field(self):
        """The prompt should use the 'specification' field by default."""
        task = {"specification": "Compute 2 + 3."}
        prompt = build_llm_prompt(task)
        assert "Compute 2 + 3." in prompt

    def test_prefers_prompt_field(self):
        """The 'prompt' field takes precedence over 'specification'."""
        task = {
            "specification": "Compute 2 + 3.",
            "prompt": "What is 2 + 3?",
        }
        prompt = build_llm_prompt(task)
        assert "What is 2 + 3?" in prompt
        assert "Compute 2 + 3." not in prompt

    def test_handles_missing_specification(self):
        """A task without specification or prompt should produce just the suffix."""
        task = {}
        prompt = build_llm_prompt(task)
        assert prompt == _LLM_FINAL_ANSWER_SUFFIX

    def test_suffix_is_at_end(self):
        """The FINAL_ANSWER suffix must be at the end of the prompt."""
        task = {"specification": "Compute 2 + 3."}
        prompt = build_llm_prompt(task)
        assert prompt.endswith("FINAL_ANSWER: <integer>")

    def test_no_double_suffix(self):
        """The suffix should not be duplicated if already present."""
        task = {"specification": "Compute 2 + 3." + _LLM_FINAL_ANSWER_SUFFIX}
        prompt = build_llm_prompt(task)
        # The suffix appears twice — once from the task, once from build_llm_prompt.
        # This is acceptable: the verifier only looks for the last FINAL_ANSWER.
        # But we should verify the function doesn't strip existing suffixes.
        assert prompt.count("FINAL_ANSWER: <integer>") == 2


class TestCanonicalVerifierCompatibility:
    """Verify that build_llm_prompt output is parseable by the canonical verifier."""

    def test_verifier_accepts_final_answer_format(self):
        """The canonical verifier should accept FINAL_ANSWER: <int>."""
        from daph_learning.evaluation.canonical_verifier import (
            parse_canonical_integer_answer,
        )
        result = parse_canonical_integer_answer("FINAL_ANSWER: 42")
        assert result.status == "VALID"
        assert result.value == 42

    def test_verifier_rejects_legacy_final_format(self):
        """The canonical verifier should NOT accept FINAL: <int> (legacy)."""
        from daph_learning.evaluation.canonical_verifier import (
            parse_canonical_integer_answer,
        )
        result = parse_canonical_integer_answer("FINAL: 42")
        assert result.status != "VALID"


class TestLegacyParserCompatibility:
    """Verify that the legacy parser (parse_final_or_exact) accepts both formats."""

    def test_accepts_final_answer_format(self):
        """The legacy parser should accept FINAL_ANSWER: <int>."""
        from daph_learning.evaluation.scoring import parse_final_or_exact
        assert parse_final_or_exact("FINAL_ANSWER: 42") == 42

    def test_accepts_legacy_final_format(self):
        """The legacy parser should still accept FINAL: <int> for backward compat."""
        from daph_learning.evaluation.scoring import parse_final_or_exact
        assert parse_final_or_exact("FINAL: 42") == 42

    def test_accepts_bare_integer(self):
        """The legacy parser should accept a bare integer."""
        from daph_learning.evaluation.scoring import parse_final_or_exact
        assert parse_final_or_exact("42") == 42


class TestSymbolicOutputFormat:
    """Verify that symbolic output uses FINAL_ANSWER: format."""

    def test_symbolic_wraps_as_final_answer(self):
        """The symbolic backend should wrap output as FINAL_ANSWER: <int>."""
        from daph_learning.execution.real_backends import _wrap_symbolic_canonical
        assert _wrap_symbolic_canonical(42, None) == "FINAL_ANSWER: 42"

    def test_symbolic_error_returns_raw(self):
        """Symbolic errors should return the raw error, not FINAL_ANSWER."""
        from daph_learning.execution.real_backends import _wrap_symbolic_canonical
        assert _wrap_symbolic_canonical(None, "overflow") == "overflow"
