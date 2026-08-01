"""Tests for DAPH v0.4 generic executive types.

Verifies that the new generic ActionSpace / ExecutiveState / ActionExecution /
CounterfactualSet / UtilityModel / ActionDecision types work correctly,
and that the legacy binary routing experiment can be represented through
the generic interface.
"""

from __future__ import annotations

import pytest
import numpy as np

from daph_learning.executive import (
    ActionDescriptor,
    ActionSpace,
    ExecutiveState,
    ActionExecution,
    CounterfactualSet,
    UtilityModel,
    UtilityBreakdown,
    ActionDecision,
    Regret,
    binary_action_space,
    action_from_backend,
    backend_from_action,
    action_decision_from_symbolic_probability,
    symbolic_probability_from_action_decision,
    utility_model_from_legacy_config,
)


# ──────────────────────────────────────────────────────────────────────
# ActionDescriptor / ActionSpace
# ──────────────────────────────────────────────────────────────────────

class TestActionDescriptor:
    def test_basic_construction(self):
        a = ActionDescriptor(
            action_id="reasoning.direct",
            display_name="Direct Reasoning",
            cost_estimate=0.1,
            tags=("reasoning", "direct"),
        )
        assert a.action_id == "reasoning.direct"
        assert a.display_name == "Direct Reasoning"
        assert a.namespace == "reasoning"

    def test_default_display_name(self):
        a = ActionDescriptor(action_id="retrieval.vector")
        assert a.display_name == "retrieval.vector"

    def test_empty_action_id_rejected(self):
        with pytest.raises(ValueError, match="action_id must be non-empty"):
            ActionDescriptor(action_id="")

    def test_cost_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="cost_estimate"):
            ActionDescriptor(action_id="x", cost_estimate=1.5)

    def test_namespace_extraction(self):
        assert ActionDescriptor(action_id="a.b.c").namespace == "a"
        assert ActionDescriptor(action_id="simple").namespace == "simple"


class TestActionSpace:
    def test_binary_action_space(self):
        space = binary_action_space()
        assert space.n_actions == 2
        assert "action.symbolic_arithmetic" in space.action_ids
        assert "action.llm_direct" in space.action_ids
        assert space.abstain_id == "action.abstain"

    def test_too_few_actions_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            ActionSpace(actions=(ActionDescriptor(action_id="a"),))

    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            ActionSpace(actions=(
                ActionDescriptor(action_id="a"),
                ActionDescriptor(action_id="a"),
            ))

    def test_abstain_in_actions_rejected(self):
        with pytest.raises(ValueError, match="abstain_id"):
            ActionSpace(
                actions=(ActionDescriptor(action_id="a"), ActionDescriptor(action_id="b")),
                abstain_id="a",
            )

    def test_multi_action_space(self):
        space = ActionSpace(actions=(
            ActionDescriptor(action_id="reasoning.direct"),
            ActionDescriptor(action_id="retrieval.vector"),
            ActionDescriptor(action_id="reasoning.decompose"),
            ActionDescriptor(action_id="tool.python"),
        ))
        assert space.n_actions == 4
        assert len(space.executable_ids) == 4

    def test_is_valid(self):
        space = binary_action_space()
        assert space.is_valid("action.symbolic_arithmetic")
        assert space.is_valid("action.llm_direct")
        assert space.is_valid("action.abstain")
        assert not space.is_valid("nonexistent")


# ──────────────────────────────────────────────────────────────────────
# ExecutiveState
# ──────────────────────────────────────────────────────────────────────

class TestExecutiveState:
    def test_basic_construction(self):
        state = ExecutiveState(
            task_id="task_001",
            prompt="What is 2+2?",
            task_metadata={"subtype": "A", "group_id": "g1"},
        )
        assert state.task_id == "task_001"
        assert state.subtype == "A"
        assert state.group_id == "g1"

    def test_empty_task_id_rejected(self):
        with pytest.raises(ValueError):
            ExecutiveState(task_id="", prompt="x")

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValueError):
            ExecutiveState(task_id="x", prompt="")


# ──────────────────────────────────────────────────────────────────────
# ActionExecution
# ──────────────────────────────────────────────────────────────────────

class TestActionExecution:
    def test_correct_execution(self):
        e = ActionExecution(
            action_id="a", selected=True, executed=True,
            verified_correct=True,
        )
        assert e.is_correct is True
        assert e.quality == 1.0
        assert e.risk == 0.0

    def test_incorrect_execution(self):
        e = ActionExecution(
            action_id="a", selected=True, executed=True,
            verified_correct=False,
        )
        assert e.is_correct is False
        assert e.quality == 0.0
        # Risk is about execution failure, not incorrect answers
        assert e.risk == 0.0

    def test_unexecuted(self):
        e = ActionExecution(
            action_id="a", selected=False, executed=False,
        )
        assert e.is_correct is False
        assert e.quality == 0.0
        assert e.risk == 0.0

    def test_protocol_invariant_unexecuted_cannot_be_correct(self):
        with pytest.raises(ValueError, match="protocol invariant"):
            ActionExecution(
                action_id="a", selected=True, executed=False,
                verified_correct=True,
            )

    def test_failure_type_sets_risk(self):
        e = ActionExecution(
            action_id="a", selected=True, executed=True,
            failure_type="timeout",
        )
        assert e.risk == 1.0


# ──────────────────────────────────────────────────────────────────────
# CounterfactualSet
# ──────────────────────────────────────────────────────────────────────

class TestCounterfactualSet:
    def _make_cf(self):
        state = ExecutiveState(task_id="t1", prompt="p")
        e1 = ActionExecution(action_id="a", selected=True, executed=True, verified_correct=True)
        e2 = ActionExecution(action_id="b", selected=False, executed=True, verified_correct=False)
        return CounterfactualSet(
            state=state,
            executions={"a": e1, "b": e2},
            selected_action="a",
        )

    def test_basic(self):
        cf = self._make_cf()
        assert cf.task_id == "t1"
        assert cf.all_executed is True
        assert cf.is_correct("a") is True
        assert cf.is_correct("b") is False

    def test_execution_lookup(self):
        cf = self._make_cf()
        assert cf.execution("a") is not None
        assert cf.execution("nonexistent") is None


# ──────────────────────────────────────────────────────────────────────
# UtilityModel
# ──────────────────────────────────────────────────────────────────────

class TestUtilityModel:
    def test_quality_only(self):
        um = UtilityModel(quality_weight=1.0, lambda_time=0, lambda_compute=0, lambda_risk=0)
        e = ActionExecution(action_id="a", selected=True, executed=True, verified_correct=True)
        bd = um.compute(e)
        assert bd.utility == 1.0
        assert bd.quality == 1.0

    def test_cost_penalty(self):
        um = UtilityModel(quality_weight=1.0, lambda_time=0.01, lambda_compute=0.0, lambda_risk=0.0,
                          time_reference_ms=1000.0)
        e = ActionExecution(
            action_id="a", selected=True, executed=True,
            verified_correct=True, latency_ms=500.0,
        )
        bd = um.compute(e)
        assert bd.utility == pytest.approx(1.0 - 0.01 * 0.5)

    def test_risk_penalty(self):
        """Risk penalty applies for execution failures, not incorrect answers."""
        um = UtilityModel(quality_weight=1.0, lambda_risk=1.0)
        e = ActionExecution(
            action_id="a", selected=True, executed=True,
            verified_correct=False, failure_type="timeout",
        )
        bd = um.compute(e)
        # Q=0 (incorrect), R=1 (failure_type set) → U = 0 - 1 = -1
        assert bd.utility == pytest.approx(0.0 - 1.0)

    def test_incorrect_without_failure_has_no_risk(self):
        """An incorrect answer without execution failure has Q=0 but R=0."""
        um = UtilityModel(quality_weight=1.0, lambda_risk=1.0)
        e = ActionExecution(
            action_id="a", selected=True, executed=True,
            verified_correct=False,
        )
        bd = um.compute(e)
        assert bd.utility == pytest.approx(0.0)

    def test_freeze(self):
        um = UtilityModel()
        frozen = um.freeze()
        assert frozen.config_sha256 is not None
        # Idempotent
        assert frozen.freeze().config_sha256 == frozen.config_sha256

    def test_best_action(self):
        state = ExecutiveState(task_id="t1", prompt="p")
        e1 = ActionExecution(action_id="a", selected=True, executed=True, verified_correct=True, latency_ms=10)
        e2 = ActionExecution(action_id="b", selected=False, executed=True, verified_correct=True, latency_ms=500)
        cf = CounterfactualSet(state=state, executions={"a": e1, "b": e2})
        um = UtilityModel(lambda_time=0.01)
        best_id, best_u = um.best_action(cf)
        assert best_id == "a"  # faster → higher utility

    def test_regret(self):
        state = ExecutiveState(task_id="t1", prompt="p")
        e1 = ActionExecution(action_id="a", selected=True, executed=True, verified_correct=True, latency_ms=10)
        e2 = ActionExecution(action_id="b", selected=False, executed=True, verified_correct=True, latency_ms=500)
        cf = CounterfactualSet(state=state, executions={"a": e1, "b": e2})
        um = UtilityModel(lambda_time=0.01)
        regret = um.regret(cf, "b")
        assert regret.regret > 0
        assert regret.best_action == "a"
        assert regret.chosen_action == "b"


# ──────────────────────────────────────────────────────────────────────
# ActionDecision
# ──────────────────────────────────────────────────────────────────────

class TestActionDecision:
    def test_basic(self):
        dec = ActionDecision(
            state_id="t1",
            scores={"a": 0.7, "b": 0.3},
            probabilities={"a": 0.7, "b": 0.3},
            selected_action="a",
            confidence=0.7,
        )
        assert dec.selected_action == "a"
        assert dec.probability("a") == 0.7

    def test_probabilities_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to ~1.0"):
            ActionDecision(
                state_id="t1",
                scores={"a": 0.7, "b": 0.7},
                probabilities={"a": 0.7, "b": 0.7},
                selected_action="a",
                confidence=0.7,
            )

    def test_selected_must_be_in_probabilities(self):
        with pytest.raises(ValueError, match="not in probabilities"):
            ActionDecision(
                state_id="t1",
                scores={"a": 1.0},
                probabilities={"a": 1.0},
                selected_action="b",
                confidence=1.0,
            )


# ──────────────────────────────────────────────────────────────────────
# Legacy Adapters
# ──────────────────────────────────────────────────────────────────────

class TestAdapters:
    def test_action_from_backend(self):
        assert action_from_backend("symbolic") == "action.symbolic_arithmetic"
        assert action_from_backend("llm") == "action.llm_direct"
        assert action_from_backend("abstain") == "action.abstain"

    def test_backend_from_action(self):
        assert backend_from_action("action.symbolic_arithmetic") == "symbolic"
        assert backend_from_action("action.llm_direct") == "llm"

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError):
            action_from_backend("unknown")

    def test_symbolic_probability_round_trip(self):
        dec = action_decision_from_symbolic_probability("t1", 0.8, calibrated=True)
        assert dec.selected_action == "action.symbolic_arithmetic"
        assert dec.confidence == pytest.approx(0.8)
        p = symbolic_probability_from_action_decision(dec)
        assert p == pytest.approx(0.8)

    def test_symbolic_probability_below_half_selects_llm(self):
        dec = action_decision_from_symbolic_probability("t1", 0.3)
        assert dec.selected_action == "action.llm_direct"
        assert dec.confidence == pytest.approx(0.7)

    def test_utility_model_from_legacy(self):
        from daph_learning.autolearn.counterfactual import UtilityConfig
        legacy = UtilityConfig(
            quality_weight=1.0, lambda_time=0.01, lambda_compute=0.1,
            lambda_risk=1.0, abstention_band=0.05,
        )
        um = utility_model_from_legacy_config(legacy)
        assert um.quality_weight == 1.0
        assert um.lambda_time == 0.01
        assert um.abstention_band == 0.05
