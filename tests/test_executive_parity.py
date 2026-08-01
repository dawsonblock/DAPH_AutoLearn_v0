"""DAPH v0.4 — Parity test: generic executive interface vs legacy binary.

This test verifies that the new generic ``daph_learning.executive``
module produces identical results to the legacy v0.3.x binary
qualification when applied to the same binary (symbolic vs llm) data.

This is the critical regression test: if parity holds, we can migrate
all consumers to the generic interface and remove the hard-coded
binary path without changing any experimental results.
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
    ActionDecision,
    ExecutiveTaskRecord,
    evaluate_qualification,
    binary_action_space,
    action_from_backend,
    action_decision_from_symbolic_probability,
    symbolic_probability_from_action_decision,
    utility_model_from_legacy_config,
)

from daph_learning.autolearn.counterfactual import (
    UtilityConfig as LegacyUtilityConfig,
    compute_backend_utility,
)
from daph_learning.autolearn.outcome import (
    BackendExecutionRecord,
    OutcomeSemantics,
    BACKENDS,
)
from daph_learning.evaluation.verification import (
    VerificationResult,
    VerificationStatus,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers: build equivalent legacy and generic data
# ──────────────────────────────────────────────────────────────────────

def _make_legacy_record(backend, correct, latency_ms=100.0, cost=0.1):
    """Build a legacy BackendExecutionRecord."""
    status = VerificationStatus.VERIFIED_CORRECT if correct else VerificationStatus.VERIFIED_INCORRECT
    return BackendExecutionRecord(
        backend=backend,
        selected=(backend == "symbolic"),
        executed=True,
        output="42" if correct else "99",
        verification=VerificationResult(status=status, verifier="numeric"),
        latency_ms=latency_ms,
        compute_cost=cost,
    )


def _make_generic_execution(action_id, correct, latency_ms=100.0, cost=0.1):
    """Build a generic ActionExecution."""
    return ActionExecution(
        action_id=action_id,
        selected=(action_id == "action.symbolic_arithmetic"),
        executed=True,
        output="42" if correct else "99",
        verified_correct=correct,
        verifier_name="numeric",
        latency_ms=latency_ms,
        compute_cost=cost,
    )


# ──────────────────────────────────────────────────────────────────────
# Parity: Utility Computation
# ──────────────────────────────────────────────────────────────────────

class TestUtilityParity:
    """Verify U(state, action) matches U_backend for the same data."""

    def test_symbolic_correct(self):
        legacy_rec = _make_legacy_record("symbolic", correct=True, latency_ms=10, cost=0.05)
        generic_exec = _make_generic_execution(
            "action.symbolic_arithmetic", correct=True, latency_ms=10, cost=0.05)

        legacy_cfg = LegacyUtilityConfig(
            quality_weight=1.0, lambda_time=0.01, lambda_compute=0.1,
            lambda_risk=1.0, time_reference_ms=1000.0, compute_reference=1.0,
        )
        generic_cfg = utility_model_from_legacy_config(legacy_cfg)

        legacy_bd = compute_backend_utility(legacy_rec, legacy_cfg)
        generic_bd = generic_cfg.compute(generic_exec)

        assert legacy_bd.utility == pytest.approx(generic_bd.utility, abs=1e-10)
        assert legacy_bd.quality == pytest.approx(generic_bd.quality)
        assert legacy_bd.time_term == pytest.approx(generic_bd.time_term)
        assert legacy_bd.compute_term == pytest.approx(generic_bd.compute_term)
        assert legacy_bd.risk_term == pytest.approx(generic_bd.risk_term)

    def test_llm_incorrect(self):
        legacy_rec = _make_legacy_record("llm", correct=False, latency_ms=500, cost=0.2)
        generic_exec = _make_generic_execution(
            "action.llm_direct", correct=False, latency_ms=500, cost=0.2)

        legacy_cfg = LegacyUtilityConfig(
            quality_weight=1.0, lambda_time=0.01, lambda_compute=0.1,
            lambda_risk=1.0,
        )
        generic_cfg = utility_model_from_legacy_config(legacy_cfg)

        legacy_bd = compute_backend_utility(legacy_rec, legacy_cfg)
        generic_bd = generic_cfg.compute(generic_exec)

        assert legacy_bd.utility == pytest.approx(generic_bd.utility, abs=1e-10)

    def test_config_hash_matches(self):
        legacy_cfg = LegacyUtilityConfig(
            quality_weight=1.0, lambda_time=0.01, lambda_compute=0.1,
            lambda_risk=1.0, abstention_band=0.05,
        ).freeze()
        generic_cfg = utility_model_from_legacy_config(legacy_cfg)
        assert legacy_cfg.config_sha256 == generic_cfg.config_sha256

    def test_delta_u_matches_regret(self):
        """ΔU = U_symbolic - U_llm should match the generic regret computation."""
        sym_rec = _make_legacy_record("symbolic", correct=True, latency_ms=10, cost=0.05)
        llm_rec = _make_legacy_record("llm", correct=False, latency_ms=500, cost=0.2)

        legacy_cfg = LegacyUtilityConfig(lambda_time=0.01, lambda_compute=0.1, lambda_risk=1.0)
        u_sym = compute_backend_utility(sym_rec, legacy_cfg).utility
        u_llm = compute_backend_utility(llm_rec, legacy_cfg).utility
        delta_u = u_sym - u_llm

        # Generic equivalent
        state = ExecutiveState(task_id="t1", prompt="p")
        e_sym = _make_generic_execution("action.symbolic_arithmetic", True, 10, 0.05)
        e_llm = _make_generic_execution("action.llm_direct", False, 500, 0.2)
        cf = CounterfactualSet(
            state=state,
            executions={"action.symbolic_arithmetic": e_sym, "action.llm_direct": e_llm},
        )
        generic_cfg = utility_model_from_legacy_config(legacy_cfg)
        breakdowns = generic_cfg.compute_all(cf)
        generic_delta = (breakdowns["action.symbolic_arithmetic"].utility
                         - breakdowns["action.llm_direct"].utility)

        assert delta_u == pytest.approx(generic_delta, abs=1e-10)


# ──────────────────────────────────────────────────────────────────────
# Parity: Policy Decision
# ──────────────────────────────────────────────────────────────────────

class TestPolicyDecisionParity:
    """Verify symbolic_probability ↔ ActionDecision round-trip."""

    def test_round_trip_high_symbolic(self):
        dec = action_decision_from_symbolic_probability("t1", 0.9)
        p = symbolic_probability_from_action_decision(dec)
        assert p == pytest.approx(0.9)
        assert dec.selected_action == "action.symbolic_arithmetic"

    def test_round_trip_low_symbolic(self):
        dec = action_decision_from_symbolic_probability("t1", 0.2)
        p = symbolic_probability_from_action_decision(dec)
        assert p == pytest.approx(0.2)
        assert dec.selected_action == "action.llm_direct"

    def test_round_trip_half(self):
        dec = action_decision_from_symbolic_probability("t1", 0.5)
        p = symbolic_probability_from_action_decision(dec)
        assert p == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────
# Parity: Action Mapping
# ──────────────────────────────────────────────────────────────────────

class TestActionMappingParity:
    """Verify backend ↔ action_id mapping is consistent."""

    def test_all_backends_mapped(self):
        for backend in BACKENDS:
            action_id = action_from_backend(backend)
            assert action_id.startswith("action.")

    def test_abstain_mapped(self):
        assert action_from_backend("abstain") == "action.abstain"

    def test_round_trip(self):
        for backend in ("symbolic", "llm", "abstain"):
            action_id = action_from_backend(backend)
            from daph_learning.executive import backend_from_action
            assert backend_from_action(action_id) == backend


# ──────────────────────────────────────────────────────────────────────
# Parity: Oracle
# ──────────────────────────────────────────────────────────────────────

class TestOracleParity:
    """Verify generic oracle matches binary argmax(U_symbolic, U_llm)."""

    def test_oracle_matches_binary_argmax(self):
        rng = np.random.RandomState(42)
        for _ in range(20):
            sym_correct = rng.random() > 0.5
            llm_correct = rng.random() > 0.5
            sym_lat = rng.uniform(10, 100)
            llm_lat = rng.uniform(100, 1000)

            # Legacy
            sym_rec = _make_legacy_record("symbolic", sym_correct, sym_lat, 0.05)
            llm_rec = _make_legacy_record("llm", llm_correct, llm_lat, 0.2)
            cfg = LegacyUtilityConfig(lambda_time=0.01, lambda_compute=0.1, lambda_risk=1.0)
            u_sym = compute_backend_utility(sym_rec, cfg).utility
            u_llm = compute_backend_utility(llm_rec, cfg).utility
            legacy_oracle = "symbolic" if u_sym >= u_llm else "llm"

            # Generic
            state = ExecutiveState(task_id="t", prompt="p")
            e_sym = _make_generic_execution("action.symbolic_arithmetic", sym_correct, sym_lat, 0.05)
            e_llm = _make_generic_execution("action.llm_direct", llm_correct, llm_lat, 0.2)
            cf = CounterfactualSet(
                state=state,
                executions={"action.symbolic_arithmetic": e_sym, "action.llm_direct": e_llm},
            )
            g_cfg = utility_model_from_legacy_config(cfg)
            generic_oracle_id, _ = g_cfg.best_action(cf)
            generic_oracle = "symbolic" if generic_oracle_id == "action.symbolic_arithmetic" else "llm"

            assert legacy_oracle == generic_oracle, (
                f"oracle mismatch: legacy={legacy_oracle}, generic={generic_oracle} "
                f"(u_sym={u_sym:.4f}, u_llm={u_llm:.4f})")
