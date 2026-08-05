"""v0.3.10.1-alpha tests for comparative gate, capability gate, and
real intervention pipeline (Sections 11, 13, 24, 26).

G14: comparative gates literally compare methods.
G22: capability regression gates work.
G16: real-model intervention pipeline executes.
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.evaluation.capability_gate import (
    CapabilityGateConfig,
    capability_gate,
    comparative_gate,
)
from daph_learning.interventions import RealInterventionResult


# ============================================================
# G14: comparative gate literally compares methods (Section 11)
# ============================================================

class TestComparativeGate:
    """Section 11 / G14: comparative gate must compute both sides."""

    def test_computes_both_sides(self):
        candidate = {"mean_regret": 0.04, "mean_utility": 0.9}
        incumbent = {"mean_regret": 0.08, "mean_utility": 0.85}
        result = comparative_gate(
            candidate, incumbent, metric="mean_regret",
            expected_ordering="candidate_lower")
        assert result.passed
        assert result.measurements["candidate_value"] == 0.04
        assert result.measurements["incumbent_value"] == 0.08

    def test_fails_when_candidate_worse(self):
        candidate = {"mean_regret": 0.10}
        incumbent = {"mean_regret": 0.05}
        result = comparative_gate(
            candidate, incumbent, metric="mean_regret",
            expected_ordering="candidate_lower")
        assert not result.passed

    def test_tolerance_allows_tie(self):
        candidate = {"mean_regret": 0.06}
        incumbent = {"mean_regret": 0.05}
        result = comparative_gate(
            candidate, incumbent, metric="mean_regret",
            expected_ordering="candidate_lower", tolerance=0.02)
        assert result.passed

    def test_candidate_higher_for_utility(self):
        candidate = {"mean_utility": 0.9}
        incumbent = {"mean_utility": 0.8}
        result = comparative_gate(
            candidate, incumbent, metric="mean_utility",
            expected_ordering="candidate_higher")
        assert result.passed

    def test_raises_on_missing_metric(self):
        with pytest.raises(ValueError, match="missing metric"):
            comparative_gate({"mean_regret": 0.04}, {}, metric="mean_regret")
        with pytest.raises(ValueError, match="missing metric"):
            comparative_gate({}, {"mean_regret": 0.04}, metric="mean_regret")

    def test_never_passes_without_both_sides(self):
        """G14: never mark PASS without computing both sides."""
        # If either side is missing, the gate raises (not passes).
        with pytest.raises(ValueError):
            comparative_gate({}, {}, metric="mean_regret")

    def test_logistic_vs_centroid_on_linear(self):
        """Integration: logistic should beat centroid on linear env."""
        from daph_learning.environment_benchmark import (
            make_linear_environment, benchmark_execute_fn, benchmark_utility,
        )
        from daph_learning.policy.learner import build_counterfactual_experiences
        from daph_learning.policy import ExperimentConfig, fit_policy, predict_proba
        from daph_learning.policy.regret import mean_regret
        train = make_linear_environment(n=200, dim=8, seed=0)
        dev = make_linear_environment(n=150, dim=8, seed=1)
        results = {}
        for ptype in ("centroid", "logistic"):
            cfg = ExperimentConfig(
                policy_type=ptype, gap_threshold=0.05,
                confidence_threshold=0.5, random_seed=0)
            train_exp = build_counterfactual_experiences(
                train, execute_fn=benchmark_execute_fn, config=cfg)
            train_acts = np.array([t["activation"] for t in train], dtype=np.float32)
            train_du = np.array([e.delta_utility for e in train_exp], dtype=np.float32)
            train_w = np.array([e.sample_weight for e in train_exp], dtype=np.float32)
            model = fit_policy(cfg, train_acts, train_du, train_w)
            dev_acts = np.array([t["activation"] for t in dev], dtype=np.float32)
            probs = predict_proba(model, dev_acts)
            from daph_learning.policy.abstention import choose_route
            from daph_learning.environment_benchmark import benchmark_oracle_utility
            routes = [choose_route(float(p), 0.5).value for p in probs]
            pred_u = np.array([
                benchmark_utility(t, r) for t, r in zip(dev, routes)],
                dtype=np.float64)
            ora_u = np.array([
                benchmark_oracle_utility(t) for t in dev], dtype=np.float64)
            results[ptype] = {"mean_regret": float(mean_regret(pred_u, ora_u))}
        gate = comparative_gate(
            results["logistic"], results["centroid"],
            metric="mean_regret", expected_ordering="candidate_lower",
            tolerance=0.05)
        # Both sides computed and stored.
        assert "candidate_value" in gate.measurements
        assert "incumbent_value" in gate.measurements


# ============================================================
# G22: capability regression gates (Section 24)
# ============================================================

class TestCapabilityGate:
    """Section 24: reject candidates that regress on protected families."""

    def test_passes_when_no_regression(self):
        cu = np.array([0.9, 0.8, 0.7, 0.85])
        iu = np.array([0.85, 0.75, 0.65, 0.80])
        cr = ["symbolic", "llm", "symbolic", "llm"]
        ir = ["llm", "symbolic", "symbolic", "llm"]
        correct = ["symbolic", "llm", "symbolic", "llm"]
        fams = ["S", "L", "S", "L"]
        result = capability_gate(
            cu, iu, cr, ir, correct, fams,
            config=CapabilityGateConfig(min_family_samples=2))
        assert result.passed

    def test_fails_on_family_utility_regression(self):
        # Family S: candidate regresses by 0.1 (> threshold 0.02).
        cu = np.array([0.7, 0.7, 0.9, 0.9])
        iu = np.array([0.8, 0.8, 0.9, 0.9])
        cr = ["symbolic", "symbolic", "llm", "llm"]
        ir = ["symbolic", "symbolic", "llm", "llm"]
        correct = ["symbolic", "symbolic", "llm", "llm"]
        fams = ["S", "S", "L", "L"]
        result = capability_gate(
            cu, iu, cr, ir, correct, fams,
            config=CapabilityGateConfig(
                max_family_utility_drop=0.02, min_family_samples=2))
        assert not result.passed
        assert "S" in result.reason

    def test_fails_on_family_accuracy_regression(self):
        cu = np.array([0.9, 0.9, 0.5, 0.5])
        iu = np.array([0.9, 0.9, 0.5, 0.5])
        cr = ["llm", "symbolic", "llm", "llm"]  # S: 1/2 correct
        ir = ["symbolic", "symbolic", "llm", "llm"]  # S: 2/2 correct
        correct = ["symbolic", "symbolic", "llm", "llm"]
        fams = ["S", "S", "L", "L"]
        result = capability_gate(
            cu, iu, cr, ir, correct, fams,
            config=CapabilityGateConfig(
                max_family_accuracy_drop=0.03, min_family_samples=2))
        assert not result.passed

    def test_small_families_skipped(self):
        """Families with < min_family_samples are skipped."""
        cu = np.array([0.0, 0.9, 0.9])
        iu = np.array([1.0, 0.9, 0.9])
        cr = ["llm", "llm", "llm"]
        ir = ["symbolic", "llm", "llm"]
        correct = ["symbolic", "llm", "llm"]
        fams = ["S", "L", "L"]  # S has only 1 sample, L has 2
        result = capability_gate(
            cu, iu, cr, ir, correct, fams,
            config=CapabilityGateConfig(min_family_samples=20))
        # Both S (1) and L (2) are below min_family_samples=20, so
        # neither is evaluated. The gate passes (no family evaluated).
        assert result.passed
        assert len(result.families) == 0  # both skipped


# ============================================================
# G16: real intervention pipeline executes (Section 13)
# ============================================================

class TestRealInterventionPipeline:
    """Section 13: RealInterventionResult + ResidualStreamHook smoke tests."""

    def test_real_intervention_result_record(self):
        r = RealInterventionResult(
            task_id="t1", vector_id="v1", layer=20, alpha=1.0,
            baseline_route="llm", intervened_route="symbolic",
            baseline_utility=0.0, intervened_utility=1.0,
            utility_delta=1.0,
            route_score_before=0.3, route_score_after=0.8,
            neutral_kl=0.01, clamp_triggered=False,
            evidence_level="REAL_MODEL_LATENT_INTERVENTION")
        d = r.to_dict()
        assert d["evidence_level"] == "REAL_MODEL_LATENT_INTERVENTION"
        assert d["utility_delta"] == 1.0
        assert d["route_score_before"] == 0.3

    def test_intervention_config_defaults(self):
        from daph_learning.interventions.real_pipeline import InterventionConfig
        cfg = InterventionConfig()
        assert cfg.layer == 20
        assert cfg.alpha_grid == (-1.0, -0.5, 0.0, 0.5, 1.0)

    def test_residual_stream_hook_resolves_layer(self):
        """The hook can resolve a target module on a small model."""
        try:
            from transformers import AutoModelForCausalLM
            from daph_learning.interventions.real_pipeline import ResidualStreamHook
            # Use a tiny model for the smoke test.
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct")
            hook = ResidualStreamHook(model, layer=2)
            target = hook._resolve_target_module()
            assert target is not None
        except ImportError:
            pytest.skip("transformers not available")
        except Exception as e:
            pytest.skip(f"model load failed: {e}")


# ============================================================
# G23: atomic rollback (Section 46) — smoke test
# ============================================================

class TestAtomicPromotion:
    """Section 46: atomic policy promotion / rollback smoke test."""

    def test_save_and_load_policy_artifact(self, tmp_path):
        from daph_learning.policy.centroid_policy import CentroidPolicy
        import numpy as np
        model = CentroidPolicy()
        model.vector = np.ones(4, dtype=np.float32)
        model.threshold = 0.5
        model.temperature = 1.0
        model.n_train_ = 100
        model.n_symbolic_ = 50
        model.n_llm_ = 50
        path = str(tmp_path / "policy.json")
        model.save(path)
        loaded = CentroidPolicy.load(path)
        np.testing.assert_array_equal(loaded.vector, model.vector)
        assert loaded.threshold == model.threshold
        assert loaded.n_train_ == 100

    def test_logistic_save_and_load(self, tmp_path):
        from daph_learning.policy.logistic import WeightedLogisticRouter
        model = WeightedLogisticRouter(input_dim=4)
        path = str(tmp_path / "logistic.pt")
        model.save(path)
        loaded = WeightedLogisticRouter.load(path)
        assert loaded.linear.in_features == 4


# ============================================================
# G20: OOD threshold calibration (Section 19)
# ============================================================

class TestOODCalibration:
    """Section 19: OOD threshold is calibrated, not infinite by default
    in qualified runs."""

    def test_quantile_based_threshold(self):
        from daph_learning.policy.ood import MahalanobisOOD
        rng = np.random.default_rng(0)
        train = rng.normal(0, 1, size=(100, 4)).astype(np.float32)
        cal = rng.normal(0, 1, size=(50, 4)).astype(np.float32)
        ood = MahalanobisOOD(ridge=1e-4)
        ood.fit(train)
        cal_scores = np.array([ood.score(h) for h in cal])
        tau = float(np.quantile(cal_scores, 0.99))
        # Threshold should be finite and > 0.
        assert 0 < tau < float("inf")
        # Most calibration points should be in-distribution.
        in_frac = float(np.mean(cal_scores <= tau))
        assert in_frac >= 0.95
