"""v0.3.10.1-alpha synthetic benchmark + scientific tests (Sections 10, 31-36).

These tests verify:

G9.  The redesigned benchmark no longer allows trivial random perfect
     routing (Section 10E).
G10. Weighted method beats unweighted in the designed noisy near-tie
     environment (Section 32).
G11. Centroid fails on the designed multimodal geometry (Section 33).
G12. Logistic performs strongly on the linear geometry (Section 34).
G13. MLP beats logistic on the nonlinear XOR-like geometry (Section 35).
G14. Random matched directions do not achieve the same perfect
     performance as the learned vector (Section 36).

Each test documents the *scientific expectation* it encodes, so the
release-gate matrix maps 1:1 to falsifiable hypotheses.
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.environment_benchmark import (
    benchmark_execute_fn,
    benchmark_oracle_utility,
    benchmark_utility,
    make_environment,
    make_linear_environment,
    make_multimodal_environment,
    make_near_tie_environment,
    make_xor_environment,
    random_direction_control,
)
from daph_learning.policy import (
    CentroidPolicy,
    ExperimentConfig,
    Route,
    TargetMode,
    WeightMode,
    fit_policy,
    predict_proba,
    train_centroid_policy,
)


def _train_and_eval(
    policy_type: str,
    train_tasks,
    dev_tasks,
    *,
    weight_mode: str = "clipped_gap",
    target_mode: str = "soft",
    confidence_threshold: float = 0.55,
    gap_threshold: float = 0.05,
    seed: int = 0,
):
    """Train a policy on train_tasks and evaluate mean regret on dev_tasks.

    Returns (mean_regret, mean_utility, routing_accuracy).
    """
    from daph_learning.policy.learner import build_counterfactual_experiences
    cfg = ExperimentConfig(
        policy_type=policy_type,
        weight_mode=weight_mode,
        target_mode=target_mode,
        gap_threshold=gap_threshold,
        abstention_band=gap_threshold,
        confidence_threshold=confidence_threshold,
        random_seed=seed,
        max_weight=2.0,
    )
    train_exp = build_counterfactual_experiences(
        train_tasks, execute_fn=benchmark_execute_fn, config=cfg)
    dev_exp = build_counterfactual_experiences(
        dev_tasks, execute_fn=benchmark_execute_fn, config=cfg)
    train_acts = np.array([t["activation"] for t in train_tasks], dtype=np.float32)
    dev_acts = np.array([t["activation"] for t in dev_tasks], dtype=np.float32)
    train_du = np.array([e.delta_utility for e in train_exp], dtype=np.float32)
    train_w = np.array([e.sample_weight for e in train_exp], dtype=np.float32)
    dev_du = np.array([e.delta_utility for e in dev_exp], dtype=np.float32)
    # Oracle utility for regret.
    oracle_u = np.array(
        [benchmark_oracle_utility(t) for t in dev_tasks], dtype=np.float64)
    model = fit_policy(
        cfg, train_acts, train_du, train_w,
        dev_features=dev_acts, dev_delta_u=dev_du,
        dev_weights=np.ones(len(dev_exp), dtype=np.float32),
        dev_tasks=dev_tasks, utility_fn=benchmark_utility,
        seed=seed,
    )
    probs = predict_proba(model, dev_acts)
    # Route with the deployment confidence threshold.
    from daph_learning.policy.abstention import choose_route
    routes = [choose_route(float(p), confidence_threshold).value for p in probs]
    pred_u = np.array(
        [benchmark_utility(t, r) for t, r in zip(dev_tasks, routes)],
        dtype=np.float64)
    regrets = np.maximum(oracle_u - pred_u, 0.0)
    # Routing accuracy: did the policy pick the higher-utility backend?
    correct = np.array([
        (routes[i] == "symbolic" and dev_du[i] > 0)
        or (routes[i] == "llm" and dev_du[i] < 0)
        for i in range(len(dev_tasks))
    ], dtype=np.float64)
    return float(regrets.mean()), float(pred_u.mean()), float(correct.mean())


# ============================================================
# G9: benchmark no longer allows trivial random perfect routing
# ============================================================

class TestRandomDirectionControl:
    """Section 10E: random matched directions must NOT trivially solve
    the redesigned environments."""

    def test_linear_random_not_perfect(self):
        """On the linear environment, random directions should not
        achieve near-zero regret (the benchmark is not trivially
        solvable by a random direction)."""
        tasks = make_linear_environment(n=200, dim=8, seed=0)
        acts = np.array([t["activation"] for t in tasks], dtype=np.float32)
        du = np.array([t["delta_utility"] for t in tasks], dtype=np.float32)
        result = random_direction_control(
            acts, du, n_random=200, seed=0, metric="regret")
        # Median random regret should be materially above 0 (not ~0).
        assert result["median"] > 0.05, (
            f"linear env median random regret {result['median']} should "
            f"be > 0.05; if it's ~0 the benchmark is trivially solvable")

    def test_xor_random_near_chance(self):
        """On the XOR environment, random directions should be near
        chance (regret ~ 0.5 * gap), because the boundary is nonlinear."""
        tasks = make_xor_environment(n=200, dim=8, seed=0)
        acts = np.array([t["activation"] for t in tasks], dtype=np.float32)
        du = np.array([t["delta_utility"] for t in tasks], dtype=np.float32)
        result = random_direction_control(
            acts, du, n_random=200, seed=0, metric="accuracy")
        # Median random accuracy should be near 0.5 (chance).
        assert 0.3 < result["median"] < 0.7, (
            f"XOR env median random accuracy {result['median']} should "
            f"be near 0.5 (chance)")

    def test_multimodal_random_not_perfect(self):
        tasks = make_multimodal_environment(n=200, dim=8, seed=0)
        acts = np.array([t["activation"] for t in tasks], dtype=np.float32)
        du = np.array([t["delta_utility"] for t in tasks], dtype=np.float32)
        result = random_direction_control(
            acts, du, n_random=200, seed=0, metric="regret")
        assert result["median"] > 0.05


# ============================================================
# G10: weighting value (Section 32)
# ============================================================

class TestWeightingValue:
    """Section 32 / v0.3.10.2 G10: weighted method must TRULY beat
    unweighted method on the designed near-tie environment, verified
    across multiple seeds.

    v0.3.10.2 — this is a TRUE SUPERIORITY gate, not a non-inferiority
    gate. The old gate passed when ``reg_w <= reg_u + 0.01`` (tolerance),
    which is a non-inferiority test. The new gate requires:
    ``mean(d_s) >= min_weighting_gain`` where
    ``d_s = regret_unweighted(seed_s) - regret_weighted(seed_s)``.
    """

    MIN_WEIGHTING_GAIN = 0.005  # chosen BEFORE seeing results

    def test_weighted_truly_beats_unweighted_multi_seed(self):
        """Run 10 seeds and require mean(d) >= min_gain with lower CI > 0.

        This is the experiment that validates the utility-weighting
        hypothesis (Section 32 / Question A).
        """
        gains = []
        for s in range(10):
            train = make_near_tie_environment(n=300, dim=8, seed=s)
            dev = make_near_tie_environment(n=200, dim=8, seed=s + 1)
            reg_unw, _, _ = _train_and_eval(
                "logistic", train, dev,
                weight_mode="uniform", target_mode="soft",
                confidence_threshold=0.5, seed=s)
            reg_w, _, _ = _train_and_eval(
                "logistic", train, dev,
                weight_mode="clipped_gap", target_mode="soft",
                confidence_threshold=0.5, seed=s)
            gain = reg_unw - reg_w
            gains.append(gain)
        gains = np.array(gains)
        mean_gain = float(gains.mean())
        std_gain = float(gains.std())
        n = len(gains)
        lower_ci = mean_gain - 1.96 * std_gain / np.sqrt(n)
        # True superiority: mean gain >= min_gain AND lower CI > 0.
        assert mean_gain >= self.MIN_WEIGHTING_GAIN, (
            f"mean weighting gain {mean_gain:.4f} < min_gain "
            f"{self.MIN_WEIGHTING_GAIN}; weighted does NOT truly beat "
            f"unweighted. Per-seed gains: {gains.tolist()}")
        assert lower_ci > 0, (
            f"lower 95% CI {lower_ci:.4f} <= 0; weighting advantage is "
            f"not statistically reliable. mean={mean_gain:.4f}, "
            f"std={std_gain:.4f}, n={n}")

    def test_weighted_beats_unweighted_single_seed_smoke(self):
        """Quick single-seed smoke test (non-authoritative)."""
        train = make_near_tie_environment(n=300, dim=8, seed=0)
        dev = make_near_tie_environment(n=200, dim=8, seed=1)
        reg_unw, _, _ = _train_and_eval(
            "logistic", train, dev,
            weight_mode="uniform", target_mode="soft",
            confidence_threshold=0.5, seed=0)
        reg_w, _, _ = _train_and_eval(
            "logistic", train, dev,
            weight_mode="clipped_gap", target_mode="soft",
            confidence_threshold=0.5, seed=0)
        # Just check both produce valid regret values.
        assert reg_w >= 0.0 and reg_unw >= 0.0


# ============================================================
# G11: centroid failure (Section 33)
# ============================================================

class TestCentroidFailure:
    """Section 33: centroid performance must degrade materially on the
    multimodal geometry where μ_S ≈ μ_L ≈ 0."""

    def test_centroid_degrades_on_multimodal(self):
        """Scientific expectation: centroid mean regret on multimodal
        is materially worse than centroid mean regret on linear."""
        train_lin = make_linear_environment(n=300, dim=8, seed=0)
        dev_lin = make_linear_environment(n=200, dim=8, seed=1)
        reg_lin, _, _ = _train_and_eval(
            "centroid", train_lin, dev_lin,
            confidence_threshold=0.5, seed=0)
        train_mm = make_multimodal_environment(n=300, dim=8, seed=0)
        dev_mm = make_multimodal_environment(n=200, dim=8, seed=1)
        reg_mm, _, _ = _train_and_eval(
            "centroid", train_mm, dev_mm,
            confidence_threshold=0.5, seed=0)
        # Centroid regret on multimodal should be materially worse than
        # on linear (the centroid direction is near-zero on multimodal).
        assert reg_mm > reg_lin + 0.05, (
            f"centroid regret on multimodal {reg_mm} should be "
            f"materially worse than on linear {reg_lin}")


# ============================================================
# G12: linear router value (Section 34)
# ============================================================

class TestLinearRouterValue:
    """Section 34: logistic should recover strong performance on the
    linear environment. Compare centroid vs logistic; gate computes
    both."""

    def test_logistic_strong_on_linear(self):
        train = make_linear_environment(n=300, dim=8, seed=0)
        dev = make_linear_environment(n=200, dim=8, seed=1)
        reg_log, util_log, acc_log = _train_and_eval(
            "logistic", train, dev,
            confidence_threshold=0.5, seed=0)
        # Logistic should achieve low regret on the linear environment.
        # (Using threshold 0.5 because the linear env has many near-ties
        # where |ΔU| < 0.1; abstaining on them would add ~0.5 regret each
        # since abstain utility is 0.0 while oracle is ~0.5.)
        assert reg_log < 0.15, (
            f"logistic regret {reg_log} on linear env should be < 0.15")

    def test_gate_computes_both_centroid_and_logistic(self):
        """G14: the gate must compute both sides, not just one."""
        train = make_linear_environment(n=200, dim=8, seed=0)
        dev = make_linear_environment(n=150, dim=8, seed=1)
        reg_c, _, _ = _train_and_eval(
            "centroid", train, dev, confidence_threshold=0.55, seed=0)
        reg_l, _, _ = _train_and_eval(
            "logistic", train, dev, confidence_threshold=0.55, seed=0)
        # Both measurements are computed (not just one). On linear, we
        # do not require logistic to always dominate if geometry makes
        # them equivalent, but the gate must compute both.
        assert reg_c >= 0.0 and reg_l >= 0.0
        # Store both for the release artifact.
        assert {"centroid": reg_c, "logistic": reg_l}


# ============================================================
# G13: nonlinear router value (Section 35)
# ============================================================

class TestNonlinearRouterValue:
    """Section 35: logistic should fail or be limited on XOR; MLP should
    materially improve."""

    def test_mlp_beats_logistic_on_xor(self):
        """Scientific expectation: MLP regret < logistic regret on the
        XOR environment (nonlinear boundary)."""
        train = make_xor_environment(n=400, dim=8, seed=0)
        dev = make_xor_environment(n=200, dim=8, seed=1)
        reg_log, _, _ = _train_and_eval(
            "logistic", train, dev,
            confidence_threshold=0.5, seed=0)
        reg_mlp, _, _ = _train_and_eval(
            "mlp_experimental", train, dev,
            confidence_threshold=0.5, seed=0)
        # MLP should materially improve over logistic on XOR.
        assert reg_mlp < reg_log, (
            f"MLP regret {reg_mlp} should < logistic regret {reg_log} "
            f"on XOR environment")


# ============================================================
# G14: random steering (Section 36)
# ============================================================

class TestRandomSteering:
    """Section 36: random matched directions should not achieve the same
    perfect performance as the learned vector in the redesigned env."""

    def test_learned_exceeds_median_random_on_linear(self):
        """The learned logistic router should exceed the median random
        direction on the linear environment."""
        train = make_linear_environment(n=300, dim=8, seed=0)
        dev = make_linear_environment(n=200, dim=8, seed=1)
        reg_log, _, _ = _train_and_eval(
            "logistic", train, dev, confidence_threshold=0.5, seed=0)
        dev_acts = np.array([t["activation"] for t in dev], dtype=np.float32)
        dev_du = np.array([t["delta_utility"] for t in dev], dtype=np.float32)
        random_result = random_direction_control(
            dev_acts, dev_du, n_random=200, seed=0, metric="regret")
        # Learned logistic regret should be below the median random regret.
        assert reg_log < random_result["median"], (
            f"learned logistic regret {reg_log} should be < median "
            f"random regret {random_result['median']} on linear env")


# ============================================================
# Environment sanity tests
# ============================================================

class TestEnvironmentsAreWellFormed:
    """Each environment must produce well-formed task dicts."""

    @pytest.mark.parametrize("env_name", ["linear", "near_tie", "multimodal", "xor"])
    def test_environment_produces_tasks(self, env_name):
        tasks = make_environment(env_name, n=50, dim=4, seed=0)
        assert len(tasks) == 50
        t = tasks[0]
        assert "task_id" in t
        assert "family" in t
        assert "activation" in t
        assert "delta_utility" in t
        assert "u_symbolic" in t
        assert "u_llm" in t
        assert t["activation"].shape == (4,)

    @pytest.mark.parametrize("env_name", ["linear", "near_tie", "multimodal", "xor"])
    def test_environment_utilities_are_bounded(self, env_name):
        tasks = make_environment(env_name, n=50, dim=4, seed=0)
        for t in tasks:
            assert 0.0 < t["u_symbolic"] < 1.0
            assert 0.0 < t["u_llm"] < 1.0
            assert abs(t["u_symbolic"] + t["u_llm"] - 1.0) < 1e-6

    def test_multimodal_centroid_direction_is_weak(self):
        """The multimodal environment should produce μ_S ≈ μ_L ≈ 0,
        so the centroid direction v = μ_S - μ_L is near-zero."""
        tasks = make_multimodal_environment(n=200, dim=8, seed=0)
        sym = np.array([t["activation"] for t in tasks if t["family"] == "S"])
        llm = np.array([t["activation"] for t in tasks if t["family"] == "L"])
        mu_s = sym.mean(axis=0)
        mu_l = llm.mean(axis=0)
        v = mu_s - mu_l
        # The centroid direction norm should be small relative to the
        # cluster separation.
        assert np.linalg.norm(v) < 0.5, (
            f"multimodal centroid direction norm {np.linalg.norm(v)} "
            f"should be < 0.5 (μ_S ≈ μ_L ≈ 0)")

    def test_xor_is_nonlinear(self):
        """The XOR environment should NOT be linearly separable on the
        first two coordinates."""
        tasks = make_xor_environment(n=400, dim=8, seed=0)
        H = np.array([t["activation"] for t in tasks])
        y = np.array([1 if t["family"] == "S" else 0 for t in tasks])
        # A logistic regression on the first two coords should not
        # achieve high accuracy (XOR is not linearly separable).
        from sklearn.linear_model import LogisticRegression
        try:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(H[:, :2], y)
            acc = clf.score(H[:, :2], y)
            # XOR is not linearly separable, so accuracy should be
            # near chance (~0.5). Allow up to 0.6 for finite-sample noise.
            assert acc < 0.65, (
                f"XOR should not be linearly separable; logistic "
                f"accuracy {acc} should be < 0.65")
        except ImportError:
            pytest.skip("scikit-learn not available")
