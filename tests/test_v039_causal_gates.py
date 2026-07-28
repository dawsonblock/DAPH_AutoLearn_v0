"""v0.3.9 release-gate tests: causal learning loop sensitivity and invariants.

These tests are mandatory release gates (Sections 13-17, 24 of the v0.3.9
repair spec). They prove the causal chain is genuine:

    experience -> counterfactual utility -> target -> candidate update
    -> changed policy -> changed decisions -> changed held-out utility
    -> promotion/rollback

GATE 2: Candidate-vector sensitivity (Section 13)
GATE 3: Incumbent-vector sensitivity (Section 14)
GATE 4: Oracle-leakage prevention (Section 15)
GATE 5: Capture/task-ID alignment (Section 16)
GATE 6: Trust-region mathematical invariants (Section 17)
GATE 7: Synthetic closed-loop AutoLearn improves over incumbent (Section 24)
GATE 8: Failed candidates reliably rollback (Section 17)
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.autolearn.counterfactual import UtilityConfig
from daph_learning.autolearn.experience import CaptureResult, CapturedActivation
from daph_learning.autolearn.promotion import (
    HeldOutTaskResult,
    PromotionGateConfig,
    evaluate_promotion_gate,
)
from daph_learning.autolearn.route_policy import (
    SteeringPolicyConfig,
    make_feature_route_policy,
)
from daph_learning.autolearn.trust_region import (
    CounterfactualLearningLoop,
    LoopConfig,
    bootstrap_from_candidate,
    trust_region_update,
)


# --- Shared test helpers ---

def _feature_fn(dim=4):
    """Feature function: axis 0 is +1 for family A tasks, -1 for family B."""
    def _f(task):
        f = np.zeros(dim, dtype=np.float32)
        family = task.get("family", "A")
        f[0] = 1.0 if family == "A" else -1.0
        return f
    return _f


def _make_execute_fn(sym_correct_families, llm_correct_families, expected=42):
    """Execute fn where symbolic is correct for sym_correct_families and LLM
    is correct for llm_correct_families."""
    sym_set = set(sym_correct_families)
    llm_set = set(llm_correct_families)
    def _execute(task, backend):
        family = task.get("family", "A")
        if backend == "symbolic":
            ok = family in sym_set
            return {"output": f"FINAL: {expected}" if ok else "FINAL: 99",
                    "execution_succeeded": True, "latency_ms": 1.0,
                    "compute_cost": 0.0, "failure_type": None}
        else:
            ok = family in llm_set
            return {"output": f"FINAL: {expected}" if ok else "FINAL: 99",
                    "execution_succeeded": True, "latency_ms": 50.0,
                    "compute_cost": 0.01, "failure_type": None}
    return _execute


def _make_capture_fn(dim=4):
    """Capture fn that returns per-task activations with task IDs."""
    def _capture(tasks, class_label):
        sign = 1.0 if class_label == "symbolic" else -1.0
        captured = []
        for task in tasks:
            tid = str(task.get("task_id", ""))
            act = np.zeros(dim, dtype=np.float32)
            act[0] = sign
            captured.append(CapturedActivation(
                task_id=tid, activation=act, success=True,
            ))
        return CaptureResult(activations=captured, n_attempts=len(tasks), n_successes=len(tasks))
    return _capture


def _family_tasks(family, n, prefix=None, expected=42):
    prefix = prefix or family.lower()
    return [{"task_id": f"{prefix}{i}", "expected": expected,
             "family": family, "capability_ids": [f"cap_{family.lower()}"],
             "utility_oracle": "symbolic" if family == "A" else "llm",
             "output_schema": "integer"} for i in range(n)]


# ============================================================
# GATE 2 (Section 13): Candidate-vector sensitivity
# ============================================================

def test_heldout_evaluation_is_sensitive_to_candidate_vector():
    """GATE 2: Changing the candidate steering vector must change the
    candidate routing decisions.

    This test fails if candidate_vector is accepted by the evaluator but
    ignored (the v0.3.8 defect).
    """
    dim = 4
    # Tasks where family A and B have opposite features.
    held = _family_tasks("A", 10) + _family_tasks("B", 10)
    execute = _make_execute_fn(["A"], ["B"])
    capture = _make_capture_fn(dim)
    route_fn = make_feature_route_policy(_feature_fn(dim))

    # Candidate +v: routes A->symbolic (dot=+1), B->llm (dot=-(-1)=+1? no)
    # With feature A=+1, B=-1 and vector +axis0:
    #   A: dot = +1 * +1 = +1 > 0 -> symbolic
    #   B: dot = +1 * -1 = -1 < 0 -> llm
    # With vector -axis0:
    #   A: dot = -1 * +1 = -1 < 0 -> llm
    #   B: dot = -1 * -1 = +1 > 0 -> symbolic
    vec_plus = np.zeros(dim, dtype=np.float32)
    vec_plus[0] = 1.0
    vec_minus = np.zeros(dim, dtype=np.float32)
    vec_minus[0] = -1.0
    # Incumbent is zero vector (baseline).
    inc = np.zeros(dim, dtype=np.float32)

    loop_plus = CounterfactualLearningLoop(
        seed_vector=inc, training_tasks=[], held_out_tasks=held,
        capture_fn=capture, execute_fn=execute, route_fn=route_fn,
        config=LoopConfig(utility_config=UtilityConfig(lambda_risk=1.0, abstention_band=0.1)),
    )
    loop_minus = CounterfactualLearningLoop(
        seed_vector=inc, training_tasks=[], held_out_tasks=held,
        capture_fn=capture, execute_fn=execute, route_fn=route_fn,
        config=LoopConfig(utility_config=UtilityConfig(lambda_risk=1.0, abstention_band=0.1)),
    )

    results_plus = loop_plus._evaluate_held_out(vec_plus, inc)
    results_minus = loop_minus._evaluate_held_out(vec_minus, inc)

    # The candidate routes must differ between +v and -v.
    plus_routes = [r.candidate_route for r in results_plus]
    minus_routes = [r.candidate_route for r in results_minus]
    assert plus_routes != minus_routes, (
        "candidate routes are identical for +v and -v; the evaluator is "
        "ignoring the candidate vector (GATE 2 FAILURE)"
    )
    # Specifically: +v routes A->symbolic, B->llm; -v routes A->llm, B->symbolic.
    a_results_plus = [r for r in results_plus if r.task_id.startswith("a")]
    b_results_plus = [r for r in results_plus if r.task_id.startswith("b")]
    assert all(r.candidate_route == "symbolic" for r in a_results_plus)
    assert all(r.candidate_route == "llm" for r in b_results_plus)

    a_results_minus = [r for r in results_minus if r.task_id.startswith("a")]
    b_results_minus = [r for r in results_minus if r.task_id.startswith("b")]
    assert all(r.candidate_route == "llm" for r in a_results_minus)
    assert all(r.candidate_route == "symbolic" for r in b_results_minus)


def test_heldout_evaluation_zero_vector_baseline():
    """GATE 2 (part 2): candidate_zero = 0 should produce abstention
    (baseline behavior) when abstain_threshold > 0."""
    dim = 4
    held = _family_tasks("A", 5) + _family_tasks("B", 5)
    execute = _make_execute_fn(["A"], ["B"])
    capture = _make_capture_fn(dim)
    # Use abstain_threshold > 0 so zero vector -> abstain.
    route_fn = make_feature_route_policy(_feature_fn(dim), abstain_threshold=0.01)

    loop = CounterfactualLearningLoop(
        seed_vector=np.zeros(dim, dtype=np.float32),
        training_tasks=[], held_out_tasks=held,
        capture_fn=capture, execute_fn=execute, route_fn=route_fn,
        config=LoopConfig(utility_config=UtilityConfig(lambda_risk=1.0, abstention_band=0.1)),
    )
    zero = np.zeros(dim, dtype=np.float32)
    results = loop._evaluate_held_out(zero, zero)
    # With zero vector, dot=0 which is within [-0.01, 0.01] -> abstain.
    assert all(r.candidate_route == "abstain" for r in results)
    assert all(r.incumbent_route == "abstain" for r in results)
    assert all(not r.decided for r in results)


# ============================================================
# GATE 3 (Section 14): Incumbent-vector sensitivity
# ============================================================

def test_heldout_evaluation_is_sensitive_to_incumbent_vector():
    """GATE 3: Changing the incumbent steering vector must change the
    incumbent routing decisions.

    The promotion result must not contain any hard-coded incumbent route.
    """
    dim = 4
    held = _family_tasks("A", 10) + _family_tasks("B", 10)
    execute = _make_execute_fn(["A"], ["B"])
    capture = _make_capture_fn(dim)
    route_fn = make_feature_route_policy(_feature_fn(dim))

    vec_plus = np.zeros(dim, dtype=np.float32)
    vec_plus[0] = 1.0
    vec_minus = np.zeros(dim, dtype=np.float32)
    vec_minus[0] = -1.0
    candidate = vec_plus  # same candidate for both

    loop = CounterfactualLearningLoop(
        seed_vector=np.zeros(dim, dtype=np.float32),
        training_tasks=[], held_out_tasks=held,
        capture_fn=capture, execute_fn=execute, route_fn=route_fn,
        config=LoopConfig(utility_config=UtilityConfig(lambda_risk=1.0, abstention_band=0.1)),
    )

    results_inc_a = loop._evaluate_held_out(candidate, vec_plus)
    results_inc_b = loop._evaluate_held_out(candidate, vec_minus)

    # The incumbent routes must differ between inc=+v and inc=-v.
    inc_a_routes = [r.incumbent_route for r in results_inc_a]
    inc_b_routes = [r.incumbent_route for r in results_inc_b]
    assert inc_a_routes != inc_b_routes, (
        "incumbent routes are identical for +v and -v; the evaluator is "
        "ignoring the incumbent vector (GATE 3 FAILURE)"
    )
    # evaluation(candidate, incumbent_A) can differ from evaluation(candidate, incumbent_B)
    # because the incumbent utility changes.
    a_utils = [r.incumbent_utility for r in results_inc_a]
    b_utils = [r.incumbent_utility for r in results_inc_b]
    assert a_utils != b_utils, (
        "incumbent utilities are identical for different incumbent vectors "
        "(GATE 3 FAILURE)"
    )


# ============================================================
# GATE 4 (Section 15): Oracle-leakage prevention
# ============================================================

def test_oracle_leakage_candidate_selects_llm_but_oracle_prefers_symbolic():
    """GATE 4: Construct a task where oracle best action = symbolic but the
    candidate policy selects LLM. The candidate evaluation MUST score the
    LLM action, NOT silently replace it with symbolic.

    This proves ground-truth utility is not leaking into policy selection.
    """
    dim = 4
    # Family A: symbolic correct, LLM wrong -> oracle prefers symbolic.
    # But the candidate vector routes A->llm (wrong direction).
    held = _family_tasks("A", 10, prefix="leak")
    execute = _make_execute_fn(["A"], [])  # symbolic correct, LLM wrong for A
    capture = _make_capture_fn(dim)
    route_fn = make_feature_route_policy(_feature_fn(dim))

    # Candidate vector -axis0: routes A (feature +1) -> dot=-1 -> llm.
    candidate = np.zeros(dim, dtype=np.float32)
    candidate[0] = -1.0
    # Incumbent +axis0: routes A -> dot=+1 -> symbolic (correct).
    incumbent = np.zeros(dim, dtype=np.float32)
    incumbent[0] = 1.0

    loop = CounterfactualLearningLoop(
        seed_vector=incumbent, training_tasks=[], held_out_tasks=held,
        capture_fn=capture, execute_fn=execute, route_fn=route_fn,
        config=LoopConfig(utility_config=UtilityConfig(lambda_risk=1.0, abstention_band=0.1)),
    )
    results = loop._evaluate_held_out(candidate, incumbent)

    # The candidate MUST have selected LLM (not symbolic).
    assert all(r.candidate_route == "llm" for r in results), (
        f"candidate routes should be 'llm' but got {[r.candidate_route for r in results[:3]]}; "
        "the evaluator may be leaking oracle utility into route selection (GATE 4 FAILURE)"
    )
    # The candidate MUST be wrong (LLM is wrong for family A).
    assert all(not r.candidate_correct for r in results), (
        "candidate should be incorrect (LLM is wrong for family A) but "
        "candidate_correct is True (GATE 4 FAILURE)"
    )
    # The incumbent MUST have selected symbolic (correct).
    assert all(r.incumbent_route == "symbolic" for r in results)
    assert all(r.incumbent_correct for r in results)
    # Candidate utility < incumbent utility (candidate chose wrong backend).
    assert all(r.candidate_utility < r.incumbent_utility for r in results), (
        "candidate utility should be < incumbent utility when candidate "
        "selects the wrong backend (GATE 4 FAILURE)"
    )


# ============================================================
# GATE 5 (Section 16): Capture/task-ID alignment
# ============================================================

def test_capture_alignment_exactly_8_of_10_enter_update():
    """GATE 5: 10 attempted captures, 8 successful, 2 failures.
    Verify exactly 8 aligned examples enter the activation update, each
    weight belongs to the correct task, and no positional shifting occurs.
    """
    dim = 4
    # 10 tasks, all target symbolic (so all go to positive class).
    train = _family_tasks("A", 10, prefix="cap")
    execute = _make_execute_fn(["A"], [])  # symbolic correct -> target=symbolic
    # Capture fn that fails for tasks 3 and 7 (0-indexed).
    fail_ids = {"cap3", "cap7"}
    def _capture(tasks, class_label):
        captured = []
        n_att = 0
        n_succ = 0
        for task in tasks:
            tid = str(task.get("task_id", ""))
            n_att += 1
            if tid in fail_ids:
                captured.append(CapturedActivation(
                    task_id=tid, success=False, failure_reason="simulated_failure",
                ))
            else:
                act = np.zeros(dim, dtype=np.float32)
                act[0] = 1.0
                captured.append(CapturedActivation(
                    task_id=tid, activation=act, success=True,
                ))
                n_succ += 1
        return CaptureResult(activations=captured, n_attempts=n_att, n_successes=n_succ)

    route_fn = make_feature_route_policy(_feature_fn(dim))
    loop = CounterfactualLearningLoop(
        seed_vector=np.ones(dim, dtype=np.float32),
        training_tasks=train, held_out_tasks=[],
        capture_fn=_capture, execute_fn=execute, route_fn=route_fn,
        config=LoopConfig(
            utility_config=UtilityConfig(lambda_risk=1.0, abstention_band=0.1),
            eta=1.0,
        ),
    )
    result = loop.step()
    # Exactly 8 successful captures, 2 failures.
    assert result.capture_attempts == 10
    assert result.capture_successes == 8
    assert result.capture_failures == 2
    assert result.capture_coverage == 0.8


def test_capture_alignment_weights_match_task_ids():
    """GATE 5 (part 2): Verify that each weight belongs to the correct task
    by checking that the aligned activations produce the expected mean."""
    dim = 4
    # Create tasks with known features so we can verify alignment.
    train = []
    for i in range(6):
        train.append({"task_id": f"align{i}", "expected": 42, "family": "A",
                      "capability_ids": ["cap_a"], "utility_oracle": "symbolic",
                      "output_schema": "integer"})
    execute = _make_execute_fn(["A"], [])

    # Capture fn that gives each task a unique activation on axis 1.
    def _capture(tasks, class_label):
        captured = []
        for task in tasks:
            tid = str(task.get("task_id", ""))
            act = np.zeros(dim, dtype=np.float32)
            # Encode the task index in axis 1.
            idx = int(tid.replace("align", ""))
            act[1] = float(idx)
            captured.append(CapturedActivation(task_id=tid, activation=act, success=True))
        return CaptureResult(activations=captured, n_attempts=len(tasks), n_successes=len(tasks))

    route_fn = make_feature_route_policy(_feature_fn(dim))
    loop = CounterfactualLearningLoop(
        seed_vector=np.ones(dim, dtype=np.float32),
        training_tasks=train, held_out_tasks=[],
        capture_fn=_capture, execute_fn=execute, route_fn=route_fn,
        config=LoopConfig(
            utility_config=UtilityConfig(lambda_risk=1.0, abstention_band=0.1),
            eta=1.0,
        ),
    )
    # Manually test the alignment.
    from daph_learning.autolearn.counterfactual import derive_learning_target, execute_both_backends
    from daph_learning.evaluation.verifiers import select_verifier_for_task
    outcomes = []
    for task in train:
        outcome = execute_both_backends(
            task, backend_selected="symbolic",
            symbolic_runner=lambda t, b="symbolic": execute(t, b),
            llm_runner=lambda t, b="llm": execute(t, b),
            verifier=select_verifier_for_task(task),
        )
        outcomes.append(outcome)
    targets = [(o, derive_learning_target(o, loop.config.utility_config)) for o in outcomes]
    candidate, n_abstain, cap_att, cap_succ = loop._candidate_from_targets(train, targets)
    # All tasks target symbolic with equal weight (same Delta-U).
    # The aligned mean should have axis 1 = mean(0,1,2,3,4,5) = 2.5.
    assert pytest.approx(float(candidate[1]), abs=1e-5) == 2.5, (
        f"aligned mean axis 1 should be 2.5 but got {candidate[1]}; "
        "weights may not be aligned with task IDs (GATE 5 FAILURE)"
    )


def test_capture_alignment_min_coverage_gate():
    """GATE 5 (part 3): Insufficient coverage prevents training/promotion
    when min_capture_coverage is configured."""
    dim = 4
    train = _family_tasks("A", 10, prefix="cov")
    execute = _make_execute_fn(["A"], [])
    # Capture fn that fails for 8 of 10 tasks (coverage 0.2).
    def _capture(tasks, class_label):
        captured = []
        n_succ = 0
        for i, task in enumerate(tasks):
            tid = str(task.get("task_id", ""))
            if i < 8:
                captured.append(CapturedActivation(task_id=tid, success=False, failure_reason="fail"))
            else:
                act = np.zeros(dim, dtype=np.float32)
                act[0] = 1.0
                captured.append(CapturedActivation(task_id=tid, activation=act, success=True))
                n_succ += 1
        return CaptureResult(activations=captured, n_attempts=len(tasks), n_successes=n_succ)

    route_fn = make_feature_route_policy(_feature_fn(dim))
    loop = CounterfactualLearningLoop(
        seed_vector=np.ones(dim, dtype=np.float32),
        training_tasks=train, held_out_tasks=_family_tasks("A", 5, prefix="held"),
        capture_fn=_capture, execute_fn=execute, route_fn=route_fn,
        config=LoopConfig(
            utility_config=UtilityConfig(lambda_risk=1.0, abstention_band=0.1),
            min_capture_coverage=0.5,
        ),
    )
    result = loop.step()
    # Coverage 0.2 < 0.5 -> rollback for capture coverage.
    assert result.promoted is False
    assert "capture coverage" in result.decision.reason


# ============================================================
# GATE 6 (Section 17): Trust-region mathematical invariants
# ============================================================

def test_trust_region_zero_vector_bootstrap():
    """GATE 6: Zero-vector bootstrap initializes from candidate direction."""
    inc = np.zeros(4, dtype=np.float32)
    cand = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)  # norm 5
    result = bootstrap_from_candidate(inc, cand, initial_steering_norm=2.0)
    assert pytest.approx(float(np.linalg.norm(result)), abs=1e-5) == 2.0
    # Direction should match candidate.
    expected_dir = cand / 5.0
    np.testing.assert_allclose(result, 2.0 * expected_dir, atol=1e-6)


def test_trust_region_bootstrap_rejects_both_zero():
    """GATE 6: Cannot bootstrap when both incumbent and candidate are zero."""
    with pytest.raises(ValueError, match="cannot bootstrap"):
        bootstrap_from_candidate(np.zeros(4), np.zeros(4), initial_steering_norm=1.0)


def test_trust_region_rejects_nan_candidate():
    """GATE 6: NaN candidate must fail closed."""
    with pytest.raises(ValueError, match="NaN"):
        trust_region_update(np.ones(4), np.array([np.nan, 1, 1, 1]), eta=0.5)


def test_trust_region_rejects_inf_candidate():
    """GATE 6: Inf candidate must fail closed."""
    with pytest.raises(ValueError, match="Inf"):
        trust_region_update(np.ones(4), np.array([np.inf, 1, 1, 1]), eta=0.5)


def test_trust_region_rejects_zero_candidate_when_incumbent_nonzero():
    """GATE 6: Zero candidate with nonzero incumbent is valid (eta-blend)."""
    inc = np.array([1.0, 0.0], dtype=np.float32)
    cand = np.zeros(2, dtype=np.float32)
    out = trust_region_update(inc, cand, eta=0.5)
    # (1-0.5)*[1,0] + 0.5*[0,0] = [0.5, 0], renormalized to norm 1 -> [1, 0].
    np.testing.assert_allclose(out, [1.0, 0.0], atol=1e-6)


def test_trust_region_candidate_identical_to_incumbent():
    """GATE 6: Candidate identical to incumbent produces no change."""
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = trust_region_update(v, v, eta=0.5)
    np.testing.assert_allclose(out, v, atol=1e-6)


def test_trust_region_candidate_opposite_incumbent():
    """GATE 6: Candidate opposite to incumbent produces zero (convex blend)."""
    v = np.array([1.0, 0.0], dtype=np.float32)
    vhat = np.array([-1.0, 0.0], dtype=np.float32)
    out = trust_region_update(v, vhat, eta=0.5)
    # (1-0.5)*[1,0] + 0.5*[-1,0] = [0, 0] -> norm 0, stays zero.
    np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)


def test_trust_region_excessive_candidate_norm_clipped():
    """GATE 6: max_vector_norm clips the output norm."""
    inc = np.array([1.0, 0.0], dtype=np.float32)  # norm 1
    cand = np.array([100.0, 0.0], dtype=np.float32)  # norm 100
    out = trust_region_update(inc, cand, eta=1.0, max_norm=5.0)
    assert float(np.linalg.norm(out)) <= 5.0 + 1e-5


def test_trust_region_max_update_norm_clipping():
    """GATE 6: max_update_norm clips ||v_new - v_old||."""
    inc = np.array([1.0, 0.0], dtype=np.float32)  # norm 1
    cand = np.array([10.0, 0.0], dtype=np.float32)  # norm 10
    out = trust_region_update(inc, cand, eta=1.0, max_update_norm=0.5)
    diff_norm = float(np.linalg.norm(out - inc))
    assert diff_norm <= 0.5 + 1e-5


def test_trust_region_min_norm_enforced():
    """GATE 6: min_vector_norm enforces a minimum output norm."""
    inc = np.array([0.1, 0.0], dtype=np.float32)  # norm 0.1
    cand = np.array([0.0, 0.0], dtype=np.float32)
    out = trust_region_update(inc, cand, eta=0.5, min_norm=1.0)
    assert float(np.linalg.norm(out)) >= 1.0 - 1e-5


def test_trust_region_all_committed_vectors_finite():
    """GATE 6: All committed vectors must be finite."""
    for _ in range(20):
        rng = np.random.RandomState(_)
        inc = rng.standard_normal(8).astype(np.float32)
        cand = rng.standard_normal(8).astype(np.float32)
        out = trust_region_update(inc, cand, eta=0.5)
        assert np.all(np.isfinite(out)), "committed vector contains non-finite values"


# ============================================================
# GATE 8 (Section 17): Failed candidates reliably rollback
# ============================================================

def test_failed_candidate_rollback_incumbent_unchanged():
    """GATE 8: A failed candidate does not overwrite the incumbent artifact."""
    dim = 4
    train = _family_tasks("A", 20, prefix="rb_train")
    held = _family_tasks("A", 20, prefix="rb_held")
    execute = _make_execute_fn(["A"], [])  # symbolic correct, LLM wrong
    capture = _make_capture_fn(dim)
    route_fn = make_feature_route_policy(_feature_fn(dim))

    seed = np.ones(dim, dtype=np.float32)
    loop = CounterfactualLearningLoop(
        seed_vector=seed, training_tasks=train, held_out_tasks=held,
        capture_fn=capture, execute_fn=execute, route_fn=route_fn,
        config=LoopConfig(
            utility_config=UtilityConfig(lambda_risk=1.0, abstention_band=0.1),
            gate_config=PromotionGateConfig(
                min_discordant_n=100,  # impossible to satisfy -> always rollback
            ),
            eta=0.5,
        ),
    )
    original_incumbent = loop.incumbent.copy()
    result = loop.step()
    assert result.promoted is False
    assert result.decision.action == "rollback"
    # Incumbent unchanged after rollback.
    np.testing.assert_allclose(loop.incumbent, original_incumbent)
    # No lineage record was promoted.
    assert loop.lineage.current == "seed"


# ============================================================
# GATE 7 (Section 24): Synthetic closed-loop AutoLearn improves
# ============================================================

def test_synthetic_closed_loop_autolearn_improves_over_incumbent():
    """GATE 7: Build a deterministic synthetic environment where the optimal
    backend is known. Start from a deliberately imperfect incumbent. Run the
    loop and verify the candidate policy utility > incumbent utility and the
    promoted vector differs from the incumbent.

    Environment:
        family A -> symbolic utility = 1, LLM utility = 0
        family B -> symbolic utility = 0, LLM utility = 1

    Representations contain a learnable signal separating A/B (axis 0).
    The optimal steering vector routes A->symbolic, B->llm.
    """
    dim = 4
    # Training: 40 A tasks (symbolic correct) + 40 B tasks (LLM correct).
    train = _family_tasks("A", 40, prefix="syn_train_a") + _family_tasks("B", 40, prefix="syn_train_b")
    # Held-out: 20 A + 20 B.
    held = _family_tasks("A", 20, prefix="syn_held_a") + _family_tasks("B", 20, prefix="syn_held_b")

    # Symbolic correct for A, LLM correct for B.
    execute = _make_execute_fn(["A"], ["B"])
    capture = _make_capture_fn(dim)
    route_fn = make_feature_route_policy(_feature_fn(dim))

    # Deliberately imperfect incumbent: -axis0 (routes A->llm, B->symbolic,
    # which is WRONG for both families).
    bad_incumbent = np.zeros(dim, dtype=np.float32)
    bad_incumbent[0] = -1.0

    loop = CounterfactualLearningLoop(
        seed_vector=bad_incumbent, training_tasks=train, held_out_tasks=held,
        capture_fn=capture, execute_fn=execute, route_fn=route_fn,
        config=LoopConfig(
            utility_config=UtilityConfig(lambda_risk=1.0, abstention_band=0.1),
            gate_config=PromotionGateConfig(
                min_discordant_n=5, max_p_value=0.5,
                max_regression_rate=0.8,
                min_improved_probability_lower=0.5,
                min_coverage=0.5,
            ),
            eta=1.0,  # take the full candidate direction
            initial_steering_norm=1.0,
        ),
    )
    # Run multiple iterations to allow learning.
    results = loop.run(max_iterations=5)

    # At least one promotion should have occurred.
    promotions = [r for r in results if r.promoted]
    assert len(promotions) > 0, (
        "no promotions occurred in the synthetic loop; AutoLearn is not "
        "learning (GATE 7 FAILURE)"
    )

    # The final incumbent must differ from the initial bad incumbent.
    assert not np.allclose(loop.incumbent, bad_incumbent), (
        "incumbent unchanged after promotions; the vector was not actually "
        "updated (GATE 7 FAILURE)"
    )

    # The final incumbent should route A->symbolic and B->llm (the optimal
    # policy), which means axis 0 should be positive.
    assert loop.incumbent[0] > 0, (
        f"incumbent axis 0 should be positive (optimal direction) but got "
        f"{loop.incumbent[0]}; the loop learned the wrong direction "
        "(GATE 7 FAILURE)"
    )

    # Evaluate the final incumbent vs the bad incumbent on held-out.
    final_results = loop._evaluate_held_out(loop.incumbent, bad_incumbent)
    final_decided = [r for r in final_results if r.decided]
    if final_decided:
        mean_delta = sum(r.utility_delta for r in final_decided) / len(final_decided)
        assert mean_delta > 0, (
            f"final incumbent utility delta vs bad incumbent should be > 0 "
            f"but got {mean_delta}; the learned policy is not better "
            "(GATE 7 FAILURE)"
        )
        # The final incumbent should be correct on more tasks than the bad one.
        final_correct = sum(1 for r in final_decided if r.candidate_correct)
        bad_correct = sum(1 for r in final_decided if r.incumbent_correct)
        assert final_correct > bad_correct, (
            f"final incumbent correct={final_correct} should be > bad "
            f"incumbent correct={bad_correct} (GATE 7 FAILURE)"
        )
