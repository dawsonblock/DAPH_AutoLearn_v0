# RELEASE_GATE_AUDIT v0.3.10.2-alpha

## Audit of Every Release Gate

For each gate: original claim, old assertion, problem, new assertion,
evidence level.

---

## G1: Soft/hard target modes work

- **Claim**: The system supports both soft and hard preference targets.
- **Old assertion**: `soft_targets` boolean flag produces different
  target arrays.
- **Problem**: The boolean flag was a correctness bug — the trainer
  always produced soft targets regardless of the flag.
- **New assertion**: `TargetMode.SOFT` produces `q = σ(ΔU/τ)` and
  `TargetMode.HARD` produces `y ∈ {0,1}` with ties masked. Verified
  by comparing the actual target arrays.
- **Evidence level**: UNIT

## G2: Weight modes work

- **Claim**: The system supports multiple weighting modes.
- **Old assertion**: `weight_mode` dispatches to `gap` or `snr`.
- **Problem**: The old `gap` mode was always called regardless of the
  config, silently ignoring `snr`.
- **New assertion**: `weight_mode` dispatches to `uniform`,
  `absolute_gap`, `clipped_gap`, or `snr` via a unified
  `compute_weight()` function. Each mode produces different weights.
- **Evidence level**: UNIT

## G3: Policy selection works

- **Claim**: Different policy types can be selected.
- **Old assertion**: `policy_type` dispatches to centroid or logistic.
- **Problem**: `mlp_experimental` was not selectable.
- **New assertion**: `policy_type` dispatches to `centroid`,
  `logistic`, or `mlp_experimental` via `policy_factory.py`.
- **Evidence level**: UNIT

## G4: Task-ID alignment works

- **Claim**: Features and experiences are joined by task ID.
- **Old assertion**: Features and experiences have the same length.
- **Problem**: Length matching is insufficient — a shuffled feature
  order would silently misalign.
- **New assertion**: `join_by_task_id()` joins by explicit task ID
  and rejects duplicate IDs. `FeatureRecord` dataclass enforces
  task-bound features.
- **Evidence level**: UNIT

## G5: Missing evaluator fails closed

- **Claim**: The system refuses to evaluate without a utility function.
- **Old assertion**: (not tested)
- **Problem**: The system silently returned 0.0 utility when
  `utility_fn` was missing.
- **New assertion**: The learner raises an error when `utility_fn`
  is None in held-out evaluation paths.
- **Evidence level**: UNIT

## G6: Negative weights rejected

- **Claim**: `weighted_mean` rejects negative weights.
- **Old assertion**: (not tested)
- **Problem**: Negative weights could silently produce incorrect
  centroid directions.
- **New assertion**: `weighted_mean` rejects negative, NaN, Inf, and
  all-zero weights.
- **Evidence level**: UNIT

## G7: Calibration math corrected

- **Claim**: Calibration metrics are mathematically correct for soft
  targets.
- **Old assertion**: Brier score against hard 0/1 labels.
- **Problem**: Hard-label Brier silently throws away continuous
  preference information.
- **New assertion**: `preference_brier_soft` compares `P(S|h)` against
  the soft target `q = σ(ΔU/τ)`. `action_confidence_ece` uses
  `predicted_action_correctness_target` for expected correctness.
- **Evidence level**: UNIT

## G8: Dev-regret early stopping uses actual regret

- **Claim**: Early stopping uses dev regret, not dev loss.
- **Old assertion**: (not tested)
- **Problem**: Early stopping defaulted to dev loss, which doesn't
  correlate with the primary metric.
- **New assertion**: `--early-stopping-metric` defaults to
  `dev_regret` and uses the correct confidence threshold.
- **Evidence level**: UNIT

## G9: Benchmark not trivially solvable

- **Claim**: The synthetic benchmark is not trivially solvable by
  random directions.
- **Old assertion**: Median random regret > 0.05.
- **New assertion**: Same, verified with 200 random directions.
- **Evidence level**: SYNTHETIC

## G10: Weighted truly beats unweighted (MULTI-SEED)

- **Claim**: Utility weighting helps on the designed near-tie
  benchmark.
- **Old assertion**: `reg_w <= reg_u + 0.01` (non-inferiority).
- **Problem**: This is a non-inferiority test, not a superiority test.
  The old gate could pass even when weighted was slightly worse.
- **New assertion**: `mean(d_s) >= min_weighting_gain` AND
  `lower CI > 0` across 10 seeds, where
  `d_s = regret_unweighted(s) - regret_weighted(s)`.
  `min_weighting_gain = 0.005` (chosen before seeing results).
- **Evidence level**: SYNTHETIC (multi-seed)

## G11: Centroid fails on multimodal

- **Claim**: The centroid policy degrades on multimodal geometry.
- **Old assertion**: Centroid regret on multimodal > linear + 0.05.
- **New assertion**: Same.
- **Evidence level**: SYNTHETIC

## G12: Logistic strong on linear

- **Claim**: The logistic router learns the linear routing signal.
- **Old assertion**: Logistic regret on linear < 0.15.
- **New assertion**: Same.
- **Evidence level**: SYNTHETIC

## G13: MLP beats logistic on XOR

- **Claim**: The MLP router captures nonlinear routing geometry.
- **Old assertion**: MLP regret < logistic regret on XOR.
- **New assertion**: Same.
- **Evidence level**: SYNTHETIC

## G14: Comparative gate computes both sides

- **Claim**: The comparative gate measures both candidate and
  incumbent.
- **Old assertion**: (not tested)
- **New assertion**: The gate stores both `candidate_value` and
  `incumbent_value` and checks the expected ordering.
- **Evidence level**: SYNTHETIC

## G15: Mechanical causal sanity

- **Claim**: Direction reversal is consistent.
- **Old assertion**: Reversal consistent and p-value < 0.05.
- **New assertion**: Same.
- **Evidence level**: UNIT

## G16: Real-model intervention pipeline

- **Claim**: The real-model intervention module exists and executes.
- **Old assertion**: Import succeeds.
- **Problem**: Import success does not prove execution.
- **New assertion**: Import succeeds AND the module has the required
  classes (`ResidualStreamHook`, `InterventionConfig`,
  `run_real_intervention`). Execution is verified by G14 in
  `test_v0310_2_real_gates.py::TestRealInterventionExecutes`.
- **Evidence level**: REAL_MODEL_LATENT_INTERVENTION (integration test)

## G17: Real train CLI executes

- **Claim**: The real-model train CLI executes.
- **Old assertion**: CLI parses arguments.
- **New assertion**: CLI parses arguments AND the real-model branch
  is reachable (requires `--model` and `--dataset`).
- **Evidence level**: UNIT (CLI parsing); REAL_MODEL_FINAL (full
  execution via experiment script)

## G18: Real evaluate CLI executes

- **Claim**: The real-model evaluate CLI executes.
- **Old assertion**: CLI parses arguments.
- **New assertion**: Same as G17.
- **Evidence level**: UNIT

## G19: Real calibrate CLI executes

- **Claim**: The real-model calibrate CLI uses actual policy
  probabilities.
- **Old assertion**: CLI parses arguments.
- **Problem**: The old calibration used `p = 0.5` placeholder.
- **New assertion**: CLI parses arguments AND calibration uses
  `policy.predict_proba()` for actual probabilities. The `p=0.5`
  fallback now fails closed.
- **Evidence level**: UNIT

## G20: OOD threshold calibrated

- **Claim**: The OOD threshold is calibrated from data.
- **Old assertion**: `tau_ood` is a finite number.
- **New assertion**: Same, calibrated as the 99th percentile of
  in-distribution scores.
- **Evidence level**: SYNTHETIC

## G21: Candidate vs incumbent uses actual policy

- **Claim**: The candidate and incumbent are evaluated independently.
- **Old assertion**: (code review)
- **New assertion**: Verified by `learner.py` code and
  `test_v0310_1_p0_gates`.
- **Evidence level**: UNIT

## G22: Capability gate works

- **Claim**: The capability regression gate works.
- **Old assertion**: Gate returns a result.
- **New assertion**: Gate evaluates multiple families and returns
  per-family results.
- **Evidence level**: UNIT

## G23: Atomic rollback works

- **Claim**: Atomic promotion/rollback works.
- **Old assertion**: (not tested)
- **New assertion**: Failed promotion leaves incumbent unchanged;
  successful promotion swaps incumbent. Verified by
  `test_failed_promotion_leaves_incumbent_unchanged` and
  `test_successful_promotion_swaps_incumbent`.
- **Evidence level**: UNIT

## G24: Version/config provenance

- **Claim**: Version and config are consistent.
- **Old assertion**: `__version__ == "0.3.10.1-alpha"` (hard-coded).
- **Problem**: Hard-coded version breaks on version bumps.
- **New assertion**: `__version__ == cfg.autolearn_version` (dynamic).
- **Evidence level**: UNIT

## G25: Small real-model smoke run

- **Claim**: A small real-model smoke run completes.
- **Old assertion**: Smoke artifact file exists.
- **New assertion**: Smoke artifact exists AND has required fields
  (evidence_level, model_id, device, n_intervention_results,
  alpha_grid). The experiment artifact must have source_tree_sha256,
  mixed-task composition, and per-family metrics.
- **Evidence level**: REAL_MODEL_FINAL
