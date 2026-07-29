# CHANGELOG v0.3.10.2-alpha

## Real-Model Loop Completion, Release-Gate Integrity, Calibration Repair,
## Verified Counterfactual Execution, and Scientific Qualification

Release date: 2026-07-28

## Summary

v0.3.10.2-alpha completes the real-model execution loop and produces the
first scientifically meaningful real Qwen experiment. The key change is
the introduction of **mixed task types** (arithmetic + letter counting)
that create a genuine routing decision where sometimes symbolic wins
and sometimes the LLM wins. This replaces the v0.3.10.1 experiment which
used only arithmetic tasks where symbolic always won, making the
routing decision trivial.

## Critical Fixes

### G10: True Superiority Gate (Section 2)
- **Old**: `reg_w <= reg_u + 0.01` (non-inferiority, not superiority)
- **New**: `mean(d_s) >= min_weighting_gain` AND `lower CI > 0` across
  10 seeds, where `d_s = regret_unweighted(s) - regret_weighted(s)`.
- The gate now runs 10 seeds and reports mean, std, and 95% CI.

### Mixed Task Types (Section 50)
- **Old**: Only arithmetic tasks → symbolic always wins → no routing
  decision → scientifically meaningless experiment.
- **New**: `make_mixed_tasks()` generates both arithmetic (symbolic wins)
  and letter counting (LLM wins) tasks. This creates tasks where
  `ΔU > 0` (route symbolic) AND `ΔU < 0` (route LLM), so the policy has
  a real decision to learn.
- Added `make_letter_counting_tasks()` and `make_mixed_tasks()`.
- Added `verify_exact_string()` verifier for non-arithmetic tasks.

### Calibration Repair (Section 8)
- **Old**: Calibration used `p = 0.5` placeholder probabilities, making
  the threshold grid search meaningless.
- **New**: Calibration loads the trained policy and computes
  `p_i = policy.predict_proba(h_i)` for actual policy probabilities.
  The `p=0.5` fallback now fails closed instead of warning.

### Real Utility Function Fix
- **Old**: `real_utility_fn` returned `0.0` for all routes (no learning
  signal).
- **New**: Looks up pre-computed experience by `task_id` and returns
  the appropriate backend's verified quality.

### Evidence Label Refinement (Section 32)
- **Old**: Broad `real_model_causal` label.
- **New**: Refined labels:
  - `REAL_MODEL_LATENT_INTERVENTION` — hidden state changed, policy
    score changed.
  - `REAL_MODEL_BEHAVIORAL_INTERVENTION` — actual routed behavior
    changed.
  - `REAL_MODEL_UTILITY_INTERVENTION` — verified downstream utility
    changed.
  - `REAL_MODEL_FINAL` — complete real-model final evaluation.

### Source Tree Hash (Section 24)
- Added `source_tree_sha256()` to `provenance.py`.
- Included in experiment artifact, policy artifact, and release gates.

### CalibrationArtifact (Section 29)
- Added `CalibrationArtifact` frozen dataclass with policy_id,
  policy_hash, calibration_dataset_sha256, confidence_threshold,
  ood_threshold, temperature_scale, and calibration_metrics.

## Real Experiment Results

Model: Qwen/Qwen2.5-0.5B-Instruct (MPS, layer 10)
Splits: train=60, dev=30, cal=30, final=60 (30 arithmetic + 30 counting)

### Phase A: Routing Signal
- sym_better=10, llm_better=8, tie=42
- ΔU range: [-1.0, 1.0], mean=0.033

### Phase F: Final Results

| Method | Utility | Regret |
|--------|---------|--------|
| Always LLM | 0.4833 | 0.2167 |
| Always symbolic | 0.5000 | 0.2000 |
| Hand router | 0.7000 | 0.0000 |
| Unweighted logistic | N/A | 0.2000 |
| Weighted centroid | N/A | 0.0000 |
| Logistic soft | N/A | 0.0000 |
| Logistic hard | N/A | 0.2000 |
| MLP | N/A | 0.2000 |

### Scientific Success Tiers
- Tier 2 (beats always-LLM): **YES** (cand=0.0 < llm=0.22)
- Tier 3 (beats hand router): NO (both 0.0)
- Tier 4 (beats unweighted): **YES** (cand=0.0 < unw=0.2)

### Phase E: Steering
- Centroid vector norm: 2.87 (non-zero!)
- E[P(S|+v)] = 0.69, E[P(S|0)] = 0.53, E[P(S|-v)] = 0.37
- Evidence level: REAL_MODEL_LATENT_INTERVENTION

## Release Gates

All 25 gates pass:
- G1-G9: Core functionality (target modes, weight modes, policy
  selection, alignment, calibration math, benchmarks)
- G10: Weighted truly beats unweighted (multi-seed, true superiority)
- G11-G15: Scientific tests (centroid failure, linear router, MLP,
  comparative gate, causal sanity)
- G16-G19: Real-model CLI (intervention, train, evaluate, calibrate)
- G20-G25: OOD, candidate-vs-incumbent, capability gate, atomic
  rollback, version provenance, smoke run

## Files Changed

- `src/daph_learning/execution/real_backends.py` — mixed tasks,
  exact_string verifier
- `src/daph_learning/policy/calibration.py` — CalibrationArtifact
- `src/daph_learning/policy/provenance.py` — source_tree_sha256
- `src/daph_learning/policy/config.py` — refined evidence levels
- `src/daph_learning/policy/__init__.py` — new exports
- `src/daph_learning/interventions/__init__.py` — refined evidence
  labels
- `src/daph_learning/cli/commands/policy.py` — calibration repair,
  utility function fix, fail-closed
- `scripts/run_real_qwen_experiment.py` — mixed tasks, all baselines
- `scripts/generate_v0310_1_gates.py` — multi-seed G10, version fix
- `tests/test_v0310_2_real_gates.py` — new tests for mixed tasks,
  verifier, CalibrationArtifact, source hash, evidence labels
- `release_gates.json` — regenerated with all 25 gates passing
- `experiment_results.json` — regenerated
- `artifacts/real_qwen_experiment_result.json` — new mixed-task
  results
