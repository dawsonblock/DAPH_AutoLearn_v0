# CHANGELOG V0.3.9 — Causal Learning Loop Repair

## 1. Critical Defect: Real Candidate Policy Evaluation

**Problem**: The held-out promotion evaluation in `_evaluate_held_out` accepted
`candidate_vector` but did not use it to determine the candidate route. Instead,
the candidate route was derived from the oracle/counterfactual Delta-U target,
and the incumbent route was hard-coded to `"symbolic"`. This made promotion
invalid because the candidate's held-out decision did not depend on the
candidate vector at all.

**Root cause**: The original v0.3.9 counterfactual loop was model-optional and
used the Delta-U target as a proxy for the candidate route, bypassing the
routing policy entirely.

**Files changed**:
- `src/daph_learning/autolearn/trust_region.py` (rewritten `_evaluate_held_out`)
- `src/daph_learning/autolearn/route_policy.py` (new — routing policy contract)
- `src/daph_learning/autolearn/adapters.py` (new — model-backed adapters)

**Solution**: Introduced an explicit `RoutePolicyFn` contract that maps
`(tasks, steering_vector, config) -> [PolicyRouteDecision]`. The
`_evaluate_held_out` method now calls `route_fn` with the candidate vector for
candidate routing decisions and the incumbent vector for incumbent routing
decisions. Both backends are executed counterfactually for utility scoring,
but the utility counted for each policy is the utility of the backend THAT
POLICY SELECTED — not the oracle's preferred backend.

**Tests added**:
- `test_heldout_evaluation_is_sensitive_to_candidate_vector` (GATE 2)
- `test_heldout_evaluation_is_sensitive_to_incumbent_vector` (GATE 3)
- `test_oracle_leakage_candidate_selects_llm_but_oracle_prefers_symbolic` (GATE 4)

## 2. Routing Policy Contract

**Problem**: No explicit routing-policy interface existed; the trust-region
optimizer was coupled to the oracle Delta-U for routing decisions.

**Files changed**: `src/daph_learning/autolearn/route_policy.py` (new)

**Solution**: Created `SteeringPolicyConfig` (frozen runtime config with
layer, alpha, token_location, model/tokenizer revision, vector version),
`PolicyRouteDecision` (with full provenance), `SafetyClampTelemetry`, and
`RoutePolicyFn` type alias. Added `make_feature_route_policy` helper for
model-optional testing.

## 3. Trust-Region Bootstrap

**Problem**: Zero-vector initialization used norm-preservation, which kept the
vector at zero forever. No configurable bounds existed.

**Files changed**: `src/daph_learning/autolearn/trust_region.py`

**Solution**: Added `bootstrap_from_candidate` function that initializes from
the candidate direction with a configured `initial_steering_norm` when
`||v_incumbent|| < epsilon_norm`. Added `min_vector_norm`, `max_vector_norm`,
`max_update_norm` bounds. Fail-closed on NaN/Inf via `_validate_vector`.

**Tests added**: `test_trust_region_zero_vector_bootstrap`,
`test_trust_region_rejects_nan_candidate`, `test_trust_region_rejects_inf_candidate`,
`test_trust_region_excessive_candidate_norm_clipped`,
`test_trust_region_max_update_norm_clipping`, `test_trust_region_min_norm_enforced`.

## 4. Capture Alignment by Task ID

**Problem**: Activation capture returned positional arrays with no task IDs.
Shape mismatches could be silently repaired by truncating/slicing.

**Files changed**: `src/daph_learning/autolearn/experience.py`, `src/daph_learning/autolearn/trust_region.py`

**Solution**: Added `CapturedActivation` and `CaptureResult` structs with
per-task `task_id`, `activation`, `success`, `failure_reason`,
`activation_hash`. The loop's `_align_by_task_id` method joins activations
and utility weights by task_id. A compatibility adapter handles legacy
`(array, int, int)` capture returns.

**Tests added**: `test_capture_alignment_exactly_8_of_10_enter_update`,
`test_capture_alignment_weights_match_task_ids`,
`test_capture_alignment_min_coverage_gate`.

## 5. Hard-Coded Steering Lineage Removed

**Problem**: Lineage records used hard-coded `layer=24`,
`token_location="anchor"`, `alpha=1.0` regardless of actual runtime config.

**Files changed**: `src/daph_learning/autolearn/trust_region.py`

**Solution**: Lineage is now generated from `SteeringPolicyConfig` fields
(`spc.layer`, `spc.token_location`, `spc.alpha`).

## 6. Dataset Hash Lineage Fixed

**Problem**: Training and development datasets used the same hash.

**Files changed**: `src/daph_learning/autolearn/trust_region.py`

**Solution**: Added `training_dataset_sha256` and
`development_dataset_sha256` as separate constructor parameters. Lineage
records now carry both independently.

## 7. Versioning Consolidated

**Problem**: Version surfaces mixed 0.3.6, 0.3.8, 0.3.8.1, 0.3.9.

**Files changed**: `pyproject.toml`, `src/daph_learning/__init__.py`, `README.md`,
`CLAIMS.md`, `src/daph_learning/cli/commands/build_oracles.py`,
`tests/test_cli_entrypoints.py`, `tests/test_v037_exit_gate.py`

**Solution**: All version surfaces now agree on `0.3.9`.

## 8. CLI Integration

**Problem**: CLI used the legacy `run_autolearn_loop` only.

**Files changed**: `src/daph_learning/cli/commands/autolearn.py`

**Solution**: Added `--engine` flag with `counterfactual` (default) and
`legacy` (deprecated). The counterfactual engine wires model-backed adapters
(`make_model_route_policy`, `make_model_capture_fn`, `make_model_execute_fn`)
into the corrected `CounterfactualLearningLoop`.

## 9. Promotion Statistics and Capability Regression Gates

**Problem**: Promotion gate lacked min_sample_count and per-capability
regression gates.

**Files changed**: `src/daph_learning/autolearn/promotion.py`

**Solution**: Added `min_sample_count`, `capability_regression_thresholds`,
`protected_capabilities` to `PromotionGateConfig`. Added
`_check_capability_regressions` to `evaluate_promotion_gate`. Added
`candidate_route`, `incumbent_route`, `capability_id`, `task_family`,
`utility_delta` to `HeldOutTaskResult`.

## 10. Synthetic Closed-Loop Test

**Problem**: No end-to-end test proving AutoLearn actually learns.

**Files changed**: `tests/test_v039_causal_gates.py` (new)

**Solution**: `test_synthetic_closed_loop_autolearn_improves_over_incumbent`
builds a deterministic environment (family A -> symbolic, family B -> LLM),
starts from a deliberately imperfect incumbent, runs the loop, and verifies
the promoted vector differs from the incumbent and achieves higher utility.
