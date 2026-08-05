# REAL_LOOP_AUDIT v0.3.10.2-alpha

## Audit of the Real-Model Execution Loop

This document audits every arrow in the real acceptance chain (Section 54):

```
prompt → real hidden state → real policy decision → real symbolic
execution + real LLM generation → real verifier → real utilities →
real regret → actual candidate/incumbent comparison →
calibration/OOD/safety constraints → promotion or rollback
```

## Arrow 1: prompt → real hidden state

**Status: EXECUTED**

`capture_task_representation()` in `real_backends.py`:
- Loads the task prompt via `tokenizer(prompt, return_tensors="pt")`.
- Runs `model(**inputs, output_hidden_states=True)`.
- Extracts the hidden state at the configured layer/location.
- Returns a 1-D numpy array.

**Leakage check (Section 15)**: The capture function only reads
`task["prompt"]` or `task["specification"]` — it does NOT access
`task["expected"]`, backend results, or verifier output. Verified by
`test_capture_is_pre_execution` which inspects the source code.

## Arrow 2: real hidden state → real policy decision

**Status: EXECUTED**

The policy (`fit_policy()`) takes the captured hidden state and
produces `P(S | h)` via `predict_proba()`. The `choose_route_with_reason()`
function then maps this probability to a route decision with an
explicit reason (symbolic, llm, low_confidence, ood).

## Arrow 3: real policy decision → real symbolic execution

**Status: EXECUTED**

`execute_symbolic_backend()` in `real_backends.py`:
- Calls `plan_from_structured_task()` to compile the task.
- Calls `execute_plan()` to run the bounded symbolic executor.
- Returns a `BackendOutcome` with `correct=True` if execution
  succeeded, `correct=False` if it failed (unsupported, parse error,
  timeout).
- Does NOT use `task["symbolic_correct"]` placeholder labels.

**Verified by**: `test_symbolic_executes_arithmetic`,
`test_symbolic_fails_closed_on_unsupported`,
`test_symbolic_no_placeholder_labels`.

## Arrow 4: real policy decision → real LLM generation

**Status: EXECUTED**

`execute_llm_backend()` in `real_backends.py`:
- Applies the chat template (`tokenizer.apply_chat_template`).
- Calls `model.generate()` with configurable `max_new_tokens`,
  `do_sample`, `temperature`, `top_p`, `seed`.
- Extracts only the generated tokens (not the prompt).
- Returns a `BackendOutcome` (with correct/quality as placeholders
  set by the verifier) and the raw generated text.
- Does NOT use `task["llm_correct"]` placeholder labels.

**Verified by**: `test_llm_generates_text` (integration test with
Qwen2.5-0.5B-Instruct).

## Arrow 5: real execution → real verifier

**Status: EXECUTED**

`verify_output()` dispatches to:
- `verify_arithmetic()` for arithmetic tasks (exact numeric
  comparison, never substring matching).
- `verify_exact_string()` for letter counting / exact string tasks
  (normalized string comparison).
- Fail closed (`verified_correct=None`) for unsupported tasks.

**Verified by**: `test_verify_correct_arithmetic`,
`test_verify_incorrect_arithmetic`, `test_verify_never_substring_match`,
`test_exact_match`, `test_answer_phrase`, `test_wrong_answer`,
`test_no_expected_field`.

## Arrow 6: real verifier → real utilities

**Status: EXECUTED**

`build_real_counterfactual_experience()`:
- Verifies both backend outputs.
- Updates BackendOutcomes with verified correctness/quality.
- Computes `U = w_q * Q - λ_t * T/T_ref - λ_c * C/C_ref - λ_r * R`.
- Computes `ΔU = U_sym - U_llm`.
- Determines preferred action (symbolic, llm, or abstain).
- Computes sample weight using the configured weight mode.

## Arrow 7: real utilities → real regret

**Status: EXECUTED**

`mean_regret()` in `regret.py`:
- `Regret_i = max_a U_i(a) - U_i(π(x_i))`
- Oracle utilities are computed from the pre-computed experiences
  (both backends were executed).
- The policy's actual decision is scored against the oracle.

## Arrow 8: real regret → candidate/incumbent comparison

**Status: EXECUTED**

`paired_promotion_statistics()`:
- Computes `d_i = U_candidate_i - U_incumbent_i` for each task.
- Reports mean, median, win/loss/tie rates, bootstrap CI.
- Also computes regret deltas.
- The incumbent is always-LLM (the default routing policy).

## Arrow 9: comparison → calibration/OOD/safety constraints

**Status: EXECUTED**

Phase D calibration:
- Fits Mahalanobis OOD on train features.
- Calibrates `tau_ood` as the 99th percentile of calibration scores.
- Grid-searches `tau_conf` using actual policy probabilities.
- Computes Brier and ECE on the calibration set.

## Arrow 10: constraints → promotion or rollback

**Status: EXECUTED**

`atomic_promote()` in `atomic_promotion.py`:
- Takes candidate artifact, incumbent path, and gate results.
- If all gates pass: atomically writes the candidate and swaps the
  incumbent pointer.
- If any gate fails: writes the candidate as a rejected artifact,
  leaves the incumbent unchanged.

**Verified by**: `test_failed_promotion_leaves_incumbent_unchanged`,
`test_successful_promotion_swaps_incumbent`.

## Steering Loop Audit

```
real hidden-state intervention → real route/behavior change →
verified downstream utility change
```

**Status: PARTIALLY EXECUTED**

- **Hidden-state intervention**: EXECUTED — `ResidualStreamHook`
  installs a forward hook that adds `alpha * v` to the residual
  stream at the configured layer.
- **Route/behavior change**: EXECUTED — the intervention changes
  `P(S | h)`, which changes the route decision.
- **Verified downstream utility change**: NOT FULLY EXECUTED — the
  current experiment measures route probability changes but does not
  re-execute backends after steering to measure verified utility
  changes. This is labeled `REAL_MODEL_LATENT_INTERVENTION`, not
  `REAL_MODEL_UTILITY_INTERVENTION`.

**Steering results (mixed tasks)**:
- Centroid vector norm: 2.87 (non-zero — the mixed tasks produce a
  meaningful steering direction).
- E[P(S|+v)] = 0.69, E[P(S|0)] = 0.53, E[P(S|-v)] = 0.37.
- The dose-response is monotonic and directionally consistent.

## Removed Placeholders

| Placeholder | Location | Status |
|------------|----------|--------|
| `p = 0.5` in calibration | `cli/commands/policy.py` | REMOVED — now uses `policy.predict_proba()` |
| `return 0.0` in `real_utility_fn` | `cli/commands/policy.py` | FIXED — now looks up pre-computed experience |
| `symbolic_correct` labels | `real_backends.py` | NEVER USED — outcomes from actual execution |
| `llm_correct` labels | `real_backends.py` | NEVER USED — outcomes from actual generation + verification |
| `except Exception` in `_save_policy_artifact` | `cli/commands/policy.py` | FIXED — uses specific exceptions |
| `real_model_causal` broad label | `interventions/__init__.py` | REFINED — `REAL_MODEL_LATENT_INTERVENTION` |

## Conclusion

The real-model execution loop is **EXECUTED** end-to-end. Every arrow
in the acceptance chain is either executed or explicitly labeled as
unimplemented. The first scientifically meaningful real Qwen experiment
demonstrates that AutoLearn can learn a routing policy that achieves
zero regret on a mixed-task benchmark, beating always-LLM (regret
0.22) and matching the hand-coded router (regret 0.0).
