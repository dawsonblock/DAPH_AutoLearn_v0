# 0.3.10.6-alpha (Scientific repair + prompt interface fix + 1.5B experiment)

## What changed

### Scientific repair (Priority 0 + Priority 1 + Priority 2)

- **Matched sham control (P0)**: sham now permutes signed continuous ΔU and
  matching weights within strata — not binary labels — so `fit_policy` applies
  the identical `sigmoid(ΔU/τ)` transform the real P1 model received.
- **Ablation pipeline parity (P0)**: all ablation policies now traverse the
  same calibration + frozen thresholds + abstention pipeline as P1.
- **Real pooling (P0)**: `last_prompt_token`, `mean_prompt_tokens`, and
  `mean_content_tokens` are independently computed (previously identical).
- **Proper negative controls (P0)**: replaced invalid square random-projection
  with orthogonal-rotation invariance, dimension-reduced JL projection,
  Gaussian noise, coordinate permutation, and subtype-stratified shuffled-hidden.
- **Subtype-only dev freeze (P0)**: subtype-only baseline frozen on development
  data, never derived from final labels.
- **Artifact integrity (P1)**: failed runs moved to `gate_a_failed/`; stale
  PASS moved to `gate_a_historical/`; final stage writes to `gate_a_runs/`
  (staging) — only promotion logic moves to `gate_a_qualified/`.
- **Version normalization (P2)**: all version surfaces agree on 0.3.10.6-alpha.
  Removed stale `CURRENT_EXPERIMENT_ID` from package source.

### Prompt interface fix + 1.5B experiment

- **Prompt interface fix**: Created shared `build_llm_prompt()` in
  `real_backends.py` as the single source of truth for LLM prompt
  construction. All scripts now use this function instead of
  constructing prompts independently.
- **Fixed FINAL: → FINAL_ANSWER: format**: 5 files were using the wrong
  `FINAL:` format instead of the canonical `FINAL_ANSWER:` format expected
  by the verifier. This was the root cause of 0% LLM accuracy in the
  original 1.5B run.
- **Backward-compatible parser**: `parse_final_or_exact()` now accepts
  both `FINAL:` and `FINAL_ANSWER:` formats for backward compatibility.
- **New subtypes G and H**: Added unit conversion (G) and number
  theory/GCD-LCM (H) subtypes to the crossover benchmark, bringing the
  total from 6 to 8 subtypes. Group count increased from 420 to 560.
- **1.5B experiment (daph_gate_a_real_004_1b5)**: Confirmed that the
  prompt interface was the bottleneck, not model size. The 1.5B model
  passes Gate A with 27.5% LLM accuracy (up from 0%) and 81.3% policy
  utility. See `PROMPT_INTERFACE_FINDINGS.md`.
- **New tests**: Added `test_prompt_interface.py` with 13 tests
  verifying the FINAL_ANSWER format interface.

## Gate A results

| Model | P1 | P0 | Gain | Oracle capture | Verdict |
|-------|-----|-----|------|----------------|---------|
| Qwen2.5-7B-Instruct | 0.946 | 0.573 | 0.374 | 1.000 | PASS |
| Qwen2.5-1.5B-Instruct | 0.813 | 0.275 | 0.537 | 0.997 | PASS |

---

# 0.3.10.5-alpha (Gate A statistical correctness repair)

This is a **scientific-correctness repair** of the Gate A qualification layer.
The prior Gate A PASS (daph_gate_a_real_002) is **INVALIDATED** because:

1. The primary confidence interval was a placeholder (`point_estimate ± 0.1`).
2. The P1-minus-sham interval was actually the sham utility interval.
3. P1 utility was based on policy probabilities, not selected routing actions.
4. Positive-group fraction measured symbolic preference, not P1 improvement.
5. The final policy was retrained instead of loaded from the frozen artifact.
6. Calibration was frozen but not operationally applied.
7. Several freeze checks compared manifest values against themselves.

## What changed

- **Invalidated prior PASS**: `daph_gate_a_real_002` moved to
  `artifacts/invalidated/daph_gate_a_real_002_invalid_statistics/`
- **QualificationStatus enum**: PASS / FAIL / NOT_EVALUABLE / INVALIDATED
- **Hard routing**: P1 utility uses selected actions, not soft probability
- **Real group bootstrap**: 20,000 iterations with group-weighted estimand
- **P1-minus-sham**: Nested bootstrap of the actual difference
- **Frozen policy evaluation**: Final stage loads, hashes, and executes
  the frozen policy artifact — no retraining
- **Operational calibration**: Frozen thresholds applied to raw probabilities
- **Precondition gates**: Checked before statistical gates
- **Explicit comparators**: gt / gte / lt / lte / eq
- **Prediction artifacts**: final_predictions, final_task_metrics, sham_predictions
- **Independent validator**: Recomputes all metrics from task-level records
- **Portable pointer**: Relative paths only, no machine-local paths

## What was NOT changed

- No new cognitive architecture, latent memory, activation steering,
  model merging, recurrent reasoning, or additional agent frameworks.
- The real-model pipeline, benchmark, canonical verification,
  hidden-state extraction, and artifact infrastructure are unchanged.

---

# 0.3.10.4-alpha (Gate A scientific-integrity repair — Priority 0)

This is a **scientific-integrity, reproducibility, and qualification repair**,
not an architecture expansion. It is the first slice (Priority 0, Sections 1–7
of the repair plan) of requalifying Gate A under a new frozen experiment ID.

## What changed

- **Stale artifact discovery**: `artifacts/current/` previously contained
  synthetic evidence and manifests carrying the historical source hash
  `fd4d47e3...` (v0.3.10.3.2-alpha), which did not match the current source
  tree. The repo's reports, manifests, and experiment artifacts did not
  describe the same repository state. This ambiguity is eliminated.
- **Source-hash mismatch fixed**: a single canonical
  `compute_canonical_source_hash()` with explicit include/exclude globs,
  deterministic ordering, normalized POSIX paths, and **normalized line
  endings** (CRLF/CR → LF) is now the only accepted implementation. CLI:
  `python -m daph_learning.provenance source-hash [--json]`.
- **Old failed run archived**: the v0.3.10.3.2-alpha real-model Gate A run
  (Qwen2.5-1.5B-Instruct, point estimate +0.193, **LCB95% = −0.041 → FAIL**)
  is preserved as legacy evidence at
  `artifacts/legacy/daph_gate_a_real_001_failed/` with `LEGACY_NOTICE.md`
  and `historical_source_manifest.json`. It is NOT presented as current.
  Missing historical evidence (`raw_metrics.json`, `bootstrap_results.json`,
  `environment.json`) is recorded honestly, not fabricated.
- **New experiment identity**: version bumped to `0.3.10.4-alpha`; new
  experiment ID `daph_gate_a_real_002`; old run retained as
  `daph_gate_a_real_001_failed`. The new experiment changes material
  variables and is therefore a new experiment, not a rerun.
- **Verifier hardening**: canonical `FINAL_ANSWER: <integer>` parser
  (`parse_canonical_integer_answer`) with closed statuses
  VALID/MISSING/MULTIPLE/MALFORMED/CONTRADICTORY and a closed
  `VerificationStatus` enum (CORRECT/INCORRECT/UNVERIFIABLE/EXECUTION_ERROR/
  TIMEOUT). Qualification fails closed on ambiguity; legacy permissive
  extraction can no longer award qualification credit.
- **eval() removal**: Python `eval()` was removed from the symbolic
  execution path in `real_backends.py`. All arithmetic now routes through
  the bounded AST evaluator `safe_eval_int_expr`, extended to the spec'd
  signature (`max_ast_nodes`, `max_depth`, `max_integer_bits`,
  `max_exponent`) with explicitly enumerated permitted nodes and rejection
  of names, calls, attributes, floats, true division, exponentiation,
  containers, and lambdas.
- **Artifact layout**: new `artifacts/` tree
  (`synthetic_ci/`, `real_model_smoke/`, `gate_a_qualified/`,
  `gate_a_failed/`, `legacy/`, `current/pointer.json`). `current/` holds
  only a pointer; the pointer currently declares
  `NOT_YET_REQUALIFIED` because no new full real-model Gate A run has been
  executed.
- **Artifact validator**: `validate_artifact_bundle()` recursively checks
  source-hash consistency, experiment/run ID consistency, evidence-level
  consistency, dataset/model/tokenizer/utility/policy hash consistency,
  split validity, final-access ledger presence, and rejects synthetic-as-
  qualified, failed-as-promoted, and cross-run metric copying. Legacy
  bundles are exempt from the current-hash requirement but cannot be
  presented as current.
- **CI gate**: `tests/test_artifact_integrity.py` enforces the layout and
  validation rules; `tests/test_symbolic_safety_section6.py` statically
  scans symbolic paths for `eval(`/`exec(`/`compile(`;
  `tests/test_canonical_verifier_section7.py` covers the verifier.

## What did NOT change (out of scope for Priority 0)

Priority 1+ items from the repair plan are NOT in this slice: frozen
`UtilityConfig`, benchmark crossover redesign, dataset leakage audit
module, latency protocol, representation-selection study, structured-
feature baselines, uncertainty-aware targets, multi-seed sham control,
final-access state machine, group-aware bootstrap statistics, frozen gate
criteria config, report generator, real-model smoke, and full Gate A
execution. These remain as documented next steps
(see `REMAINING_EXPERIMENT_STEPS.md`).

## Gate A status

`NOT_YET_REQUALIFIED`. No new full real-model Gate A experiment has been
run for `daph_gate_a_real_002`. No synthetic artifact is presented as
qualification evidence.

---

# 0.3.10.3.2-alpha (qualification integrity: within-subtype crossover, frozen real evaluation, canonical provenance, real steering utility validation)

- **Mission**: prove or falsify that AutoLearn can choose the better
  computation between two available backends for individual tasks that
  share the same family and subtype. No taxonomy shortcut should
  determine the answer.
- **Within-subtype crossover**: at least 3 subtypes must individually
  contain both symbolic-preferred and LLM-preferred examples, with the
  optimal backend emerging from actual executed utilities.
- **Canonical source-tree hash**: one `compute_source_tree_sha256()`
  implementation; all components delegate to it. Full 64-char hash in
  artifacts.
- **Artifact integrity**: current artifacts must match current source
  tree; stale artifacts archived.
- **32-gate framework executed**: G01-G32 actually run against the
  current source tree.
- **Real CLI paths completed**: evaluate, calibrate, intervene use real
  frozen pipelines with zero fitting on final.
- **Real steering utility**: ΔU(α) measured via executed backend
  utility, not symbolic probability.

# 0.3.10.3.1-alpha (qualification repair: within-family crossover benchmark, steering utility validation, frozen evaluation, source-hash enforcement)

- **Mission shift**: implementation quality > qualification quality. This
  release makes the evidence trustworthy rather than expanding architecture.
  No GDN2, COCONUT, model merging, SAE, PPO/GRPO, or multi-agent systems.
- **Section 1 — version unification**: all active surfaces bumped to
  `0.3.10.3.1-alpha` (pyproject, `__version__`, `ExperimentConfig`,
  `ProvenanceRecord`, CLI `--version`, README, CLAIMS). Added
  `test_version_consistency` (G1).
- **Section 2 — centroid zero-weight fail-closed**: `ZeroWeightPolicy` enum
  (`ERROR` default, `UNWEIGHTED_FALLBACK` opt-in). `InsufficientEffectiveWeight`
  raised by default instead of silently substituting the unweighted
  estimator. Fallback records provenance and reports
  `weighted_centroid_with_unweighted_fallback`.
- **Section 3 — canonical utility**: `backend_utility` is the single
  implementation of `U_b`; `utility_for_route` centralizes route-utility
  lookup. `test_all_real_scoring_paths_use_backend_utility` (G7).
- **Section 4-5 — outcome contract + confidence**: `BackendOutcome.verifier_status`
  tightened; `execution_success != verified_correct` invariant enforced.
  Confidence kept separate from quality; paired confidence combine
  (product/min/geometric_mean, configurable).
- **Section 6-7 — true weighted vs unweighted ablation**: comparisons hold
  the policy family fixed; uniform and weighted models receive different
  weight arrays. Per-policy weight diagnostics (ESS, min/max/mean, zero
  fraction) recorded.
- **Section 8-9 — dedup + group stats**: within-split prompt dedup
  (`test_no_within_split_prompt_duplicates`); `group_id`/`template_id`/
  `family_id` metadata; `grouped_bootstrap_mean_delta` for honest CI when
  tasks share a generator.
- **Section 10-12 — within-family crossover benchmark** (highest priority):
  structured + natural-language mathematics family with subtypes A-F where
  both symbolic and LLM can win on different instances inside the same
  family. Optimal backend never encoded in task metadata; derived only
  after both backends execute and verify. Instance-level routing test
  controls for family (G26).
- **Section 13 — strong hand router**: rule-based router using only
  decision-time information (structured expression → symbolic, unsupported
  syntax → LLM, NL extraction → LLM, exact modular → symbolic).
- **Section 14-15 — stage access control**: `ExperimentStage` enum;
  final split inaccessible before `FROZEN`; final-test access ledger.
- **Section 16-19 — real CLIs**: `daph-evaluate-routes` / calibrate /
  intervene load frozen artifacts, capture real hidden states, perform zero
  fitting on final. `CalibrationArtifact` dataclass with full hashes.
- **Section 20 — fail closed**: missing utility/verifier/policy/executor
  raise instead of returning 0.0.
- **Section 21-25 — steering optimizes utility** (second priority):
  objective is `argmax_alpha E[U(pi(h+alpha v), x)]`, not
  `max_alpha P(symbolic)`. Per-alpha verified utility, regret, route flips
  (beneficial/harmful/neutral), neutral KL. Oracle alpha probe on DEV only.
  Random direction controls with `p_emp` significance.
- **Section 26 — KL release gate**: `steering/promotion KL guard` exercises
  utility-up/KL-exceeds reject, utility-up/KL-below allow, KL-unavailable
  fail-closed.
- **Section 27 — verifier naming**: `verify_exact_string` tightened to true
  `strip() == strip()`; permissive parsing renamed
  `verify_constrained_answer`. Explicit modes EXACT/FINAL_MARKER/
  NUMERIC_EXTRACTOR/TOKEN_EXTRACTOR recorded per result.
- **Section 28-31 — representation ablation, policy class comparison,
  tie-aware metrics, ESS**: decisive vs tie-aware accuracy; ESS per class.
- **Section 32-35 — artifact discipline**: `artifacts/current` vs
  `artifacts/archive/<version>/<source_hash>`; current artifacts must share
  `source_tree_sha256`; test collection hash; re-run test report on current
  tree.
- **Section 36 — CLI doc validation**: documented flags validated against
  argparse; stale `--soft-targets`/`--weight-mode gap` removed.
- **Section 37-40 — release gates**: G1-G32 with claim/evidence/
  measurement/assertion contract. G17-G19 real-model gates execute on
  Qwen2.5-1.5B-Instruct or explicitly SKIP.
- **Section 41-48 — scientific success tiers + deliverables**: Tier 0-7
  separation; negative results (weighting did not help, steering hurt
  utility) are acceptable and supported.

# 0.3.10.3-alpha (P0 repair: verified output, honest gates, disjoint splits, canonical utility)

- **P0-1/P0-2**: `BackendOutcome` expanded with `available`, `executed`,
  `execution_success`, `output_text`, `output_hash`, `error_type`,
  `error_message`. `execute_symbolic_backend` now returns the actual
  output text alongside the outcome. Correctness is set by the verifier,
  not by execution success.
- **P0-3**: Confidence semantics fixed — unsupported backend has
  `verifier_confidence=1.0` (we are certain it cannot handle the task),
  not 0.0.
- **P0-4**: Removed silent `np.ones_like` fallback in `CentroidPolicy`
  when class weights sum to zero. Now raises explicitly. Added
  `weight_fallback` flag to the dataclass for auditability.
- **P0-5**: Weighted-vs-unweighted ablation now actually uses `w=1`
  (uniform weights), not the weighted train weights.
- **P0-6**: Calibration targets now use ΔU (via sigmoid), not the
  model's own predicted probabilities.
- **P0-7**: Word pool partitioned across splits via
  `partition_word_pool()`. `assert_splits_disjoint()` verifies no
  cross-split prompt leakage.
- **P0-8**: Final set is NOT executed until after all configuration is
  frozen (Phase F runs AFTER Phase E, not before).
- **P0-9**: `max_vector_norm` now only shrinks (if norm > limit), never
  expands.
- **P0-10**: One canonical `backend_utility()` function in
  `policy/utility.py`. Both `learner.py` and `real_backends.py` import
  it instead of duplicating the formula.
- **P0-11**: Steering utility uses verified backend utility, not
  synthetic route=SYMBOLIC → 1.0.
- **P0-12**: G16 now actually runs the intervention pipeline (not just
  imports). G21 actually tests candidate routes come from policy. G23
  actually tests atomic_promote + rollback_incumbent. G25 actually runs
  the smoke script.
- **P0-14**: Routing accuracy is now tie-aware (ties count as correct
  for either route) and reports decisive-task accuracy separately.

# 0.3.10.2-alpha (real-model loop completion, release-gate integrity, calibration repair, verified counterfactual execution, and scientific qualification)

- **Mission shift (Section 0)**: this release completes the real-model
  execution loop and fixes dishonest/weak release gates. The target is
  the first scientifically meaningful real Qwen result: does AutoLearn
  improve verified real-model decision utility?
- **G10 fix (Section 2)**: gate renamed from "weighted beats unweighted"
  (non-inferiority, `<= + 0.01`) to a true superiority gate
  (`regret_unweighted - regret_weighted >= min_weighting_gain`).
  Multi-seed qualification (10 seeds, mean(d) > min_gain, lower CI > 0).
- **Near-tie env redesign (Section 3)**: decisive examples (25%) carry
  the true routing signal aligned with `w*`; ambiguous examples (75%)
  contain high-variance noise + a nuisance direction. Weighted training
  now has a real expected advantage over unweighted.
- **Real backends (Sections 9-12)**: `execute_symbolic_backend` (bounded
  executor) and `execute_llm_backend` (actual model generation with
  `do_sample=False` for determinism). No more `symbolic_correct` /
  `llm_correct` placeholder labels in the qualified real path.
- **Real verifier (Section 12)**: arithmetic exact numeric verification;
  fail closed on unsupported tasks. Never substring matching.
- **Capture once (Section 14)**: activation captured once per task
  before backend execution, not after. Router state is pre-execution.
- **Calibration repair (Section 8)**: calibration CLI now uses
  `policy.predict_proba(h_i)` instead of `p = 0.5` placeholder.
- **Gate semantics fix (Sections 5-7, 21-23)**: G16 now actually
  executes a real intervention; G17/G18 exercise the real-model branch;
  G19 uses actual policy probabilities; G22 exercises KL logic; G23
  tests actual atomic rollback; G25 validates artifact schema.
- **Evidence labels (Section 32)**: `real_model_causal` replaced by
  `REAL_MODEL_LATENT_INTERVENTION`, `REAL_MODEL_BEHAVIORAL_INTERVENTION`,
  `REAL_MODEL_UTILITY_INTERVENTION`.
- **Real steering utility test (Sections 33-35)**: dose-response with
  actual verified utility; route-flip analysis (beneficial vs harmful);
  matched random direction controls with `p_emp`.
- **Source-tree hash (Section 24)**: deterministic hash over relevant
  repo files stored in all artifacts.
- **No placeholder success (Section 44)**: all `placeholder`, `TODO`,
  `dummy`, `p = 0.5`, `return 0.0` in scientific paths audited and
  removed from qualified mode.
- **Qualified vs demo mode (Section 45)**: `--mode real` rejects
  demo-only shortcuts; `--mode synthetic` preserved for ablation.

# 0.3.10.1-alpha (correctness repair, scientific hardening, benchmark redesign, real policy evaluation, calibration, OOD, and causal validation)

- **Mission shift (Section 0)**: this release is a focused repair and
  scientific-hardening pass. The release must answer whether AutoLearn
  learns a routing policy that reduces held-out regret and improves
  held-out utility on non-trivial tasks — not merely whether pytest
  passes, a logistic model trains, or a toy steering function moves
  logits. The system must be capable of falsifying its own assumptions.
- **P0-1 / G1 — `soft_targets` correctness bug fixed (Section 1)**:
  the `soft_targets: bool` flag was a *correctness bug*: the logistic
  trainer always produced `q_i = sigmoid(ΔU_i / τ)` regardless of the
  flag. Replaced by an explicit `TargetMode` enum (`SOFT` | `HARD`) and
  `build_preference_targets(delta_u, mode, temperature, gap_threshold)`
  which returns `(targets, valid_mask)`. Hard-mode ties
  (`|ΔU| <= gap_threshold`) are *ignored* via the mask, not coerced to
  0.5. The loss applies the mask.
- **P0-2 / G2 — `weight_mode` correctness bug fixed (Section 2)**: the
  config exposed `weight_mode: "gap" | "snr"` but experience
  construction always called the gap function, silently ignoring
  `"snr"`. Replaced by a four-mode `WeightMode` enum
  (`UNIFORM`, `ABSOLUTE_GAP`, `CLIPPED_GAP`, `SNR`) and a unified
  `compute_weight(...)` that actually dispatches on mode. SNR requires
  `sigma_delta_u` and fails closed if absent. Clean break: old `"gap"`
  string raises `ValueError` at config construction.
- **P0-3 / G3 — `policy_type` correctness bug fixed (Section 3)**: the
  config accepted multiple policy types but the integrated learner
  always trained logistic routing. Added a `PolicyModel` protocol and
  real `centroid` / `logistic` / `mlp_experimental` implementations.
  The CLI now instantiates different implementations based on
  `--policy`.
- **P0-4 / G4 — fail closed on missing `utility_fn` (Section 4)**:
  held-out evaluation no longer defaults to zero utility when
  `utility_fn` is absent. `dev_tasks is not None and utility_fn is None`
  raises `ValueError`. All scorer/verifier/utility callbacks fail
  closed.
- **P0-5 / G5 — strict task-ID alignment (Sections 5, 6)**: naked row
  arrays at cross-module boundaries replaced with task-bound
  `FeatureRecord` objects. New `join_by_task_id(experiences,
  feature_records)` asserts no duplicate task_ids in either input,
  reports missing feature records explicitly (no silent truncation),
  and is order-independent. `assert_unique_task_ids` helper provided.
- **P0-6 — silent zip truncation audit (Section 6)**: every
  `zip(experiences, activations)` and `zip(dev_tasks, dev_experiences)`
  in AutoLearn code audited; replaced with task-ID joins or explicit
  length assertions.
- **P0-7 / G6 — `weighted_mean` math validation (Section 7)**:
  `weighted_mean` now rejects negative weights, NaN, Inf, all-zero
  effective weights, and non-finite activations.
- **P0-8 / G7 — calibration / ECE math fix (Section 8)**: the old ECE
  used `confidence = max(p, 1-p)` with `accuracy = y` for soft labels,
  which is mathematically wrong. Added two clearly named metrics:
  `preference_brier_soft` (compares `p_i` directly against `q_i`) and
  `action_confidence_ece` (compares `c_hat_i = max(p, 1-p)` against
  `c*_i = predicted_action_correctness_target(p, q)`). For hard labels,
  `c*_i` reduces to ordinary 0/1 correctness.
- **P0-9 / G8 — dev regret early stopping (Section 9)**: the trainer
  used dev BCE loss even when the config said `dev_regret`. Early
  stopping is now selectable (`dev_loss` | `dev_regret` |
  `dev_utility`), default `dev_regret`. The `dev_regret` path executes
  the actual utility function; `dev_loss` remains available for
  ablation.
- **P10 / G9-G13 — synthetic benchmark redesign (Section 10)**: the
  old benchmark was too easy (one coordinate almost directly encoded
  class membership; a matched random direction achieved perfect
  regret). Replaced with four mathematical environments: linear,
  near-tie/heteroskedastic, multimodal/covariance, nonlinear XOR, plus
  a random-direction control. Each environment is designed so a
  specific method can fail.
- **P11 / G14 — comparative gates literally compare methods
  (Section 11)**: the G4 gate claimed `logistic >= centroid` but only
  checked logistic accuracy. Rewritten to compute both sides and
  assert `logistic_result.mean_regret <= centroid_result.mean_regret +
  tolerance`. Both measurements stored in the gate artifact.
- **P12 / G15 — distinguish toy causal test from real evidence
  (Section 12)**: the mechanical `+v / 0 / -v` sanity test is now
  labeled `evidence_level: "unit_sanity"`, not "evidence that AutoLearn
  discovered causal transformer steering". New evidence-level field:
  `unit_sanity` | `synthetic_causal` | `real_model_causal`.
- **P13-15 / G16 — real intervention pipeline (Sections 13-15)**:
  added residual-stream hook installation, baseline hidden-state
  capture, alpha grid `{-A, -A/2, 0, A/2, A}` dose-response, and
  `RealInterventionResult` records with route/utility/KL/clamp
  telemetry.
- **P16-19 / G17-G20 — real CLI: train / evaluate / calibrate
  (Sections 16-19)**: `daph-autolearn train` loads a real model,
  executes both backends, verifies outcomes, computes utilities,
  captures hidden features, joins by `task_id`, fits the policy, saves
  the artifact. `evaluate` and `calibrate` mirror this. OOD threshold
  is now calibrated (quantile-based `tau_ood`), not infinite by
  default in qualified runs.
- **P20 — PCA safety (Section 20)**: PCA fitted on TRAIN only; artifact
  records `n_components`, explained variance, and train dataset hash.
  Dimensionality tuned on dev regret only.
- **P21 — policy calibration + abstention at inference (Section 21)**:
  `choose_route_with_reason` records abstention reason codes
  (`low_confidence`, `ood`, `policy_tie`, `execution_failure`,
  `safety_gate`). No opaque single-category abstention.
- **P22 / G21 — promotion gate uses real policy behavior (Section 22)**:
  candidate_action = `candidate_policy(h_i)`, incumbent_action =
  `incumbent_policy(h_i)`. Oracle is only for scoring regret, never for
  choosing the candidate action.
- **P23 — paired statistics (Section 23)**: mean/median utility delta,
  mean regret delta, win/loss/tie rate, 10,000-draw bootstrap CI by
  task.
- **P24 / G22 — capability regression gates (Section 24)**:
  `CapabilityGateConfig` rejects candidates that regress beyond
  per-family utility/accuracy thresholds.
- **P25 / G22 — neutral KL gate (Section 25)**: mean / p95 neutral KL
  with prompt-suite hash; promotion requires `mean_KL <= threshold`.
- **P26 — random direction qualification (Section 26)**: `N_random >= 500`
  matched random directions; `p_emp = (1 + count(M_j >= M*)) / (N+1)`.
  No Gaussian z-score as primary significance evidence.
- **P27 — prioritized replay integration (Section 27)**: optional
  `prioritized_replay_alpha ∈ [0, 1]`; `p_i = |error| * |ΔU|` or
  `regret_i`; normalized sampling `P(i) = p_i^α / Σ p_j^α`. Ablate
  uniform vs prioritized.
- **P28 — bandit logging (Section 28)**: `PolicyDecision` dataclass
  logs `task_id`, `available_actions`, `action`, `probabilities`,
  `chosen_propensity`, `policy_version` on every decision. No
  doubly-robust learning claim unless actually used.
- **P29 — low-rank controller stays experimental (Section 29)**:
  clearly labeled `experimental`, not default, only enabled after
  centroid + logistic baselines validated.
- **P30 / G13 — small MLP router (Section 30)**: `SmallMLPRouter`
  (`Linear → GELU → Linear`) trained with the same targets / weights /
  dev regret / calibration path as the logistic router. Diagnostic: if
  `MLP >> logistic`, routing geometry is nonlinear.
- **P31-36 — synthetic result matrix + 5 scientific tests (Sections
  31-36)**: per-environment result tables; weighting-value test
  (weighted < unweighted regret on near-tie), centroid-failure test
  (multimodal geometry), linear-router-value test, nonlinear-router-
  value test (MLP > logistic on XOR), random-steering test (learned >
  median random).
- **P37-39 / G25 — real-model smoke + baseline matrix + final-test
  discipline (Sections 37-39)**: Qwen2.5-1.5B-Instruct smoke split
  (100/50/50/100) and research split (2000/500/500/1000). Baselines
  A-K (always-LLM, always-symbolic, hand router, unweighted/weighted
  centroid, soft/hard logistic, MLP, random direction, incumbent,
  candidate). Final-test access ledger.
- **P40-41 — real intervention study + evidence categories (Sections
  40-41)**: `{-A, -A/2, 0, A/2, A}` dose-response on dev; frozen best
  setting repeated once on final. Every reported result tagged
  `UNIT` | `SYNTHETIC` | `REAL_MODEL_DEV` | `REAL_MODEL_FINAL`.
- **P42 — test report honesty (Section 42)**: `TEST_REPORT` distinguishes
  executed / copied / skipped / env-failure / algo-failure. No stored
  historical pytest output presented as current execution.
- **P43 / G24 — version consistency (Section 43)**: `0.3.10.1-alpha`
  unified across `pyproject.toml`, `daph_learning.__version__`, CLI,
  README, CHANGELOG, manifest schema, generated artifacts. Automated
  consistency test.
- **P44 — config field audit (Section 44)**: every config field and CLI
  option classified `USED` / `DEPRECATED` / `EXPERIMENTAL` /
  `UNUSED_BUG`. No CLI flag silently does nothing; unsupported
  combinations fail at startup.
- **P45 — provenance (Section 45)**: policy artifact contains release
  version, git/source tree hash, model ID + revision, tokenizer
  revision, dataset/split hashes, policy type, target mode, target
  temperature, weight mode, gap threshold, confidence formulation,
  feature transform, PCA artifact hash, OOD model hash, selected
  layer, steering alpha, random seed, train timestamp, environment
  metadata. No hard-coded lineage.
- **P46 / G23 — atomic policy promotion (Section 46)**: write candidate
  → evaluate → pass all gates → atomically update incumbent pointer.
  Any gate failure leaves incumbent unchanged. No partially-written
  promoted state.
- **P47 — release gates G1-G25 (Section 47)**: the release is blocked
  unless all applicable gates pass. Gates are computed, not claimed.
- **Out of scope (Section 48)**: no GDN2, COCONUT, model merging, TIES,
  DARE, full SAE training, large PPO/GRPO, multi-agent debate, new
  databases, or major repo-wide rewrite. The bottleneck is evidence,
  not architecture.
- **Final scientific standard (Section 50)**: the release answers six
  separate questions (weighting value, centroid geometry sufficiency,
  linear router sufficiency, causal steering, intervention utility,
  AutoLearn vs simpler alternatives) without blurring them together.
  AutoLearn must be able to prove its preferred method wrong.

# 0.3.10 (counterfactual utility learning, weighted latent routing, causal intervention validation, uncertainty and regret optimization)

- **Architecture shift**: upgraded AutoLearn from a "weighted steering-vector
  extractor" into a counterfactual compute-selection learner. The weighted
  centroid remains as a transparent baseline; the primary policy learner is
  now a calibrated contextual policy model (weighted soft-target logistic
  router) operating over LLM hidden-state features.
- **Mathematical objective (Section 1)**: `U_i(a) = w_q·Q_i(a) - λ_t·T_i(a)/T_ref
  - λ_c·C_i(a)/C_ref - λ_r·R_i(a)`. The continuous utility difference `ΔU =
  U(S) - U(L)` is retained as supervision rather than discarded into a hard
  class label.
- **Primary metric: regret (Section 2)**: `Regret_i = max_a U_i(a) -
  U_i(π(x_i))`. Routing accuracy is secondary; a policy can have high
  accuracy but make mistakes on high-utility-penalty tasks.
- **Weighted centroid baseline (Section 3)**: `w_i = clip(|ΔU_i|·c_i, w_min,
  w_max)` with gap-threshold tie truncation (near-ties get zero weight, not a
  positive floor). Added `WeightConfig`, `utility_weight`,
  `weighted_contrastive_mean`, `unweighted_contrastive_mean`.
- **Theoretical correction (Section 4)**: utility importance is separated
  from latent decision geometry. The weighted centroid is compared against an
  actual learned decision boundary.
- **Weighted soft-target logistic router (Section 5)**: `p = P(S|h) = σ(w^T h
  + b)`, soft target `q = σ(ΔU/τ)`, weighted BCE loss. Added
  `WeightedLogisticRouter`, `soft_preference_target`, `weighted_policy_loss`,
  `train_weighted_logistic_router`, `LogisticTrainConfig`.
- **Hard-label mode (Section 6)**: preserved for ablation. Four-way
  experiment: hard/soft × centroid/logistic.
- **Gap-threshold tie truncation (Section 7)**: `|ΔU| <= gap_threshold` →
  weight = 0. Numerical stability floor applied only after filtering.
- **Confidence model (Section 8)**: `OutcomeConfidence` with explicit
  provenance components (verifier, measurement, stability, ood). Combined =
  product of all components.
- **SNR weighting (Section 9)**: `w = |ΔU| / (σ_ΔU + ε)` interface for
  uncertainty-aware weighting.
- **Calibrated abstention (Section 10)**: `conf = max(p, 1-p) < τ_conf →
  ABSTAIN`. Added `Route` enum, `choose_route`. `τ_conf` tuned on calibration
  split.
- **Calibration metrics (Section 11)**: Brier score, ECE, reliability bins,
  selective-risk curve.
- **OOD detection (Section 12)**: `MahalanobisOOD` with regularized
  covariance. Routes to ABSTAIN when `d_M(h) > τ_OOD`.
- **Feature reduction (Section 13)**: `PCAFeatureReducer` fitted on TRAIN
  only. No dev/calibration/final data may influence the PCA basis.
- **Causal intervention experiments (Sections 14-16)**: dose-response
  `h'(δ) = h + δv` across `{-α, -α/2, 0, α/2, α}`. Added
  `run_intervention_experiment`, `dose_response_summary`.
- **Direction reversal test (Section 15)**: `+v`, `0`, `-v` must show
  statistically measurable aggregate sensitivity. Added
  `direction_reversal_test`.
- **KL/capability promotion gates (Section 17)**: `PromotionConstraints`
  requiring `utility_gain >= min_gain AND neutral_KL <= KL_budget AND
  capability_drop <= allowed_drop`. Added `evaluate_kl_capability_gate`,
  `mean_kl_neutral`.
- **Candidate vs incumbent evaluation (Section 18)**: uses actual policy
  decisions, never oracle. Oracle counterfactual utilities used only for
  regret scoring after the policy acts.
- **Paired promotion statistics (Section 19)**: mean/median utility delta,
  win/tie/loss rates, paired bootstrap CI. Added
  `paired_promotion_statistics`, `paired_bootstrap_mean_ci`.
- **Contextual bandit logging (Section 20)**: `PolicyDecision` logs chosen
  action, propensity, full probabilities, policy version, available actions.
  Added `log_policy_decision`, `inverse_propensity_weight`.
- **Doubly-robust interface (Section 21)**: `doubly_robust_utility` for
  future off-policy learning.
- **Prioritized experience replay (Section 22)**: `priority = |pred -
  realized| × |ΔU|`. Added `PrioritizedReplayBuffer`, `ReplayExperience`,
  `replay_priority`.
- **Multi-seed support (Section 23)**: all stochastic components accept a
  `seed` parameter.
- **Family-disjoint evaluation (Section 24)**: leverages existing
  `split_family` infrastructure.
- **Immutable experiment config (Section 33)**: `ExperimentConfig` (frozen,
  hashed) containing all utility weights, thresholds, layer, alpha, KL/OOD
  limits, seed, model/tokenizer revision, dataset hashes.
- **Synthetic closed-loop environment (Section 35)**: deterministic test
  environment with family S/L/tie tasks. Added `make_synthetic_tasks`,
  `synthetic_execute_fn`, `synthetic_utility`.
- **Latent verifier experiment (Section 30)**: `LatentVerifier` using
  trajectory centroids `μ_correct`, `μ_incorrect`. Auxiliary signal only.
- **Low-rank multi-vector controller (Sections 26-28, experimental)**:
  `LowRankSteeringController` with `V ∈ R^{D×K}`, `α(h) = g_φ(h)`,
  `h' = h + Vα(h)`. Added `orthogonality_loss`, `interference_matrix`.
- **Integrated policy learner**: `train_policy_learner` ties counterfactual
  experience collection, utility weighting, PCA, OOD, logistic router
  training, and dev evaluation with regret/calibration/abstention metrics.
- **67 new release-gate tests (G2-G16)**: weighted centroid, soft-target,
  regret, abstention, OOD, causal steering, oracle-leakage, KL promotion
  gate, rollback atomicity, task-ID alignment, version consistency, synthetic
  closed-loop regret reduction, calibration, PCA, replay, bandit logging,
  low-rank, latent verifier, SNR weighting.

# 0.3.9 (causal learning loop repair — counterfactual engine correction)

- **Critical fix**: held-out promotion evaluation now uses the candidate
  steering vector to produce candidate routing decisions and the incumbent
  vector to produce incumbent routing decisions, via an injected
  `RoutePolicyFn`. Previously the candidate route was derived from the oracle
  Delta-U target and the incumbent route was hard-coded to `"symbolic"`,
  making promotion invalid.
- Added `SteeringPolicyConfig` as the canonical frozen runtime configuration
  for steering lineage (layer, alpha, token_location, model/tokenizer
  revision, vector version). Lineage is now generated from this config instead
  of hard-coded values (`layer=24`, `alpha=1.0`, `token_location="anchor"`).
- Added `PolicyRouteDecision` with full provenance (task_id, route,
  confidence, vector_id, model_revision, layer, token_location, alpha,
  safety_clamp_telemetry).
- Added `CapturedActivation` and `CaptureResult` structs for per-task capture
  alignment by `task_id`. Weighted class means now join activations and
  utility weights by task_id, never by positional index.
- Fixed trust-region bootstrap: zero-vector incumbents now initialize from
  the candidate direction with a configurable `initial_steering_norm`.
  Added `min_vector_norm`, `max_vector_norm`, `max_update_norm` bounds.
  Fail-closed on NaN/Inf vectors.
- Fixed dataset hash lineage: training and development dataset hashes are
  now tracked independently (`training_dataset_sha256`,
  `development_dataset_sha256`).
- Added capability regression gates to the promotion gate
  (`capability_regression_thresholds`, `protected_capabilities`).
- Added `min_sample_count` to the promotion gate config.
- `HeldOutTaskResult` now carries `candidate_route`, `incumbent_route`,
  `capability_id`, `task_family`, and `utility_delta`.
- Consolidated versioning to 0.3.9 across all surfaces (pyproject.toml,
  `__init__.py`, README, CLAIMS, CLI scripts).
- CLI `daph-autolearn run` now supports `--engine counterfactual` (default)
  and `--engine legacy` (deprecated).

# 0.3.8 (protocol repair and truthful qualification boundary)

- Added teacher-forced full-sequence route scoring with explicit mean/sum
  normalization and fail-closed prompt-boundary validation. Model-mediated
  AutoLearn and tuning paths now use sequence scoring by default.
- Added typed outcome verification. Numeric outputs accept only an exact
  integer or `FINAL: <integer>`; numeric substring matching is prohibited.
- Added family-aware split construction plus exact, normalized,
  task-fingerprint, task-ID, template-family, and within-split duplicate
  audits.
- Added a deterministic 13-category protocol dataset generator with disjoint
  train/development/calibration/final-test template slots.
- Added purpose-aware split guards and an atomic one-shot final-test access
  ledger. Headline manifests require frozen configuration, sequence-scoring,
  leakage-report provenance, and final-test access provenance where relevant.
- Replaced the invalid five-control normal-theory z-score with finite-sample
  empirical-null inference. Controls now enforce identical ordered task
  fingerprints and scorer configuration; fewer than 500 controls are marked
  protocol-ineligible.
- Added steering safety clamps for relative perturbation and cosine shift.
  Reasoning-policy generation defaults to cosine alpha decay.
- Moved command implementations into `daph_learning.cli.commands`, fixing
  ordinary wheel installs where repository-level `scripts/` are absent.
- Corrected confidence monotonicity in the policy router and fixed tuple
  handling in the AutoLearn validation generation path.
- Bundled GDN2/ExFusion extension v0.2.1, which recomputes teacher
  coefficients against the actual low-rank compressed basis and reports
  post-compression reconstruction diagnostics.
- Quarantined the contaminated v0.3.7 Qwen experiment as historical,
  non-headline evidence and rewrote the README/claims boundary accordingly.
- Explicitly deferred counterfactual AutoLearn v2, Latent Memory v0.5.2,
  real-model qualification, and production claims.

# 0.3.7 (route_fn + dependency direction + typed errors)

- **V037-001 — route_fn dead path**: `run_autolearn_loop` accepted a
  `route_fn` argument but never executed it (a conditional check silently
  bypassed the custom router). Introduced `RouteDecision` dataclass and
  `route_tasks` dispatcher in `src/daph_learning/routing/policy.py`;
  `run_autolearn_loop` now calls `route_tasks` when `route_fn` is supplied.
  Added `tests/test_route_fn_contract.py` covering the dispatch contract,
  `RouteDecision` coercion, and exception propagation (custom routers
  must not be silently swallowed).
- **V037-002 — library no longer depends on scripts/**: the v0.3.6 library
  layer imported reusable helpers from the CLI layer
  (`from scripts.evaluate_routes import ...`, `from scripts.tune_steering
  import ...`), inverting the correct dependency direction. Moved the
  reusable logic into new library modules:
    - `src/daph_learning/evaluation/routes.py` — `evaluate_route_records`,
      `load_jsonl`
    - `src/daph_learning/routing/batched.py` —
      `evaluate_batch_steered_routes`, `as_task_map`, `score_key`, `chunks`
    - `src/daph_learning/data/task_utils.py` — `format_for_model`, `load_llm`
    - `src/daph_learning/experiments/manifest.py` — `emit_manifest`,
      `manifest_reference_line`, `GitInfo`, enrich helpers
  `scripts/_manifest.py` is now a thin re-export shim for backward
  compatibility. All scripts import from `daph_learning.*` directly (no
  cross-script imports). Added `tests/test_no_scripts_imports.py` (AST +
  text scan) enforcing that no `src/daph_learning/**` module imports from
  `scripts/`. Side effect: the pre-existing
  `tests/test_v032_repairs.py` / `tests/test_v033_repairs.py` subprocess
  failures (scripts/evaluate_routes.py invoked as a subprocess could not
  import `scripts._manifest` when only `PYTHONPATH=src` was set) are now
  fixed. Full suite passes with `PYTHONPATH=src` only.
- **V037-003 — typed errors + telemetry**: replaced every
  `except Exception:` in `src/` with typed catches from a new
  `daph_learning.routing.errors` taxonomy:
    - `RouteResolutionError` (base)
    - `MultiTokenRouteError(ValueError, RouteResolutionError)`
    - `ContextBoundaryError(ValueError, RouteResolutionError)`
    - `InvalidRouteDecisionError(RouteResolutionError)`
    - `SteeringApplicationError`
    - `ModelRoutingError`
  `MultiTokenRouteError` and `ContextBoundaryError` subclass `ValueError`
  so existing `except ValueError` callers and `pytest.raises(ValueError)`
  tests keep working. Every routing fallback now emits structured
  telemetry via `daph_learning.telemetry.emit_fallback` (in-process event
  list + configurable sink). A pre-existing iteration bug in the training
  routing cascade — `zip(task_list, steered_routes)` where
  `steered_routes` is a dict, yielding keys not items — was masked by the
  old bare-except and is now fixed. Added
  `tests/test_exception_fallbacks.py` (17 tests covering the taxonomy,
  backward-compat, typed raises, telemetry, propagation).
- **V037-004 — version + claims discipline**: bumped version surfaces
  (`pyproject.toml`, `__init__.py`, `README.md`) to 0.3.7. Updated
  `CLAIMS.md` header to v0.3.7 and added §19 documenting the v0.3.7
  engineering changes and their (non-)impact on the scientific claims.
  Updated §6 test count to 340 passing. Updated §18 AutoLearn status to
  reflect that the multi-token tokenizer limitation that previously
  prevented the loop from updating is now addressed by the typed-error
  cascade with telemetry (the loop now correctly falls back to generate
  mode instead of silently swallowing).
- **V037-005 — environment provenance capture**: added
  `src/daph_learning/environment.py` with `capture_environment`, which
  uses `importlib.metadata.version` for installed-package versions
  (torch, transformers, accelerate, peft, safetensors, numpy,
  daph-learning) — the *installed* version, not the *imported* version,
  which matters for editable installs. Captures CUDA runtime + driver,
  GPU name, GPU compute capability, attention implementation, dtype, and
  device map. `fail_closed=True` raises
  `EnvironmentProvenanceError` when a REQUIRED_FOR_HEADLINE_ENV field
  cannot be captured. `emit_manifest(fail_closed_environment=True)`
  propagates the fail-closed behavior. The v0.3.6
  `_detect_environment` in `experiments/manifest.py` now delegates to
  `capture_environment`. Added `tests/test_manifest_environment.py`
  (16 tests).
- **V037-006 — proper CLI installation via entry points**: added
  `[project.scripts]` to `pyproject.toml` with 5 entry points
  (`daph-autolearn`, `daph-evaluate-routes`, `daph-build-oracles`,
  `daph-tune-steering`, `daph-random-control`). Created
  `src/daph_learning/cli/` with one thin module per entry point that
  loads the corresponding `scripts/*.py` by file path via
  `importlib.util` and calls its `main()`. After `pip install -e .`,
  the commands are on PATH and `PYTHONPATH=src:.` is no longer
  required. Added `tests/test_cli_entrypoints.py` (13 tests).

Test count: 392 passed, 1 skipped (was 320 + 4 pre-existing failures).

# 0.3.5 (manifest + claims patch)

- Added `CLAIMS.md` pinning down what each term used in the repository is
  currently licensed to assert (AutoLearn, OOD benchmark, gold route labels,
  steering improves routing, reasoning steering, test count, latency,
  provenance, tokenizer portability). Each entry carries an
  `ESTABLISHED` / `BOOTSTRAP` / `NOT YET` / `OUT OF SCOPE` status tag.
- Added `docs/RUN_MANIFEST.md` specifying the `daph.run.v1` run manifest
  schema: model revision / config hash / dtype, tokenizer revision /
  chat-template hash, environment (torch / transformers / CUDA / GPU),
  dataset SHA-256 + split + `label_oracle_kind`, decoding (seed, prompt
  format, `route_token_resolver`), per-vector provenance, and pytorch
  determinism. Three-tier validation: REQUIRED, REQUIRED_FOR_HEADLINE,
  OPTIONAL.
- Added `src/daph_learning/evaluation/manifest.py` implementing the schema
  with `build_manifest`, `validate` (with `headline=True` tier),
  `serialize`, `manifest_sha256`, `write_manifest` / `load_manifest`,
  `hash_file`, `hash_vector_values`. Exposed via
  `daph_learning.evaluation.__init__`.
- Added programmatic anti-circularity guard: a manifest whose
  `dataset.split` is `test`/`final_test` and whose vector's
  `capture_dataset_sha256` equals the manifest's own `dataset.sha256` is
  rejected as a split-leakage violation.
- Added token-resolver honesty field: `decoding.route_token_resolver`
  must be `isolated` (what v0.3.5 actually does) or `contextual` (not yet
  implemented). Setting `contextual` without the implementation is a
  validation error.
- Extended `SteeringSpec` with capture provenance fields:
  `extraction_method`, `normalization`, `positive_n`, `negative_n`,
  `capture_anchor`, `capture_prompt_format`, `capture_dataset_path`,
  `capture_dataset_sha256`. All default to `None` for backward
  compatibility; older `.npz` files still load. Headline manifest
  validation flags vectors missing these fields.
- Updated `contrastive_mean_direction` to honor `spec.normalization`
  (`"l2"` / `"none"` / `"mean_centered"`) when the explicit `normalize`
  argument is `None`. Backward-compatible default remains `True` (L2).
- Updated `extract_steering_vector.py` with `--extraction-method`,
  `--normalization`, `--positive-n`, `--negative-n`, `--capture-anchor`,
  `--capture-prompt-format`, `--capture-dataset-path`,
  `--capture-dataset-sha256` flags. `--positive-n` / `--negative-n`
  auto-fill from array shapes when omitted.
- Updated `capture_router_activations.py` to compute and emit the capture
  dataset SHA-256, positive/negative sample counts, capture anchor, and
  capture prompt format in its JSON summary, plus a ready-to-paste
  `extract_command` with all provenance flags filled in.
- Added `scripts/_manifest.py` shared helper for best-effort environment
  detection (python / torch / transformers / CUDA / GPU / git commit /
  dirty), vector-entry construction reading provenance from `SteeringSpec`
  with `vector_sha256`, and manifest emission as a
  `<output>.manifest.json` sibling file.
- Wired manifest emission into `generate_v0_outputs.py`,
  `tune_steering.py`, and `evaluate_routes.py`. Each now accepts
  `--dataset-split`, `--label-field`, `--label-oracle-kind`,
  `--run-id`, and `--no-manifest` flags. Manifest validation failures
  print a warning naming the missing fields but do not invalidate the
  output file itself.
- `_load_vector_cli` now returns `(vectors, source_paths)` so the
  manifest can cite the on-disk vector files.
- Added 25 schema tests (`tests/test_manifest.py`), 6 emission
  integration tests (`tests/test_manifest_emission.py`), 11
  provenance roundtrip tests (`tests/test_steering_provenance.py`),
  22 OOD benchmark tests (`tests/test_ood_benchmark_v035.py`),
  14 real-model integration tests
  (`tests/test_real_model_integration.py`),
  10 contextual token resolver tests
  (`tests/test_contextual_token_resolver.py`),
  8 activation telemetry tests (`tests/test_activation_telemetry.py`),
  8 latency measurement tests (`tests/test_latency_measurement.py`),
  5 alpha-grid tests (`tests/test_alpha_grid.py`),
  10 linear-probe baseline tests
  (`tests/test_linear_probe_baseline.py`), 6 random-direction
  control tests (`tests/test_random_direction_control.py`), and
  20 AutoLearn loop tests (`tests/test_autolearn_loop.py`).

# 0.3.5 (AutoLearn learning loop)

- Added `src/daph_learning/autolearn/` module implementing the iterative
  learning loop that gives the project its name. The loop:
  1. Routes training tasks with the current steering vector.
  2. Executes the chosen backend (symbolic or LLM).
  3. Classifies outcomes as correct / misrouted / unverifiable using
     `classify_outcome()`.
  4. Captures activations from correctly-routed (positive) and
     misrouted (negative) tasks.
  5. Updates the steering vector via contrastive mean difference.
  6. Evaluates on the validation set.
  7. Repeats for N iterations, tracking the best vector by validation F1.
  Early-stops when no update is possible (not enough examples).
  See CLAIMS.md §18.
- Added `scripts/autolearn.py` CLI driver with `--train-tasks`,
  `--val-tasks`, `--model`, `--layer`, `--n-iterations`, `--alpha`,
  `--anchor`, `--min-examples`, `--normalization`, `--seed`,
  `--prompt-format`, `--label-field`, `--label-oracle-kind`,
  `--output`, and `--vector-output` flags. Prints the learning curve
  (iteration, train accuracy, val F1, val accuracy, updated flag,
  vector norm) and saves the best vector to .npz.
- Added `classify_outcome()` function that determines whether a routed
  task execution was correct, misrouted, or unverifiable, using the
  expected answer and the output format. This is the core feedback
  signal for the learning loop.
- Ran a historical experiment suite on Qwen2.5-1.5B-Instruct (28 layers,
  hidden_size=1536, MPS backend). The following values were recorded at the
  time but were invalidated by the v0.3.8 audit because of unequal control
  task sets, five-control inference, test tuning, and split leakage:
  - Baseline (no steering): 70% acc, 0% F1
  - Extract-once steering (layer=20, alpha=3.0, norm='none'):
    90% acc, 62% F1 — observation only; no causal claim licensed
  - Random-direction control: F1=0.05 ± 0.07, z=8.6
    (`real_vector_outperforms`) — invalid as significance or
    direction-specific causal evidence
  - Linear probe: 100% CV accuracy — activations perfectly separate
    symbolic/LLM, but probe weights as steering direction produce
    F1=0.0 (classification ≠ causal intervention)
  - AutoLearn loop: did not update (logit routing fails for Qwen's
    multi-token tokenizer; needs generate-mode support)
  - Key finding: `normalization='none'` is critical — L2-normalized
    vectors (norm=1) have no effect because the residual stream norm
    is ~71. The raw mean difference (norm~15) at alpha=3.0 produces
    a perturbation that is ~60% of the residual stream norm.

# 0.3.5 (alpha-grid + linear-probe baseline + random-direction control)

- Added `--alpha-grid` to `tune_steering.py` for independent per-vector
  composite coefficient sweeping. Each grid entry is a comma-separated
  list of alphas, one per vector in the bundle. Example:
  `--alpha-grid 0.5,1.0 1.0,2.0` sweeps two configurations with
  independent per-vector alphas. Mutually exclusive with `--alphas`.
  Results record `scale_semantics: "independent_per_vector"`. See
  CLAIMS.md §10 and audit §10.
- Added `scripts/linear_probe_baseline.py`: trains a logistic regression
  on captured router activations to predict the symbolic/llm route label
  via 5-fold stratified cross-validation. Reports accuracy, precision,
  recall, F1, confusion matrix, majority-class baseline, and lift over
  baseline. Provides the steering-free upper bound required by audit §9
  for evaluating whether steering contributes beyond what a simple linear
  classifier on the same activations could do.
- Added `scripts/random_direction_control.py`: matched-norm
  random-direction control experiment. Generates N random directions
  with the same L2 norm as the real steering vector, runs the routing
  pipeline with each, and compares metrics. Reports `f1_lift` and
  `f1_z_score` with an interpretation tag
  (`real_vector_outperforms` / `within_random_range` / `underperforms`).
  This is the causal ablation required by audit §11: if random
  directions produce the same routing change, the steering vector's
  direction is not causally responsible.

- Added `resolve_route_token_ids_contextual` to
  `daph_learning.routing.steered_router`. Derives route-label token IDs
  by diffing `T(rendered_prompt + label)` against `T(rendered_prompt)`,
  eliminating the boundary assumption that fails for some BPE/SentencePiece
  tokenizers. See CLAIMS.md §9 and audit §7.
- Added `token_resolver` parameter to `score_route_batch_from_logits`
  (`"isolated"` default for backward compat, `"contextual"` for
  headline-eligible results). Wired through `generate_v0_outputs.py`
  and `tune_steering.py` via `--route-token-resolver` CLI flag.
- Added activation telemetry to `residual_addition_hook` and
  `multi_layer_residual_addition_hook`. When a `telemetry_sink` list is
  provided, the hook appends per-forward-pass stats: `layer_index`,
  `n_steered_positions`, `h_norm_mean`, `av_norm_mean`,
  `relative_perturbation` (`|αv|/|h|`), and `cosine_shift_mean`
  (`1 - cos(h, h+αv)`). Wired through `score_route_batch_from_logits`
  and `_generate_batch`. See CLAIMS.md §20 and audit §20.
- Added per-stage latency measurement to `generate_v0_outputs.py`.
  Route records now include `route_batch_ms`, `symbolic_exec_ms`, and
  `answer_batch_ms` in addition to the existing `pipeline_elapsed_ms`.
  Skipped stages report 0.0. See CLAIMS.md §22 and audit §22.

# 0.3.5 (OOD benchmark + typed oracle labels + real-model tests)

- Expanded `generate_ood_benchmark.py` from 5 to 13 categories. Added
  `nested_arithmetic`, `signed_nested_arithmetic` (symbolic group);
  `text_question`, `coding_question`, `knowledge_question`
  (non-symbolic controls); `irrelevant_numeric`, `ambiguous_tool`
  (adversarial numeric controls); `unsupported_arithmetic` (malformed,
  uses true division to test fallback). Each task carries a
  `category_group` field (`symbolic` / `non_symbolic` / `adversarial` /
  `malformed`) for aggregate analysis. Added `--symbolic-only`,
  `--non-symbolic-only`, and `--categories` CLI flags for generating
  targeted subsets.
- Added typed oracle labels: every task now carries
  `capability_oracle`, `accuracy_oracle`, and `utility_oracle` fields
  (each `"symbolic"` or `"llm"`), in addition to the backward-compatible
  `route_label` (which equals `utility_oracle`). This fixes the
  three-oracles-conflated-into-one-string problem from the audit §13.
  For example, `easy_add` now has `capability_oracle="symbolic"` (the
  tool can do it), `accuracy_oracle="llm"` (the LLM is more accurate on
  small numbers), `utility_oracle="llm"` (prefer LLM for this category).
- Updated `evaluate_route_records` to accept `label_oracle_kind` and
  include it in the returned metrics dict, so downstream consumers
  cannot silently re-interpret a `policy_heuristic` result as a
  capability oracle.
- Updated `tune_steering.py` to pass `label_oracle_kind` through to
  `evaluate_route_records`.
- Added real-model integration tests
  (`tests/test_real_model_integration.py`) using `sshleifer/tiny-gpt2`.
  Covers: layer resolution, vector validation, anchor mapping, activation
  capture, multi-layer hooks, batched generation, batch-size consistency
  (the §24 check: same prompt produces same output in batch size 1 vs
  batch), symbolic execution, symbolic→LLM fallback, extract/load
  roundtrip with provenance, chat-template guard. Skipped if torch or
  transformers is unavailable.

# 0.3.5

- Auto-detects leading-space vs raw single-token route labels for direct-logit routing.
- Added comma-separated and JSON multi-layer steering bundle loading with duplicate-layer validation.
- Added composite tool-policy and reasoning-policy steering CLI support in full output generation.
- Refactored full output generation into batched model-mediated routing and batched answer-generation stages.
- Added configurable `--batch-size` and `--route-batch-size`.
- Added composite bundle tuning; bundle sweep alpha acts as a global multiplier over stored per-layer alphas.
- Added McNemar continuity-corrected statistic/p-value, discordant odds ratio, and exact Clopper-Pearson confidence intervals.
- Extended V0 hint-ablation gate JSON telemetry with paired-effect confidence intervals.

# 0.3.4

- Added direct next-token logit-contrast routing with explicit single-token label validation.
- Added `route-decision-mode={auto,logit,generate}` and route margin/method telemetry.
- Added batch-aware anchor steering for variable-length left-padded prompt batches.
- Added multi-layer composite residual steering context manager.
- Refactored steering tuning to batched single-forward logit scoring with configurable batch size and margin threshold.
- Added deterministic, balanced, seed-controlled OOD benchmark generation with SHA-256 output.
- Added regression tests for logit routing, composite hooks, batch target indices, and benchmark determinism.

# 0.3.3

- Fixed malformed-route confusion accounting so malformed LLM-oracle decisions no longer inflate true negatives or specificity.
- Added reusable `evaluate_route_records()` for route evaluation and steering tuning.
- Added regex-based embedded `$var` discovery/substitution across nested symbolic DAG arguments.
- Added explicit `target_token_index` support to residual steering and activation capture.
- Added `find_anchor_token_index()` so `ACTION:` can be targeted inside fully rendered chat templates instead of steering the appended assistant-generation header.
- Connected anchor-aligned capture and injection paths.
- Added runtime alpha overrides for tool-policy and reasoning-policy steering.
- Added real `scripts/tune_steering.py` layer/alpha validation sweep that loads the model once, runs steered routing, scores route metrics, and selects the best configuration.
- Added v0.3.3 regression tests for TN accounting, embedded variables, target-token steering, anchor alignment, and tuning selection logic.

# 0.3.2

- Preserve retrieved procedural memory when symbolic execution fails over to the LLM.
- Count malformed route decisions in the binary confusion matrix instead of silently skipping them; report malformed rate and route accuracy.
- Make `token_scope="last"` a one-shot prompt-final-token intervention, protected against cached and non-cached autoregressive decode reapplication.
- Isolate arithmetic text before trailing natural-language/numeric instructions.
- Extend `ExecutionPlan` to multi-action dependency DAGs via `output_var`, `$var` references, topological scheduling, cycle detection, and optional `result_var`.
- Add regression tests for all v0.3.2 repairs.

# 0.3.1

- Connected ExecutionPlan actions to actual symbolic dispatch; removed gold-answer leakage from runtime verification.
- Added importable scripts package and fixed `inspect_integrity.py`.
- Added `steered_auto`: tool-policy steering now causally participates in the LLM call/no-call routing pass.
- Split tool-policy steering from reasoning-policy steering with family validation.
- Added token-scoped residual hooks (`last` for route action, `all` for reasoning experiments).
- Added vector/model dimensional validation and richer steering metadata.
- Added route evaluator, exact paired V0 hint-ablation gate, and contrastive steering-vector extraction CLI.
- Corrected `scientific_gate` semantics: integrity/paired checks are separate from the new `effect_size_pass`.
- Added optional explicit route-label fields for activation capture and routing evaluation to avoid circular policy-oracle evaluation.
- Centralized output parsers under `daph_learning.evaluation`.
- Added `mod` modulus alias and stronger task-schema/duplicate-ID validation.
- Added raw/chat prompt-format switch to preserve historical compatibility while supporting instruct chat templates.
- Added expanded integration/security tests and CLI smoke tests.

# 0.3.0

- Fixed procedural-memory loading path.
- Replaced raw/symbolic eval assumptions with typed exact integer execution.
- Added bounded AST fallback.
- Added explicit capability assessment and routing decision objects.
- Added ExecutionPlan / SymbolicAction / ExecutionResult IR.
- Added auditable route logging.
- Added deterministic symbolic, auto, and hybrid execution modes.
- Added stable `FINAL:` scoring contract.
- Added activation steering vector types, extraction, residual hooks, and serialization.
- Added integrity, memory, router, scorer, and symbolic-security tests.
