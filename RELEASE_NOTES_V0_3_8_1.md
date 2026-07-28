# DAPH AutoLearn v0.3.8.1 Release Notes — AutoLearn correctness hotfix

## Outcome

This is a targeted correctness hotfix to the v0.3.8 AutoLearn loop. It does **not**
implement the v0.3.9 counterfactual both-backend learning; it repairs seven defects
in the current build that either lied about what was measured or destroyed useful
signal. No new GPU experiment should be run on v0.3.8 without these fixes.

The protocol layer is unchanged. The changes are concentrated in the AutoLearn
loop, the activation-capture path, and the routing-decision taxonomy.

## Implemented repairs

### 1. Stop coercing `None` routes into LLM decisions

A failed or undecidable routing response is not evidence for the LLM backend.
Previously the loop did `route if route else "llm"` and the simple evaluator did
`r if r is not None else "llm"`, so a router could look good simply by failing on
LLM-labelled examples.

- `routes` in `run_autolearn_loop` now preserves `None` (invalid) and
  `"abstain"` as their own categories. Neither runs a backend; both classify
  as `unverifiable` rather than as a hidden LLM prediction.
- `_evaluate_routes_simple` no longer scores `None` as LLM. It reports
  `accuracy_on_decisions` (over decided routes only), `decision_coverage`,
  `invalid_rate`, `abstain_rate`, and `overall_utility` (a coverage-aware
  proxy). The legacy `accuracy`/`f1` keys are retained but are now computed
  over decided routes only.

A 99%-accuracy / 20%-coverage router is no longer equivalent to a
95%-accuracy / 100%-coverage router.

### 2. Stop collapsing `abstain` into `llm`

The custom-router path did `routes[tid] = "llm" if decision.route == "abstain"
else decision.route`. That destroyed the abstention signal. Abstention is an
honest controller option, not a hidden LLM route.

- `abstain` is now preserved end-to-end and counted in `abstain_rate`. A real
  abstain backend arrives in v0.3.9 (V039-003); until then abstain produces no
  execution and is reported as `unverifiable`, never as LLM.

### 3. The configured anchor is now actually captured

`_capture_activations` accepted an `anchor` argument but installed the hook with
`target_token_index=-1`, so metadata advertised `capture_anchor = ACTION:` while
the vector was learned from the last prompt token (which, under a chat template,
is the generation-header token, not the anchor).

- The anchor is now resolved in the rendered prompt via
  `find_anchor_token_index` / `find_anchor_token_span` and captured explicitly.
- Anchor resolution failure is recorded as a capture failure (with
  `reason="anchor_resolution_failed"` telemetry) rather than silently falling
  back to the last token.

### 4. `capture_token_scope` is now wired in

`AutoLearnConfig.capture_token_scope` existed but was unused. It now controls
the capture-pooling method and is recorded in `SteeringSpec.capture_token_scope`
provenance:

- `"anchor"` (new default): resolve and capture the anchor token.
- `"last"`: capture the final prompt token (legacy v0.3.x behaviour).
- `"mean_anchor_span"`: mean-pool over every token overlapping the anchor.
- `"mean_sequence"`: mean-pool over the whole sequence.

The default changed from `"last"` to `"anchor"` so the configured anchor is
honoured by default. The CLI exposes `--capture-token-scope`.

### 5. Resolver fallback semantics/documentation fixed

The loop iterated `for resolver in ("contextual",)` (only one resolver) but the
`ModelRoutingError` text claimed "both contextual and isolated resolvers failed."
Full-sequence scoring (`score_route_batch_from_sequences`) does not take a
`token_resolver`; `isolated` is a single-token-scorer concept and is not a
distinct operation for teacher-forced label scoring. The misleading claim was
removed and the error message now accurately states that only the contextual
sequence resolver is attempted.

### 6. Activation-capture coverage metrics

Capture failures were silently skipped. They are now tracked per class
(positive = should-route-symbolic, negative = should-route-LLM) so a biased
steering vector can be detected:

- `IterationMetrics` now carries `n_capture_attempts`, `n_capture_successes`,
  `capture_coverage`, `capture_failure_rate_positive` (P(F | positive)), and
  `capture_failure_rate_negative` (P(F | negative)).
- Every capture failure emits structured telemetry with `class_label` and
  `scope`. A run manifest can now report P(F | S) vs P(F | L); a protocol gate
  on `capture_coverage` is straightforward to add on top of these fields.

### 7. Explicit stop reasons

The loop stopped on `not updated and iteration > 0` with the comment "the vector
has converged." That conclusion does not follow — "no update" can mean too few
examples, capture failure, or every example being unverifiable.

- `AutoLearnResult.stop_reason` is now one of `converged`,
  `insufficient_positive`, `insufficient_negative`, `capture_failure`,
  `no_verifiable_experiences`, `validation_rejected`, `max_iterations`.
- The v0.3.8.1 loop never emits `"converged"` (real stability detection arrives
  with v0.3.9 candidate promotion); it diagnoses the actual cause. The CLI
  prints and serializes `stop_reason`.

## What this release does NOT do

It does not run the LLM backend during the learning loop. That is the central
v0.3.9 architectural change (both-backend execution, independent verification,
counterfactual reward gaps, weighted experiences, immutable replay, candidate
promotion/rollback). The v0.3.8.1 loop still skips LLM execution and honestly
classifies an LLM-routed task with no executed output as `unverifiable` — so an
LLM-routed task can no longer inflate routing metrics by failing to execute,
but the loop still cannot learn whether routing to the LLM was a good decision.
That is the next release.

## Test impact

- New tests cover stop reasons, abstain preservation, anchor resolution,
  capture-failure accounting, capture-coverage metrics, coverage-aware
  evaluation, and `capture_token_scope` provenance.
- `test_evaluate_routes_simple_none_routes` was updated to the new semantics
  (a `None` route is no longer a correct LLM guess).
- `CLAIMS.md §6` test-count lower bound was bumped to 417.
- The two `test_cli_entrypoints` failures in this environment are the known
  install-dependent tests (require `pip install -e .`); they are unchanged by
  this hotfix.
