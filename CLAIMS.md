# DAPH AutoLearn v0.3.10-alpha — Licensed Claims

This file is the authoritative claim boundary for the release. Tests can
establish that a mechanism is implemented and behaves as specified on covered
inputs. They do not by themselves establish a reproducible real-model effect,
out-of-distribution generalization, or production readiness.

Status tags:

- `ESTABLISHED` — supported as an engineering mechanism by repository tests.
- `BOOTSTRAP` — implemented for developing later experiments; not a headline
  result.
- `PARTIAL` — some necessary components exist, but the broader claim does not.
- `NOT YET` — not supported by the supplied implementation or evidence.
- `OUT OF SCOPE` — deliberately excluded from this release.

## 1. Release scope

**Status: ESTABLISHED as engineering.**

v0.3.8 is a protocol-repair release. It provides route scoring, residual
steering hooks, a bounded symbolic executor, typed verification, leakage
audits, test-access guards, empirical control inference, provenance manifests,
and wheel-safe commands.

It does not license a claim that steering improves routing on any real model.

## 2. Synthetic 13-category protocol benchmark

**Status: BOOTSTRAP.**

`daph_learning.data.benchmark` deterministically generates 13 task categories
across symbolic, non-symbolic, adversarial, and malformed groups. Template
slots are disjoint among train, development, calibration, and final-test
splits. The generator checks its own output with the leakage scanner.

This establishes a reproducible synthetic protocol dataset. It does not
establish that the tasks are naturally occurring OOD examples, representative
of deployed traffic, or scientifically validated.

## 3. Route labels and utility oracles

**Status: PARTIAL.**

The schema distinguishes capability, accuracy, utility, and legacy
policy-heuristic labels. Evaluation and manifests record which label kind was
used.

The built-in labels are generated rules, not measured backend utilities.
Calling them gold utility labels is not licensed until both backends are run
and independently verified under a frozen cost/latency/reward definition.

## 4. Multi-token route scoring

**Status: ESTABLISHED as engineering.**

`score_route_batch_from_sequences` computes teacher-forced complete-label log
probabilities for `SYMBOLIC` and `LLM`. Both candidates share the same prompts,
model state, vector, alpha, and scorer. Mean or sum normalization is explicit;
mean is the protocol default. The scorer fails closed if appending a label
retokenizes the prompt boundary.

The legacy single-token and first-subtoken modes remain available for backward
comparison, but a run using them is not protocol-eligible for a v0.3.8
headline.

## 5. Steering safety and decay

**Status: PARTIAL.**

When `SteeringSafetyLimits` is supplied, the hook clamps effective alpha to
enforce configured relative-perturbation and cosine-shift bounds. The
generation CLI applies cosine decay to reasoning-policy steering by default.
Telemetry records requested/effective alpha and clamp reasons.

The default bounds (0.65 relative perturbation and 0.25 cosine shift) are
conservative engineering limits, not universally optimal or empirically
qualified constants. Non-route KL drift is recorded in the roadmap but is not
implemented as a per-forward hook constraint in this release.

## 6. Verification and test evidence

**Status: ESTABLISHED as engineering.**

The main repository contains 655 collected tests in the release build. Tests
that require explicitly enabled model downloads or unavailable hardware may
skip. The bundled GDN2/ExFusion extension has a separate test suite.

The suite covers symbolic-executor safety, routing, full-sequence scoring,
typed verification, steering hooks and clamps, leakage checks, protocol
guards, manifests, command packaging, empirical null calculations,
regression behavior, and the v0.3.9 counterfactual outcome-semantics,
frozen-utility, immutable-experience-record, promotion-gate, and
task-appropriate output-schema tests.

The test count is engineering evidence only. It is not a sample size and does
not qualify scientific claims.

## 7. AutoLearn

**Status: BOOTSTRAP.**

The v0.3.x loop can route tasks, execute the symbolic backend, classify typed
outcomes, extract a contrastive mean direction, and repeat validation. It now
uses full-sequence route scoring on model-mediated paths.

The loop is not the proposed counterfactual AutoLearn v2 system. It does not
execute and independently verify both backends for every experience, optimize
an actual counterfactual reward-gap objective, maintain immutable replay and
promotion lineage, or demonstrate convergence on held-out real-model tasks.

Licensed wording:

> “DAPH includes a bootstrap iterative steering loop.”

Unlicensed wording:

> “DAPH autonomously learns a superior routing policy.”

## 8. Historical Qwen2.5 experiment

**Status: NOT YET.**

The archived v0.3.7 experiment observed a routing change on
Qwen2.5-1.5B-Instruct, but its design cannot establish direction-specific
causal benefit:

- the real vector used 50 test tasks while random vectors used the first 20;
- those task sets had different class balance;
- only five random directions were evaluated;
- alpha and probe settings were selected on the test set;
- prompts and generator families leaked across splits;
- the reported normal-theory z-score treated five controls as a stable null.

The materials are retained under
`experiments/legacy_contaminated_v0_3_7/` for auditability and must not be used
as headline evidence. No Qwen F1 lift, z-score, p-value, OOD benefit, or
directional-causality result from that run is licensed by v0.3.8.

## 9. Matched random-direction controls

**Status: ESTABLISHED as engineering — NOT YET scientifically qualified.**

The v0.3.8 control runner enforces the same ordered task fingerprint digest and
scorer configuration for the real and random directions. It reports the
finite-sample empirical p-value

\[
p=\frac{1+\#\{T_{\mathrm{null}}\ge T_{\mathrm{real}}\}}{N+1}.
\]

The runner marks results protocol-eligible only with at least 500 random
directions and full-sequence scoring. This implements valid bookkeeping and
inference mechanics; no qualifying real-model run is bundled.

## 10. Split discipline and provenance

**Status: ESTABLISHED as engineering.**

The leakage scanner checks cross-split exact prompts, normalized prompts,
task fingerprints, task IDs, and generator/template families. Adaptive
purposes are prohibited from using `test` or `final_test`. Final evaluation
requires frozen configuration and a one-shot access record.

Headline manifests additionally require full-label scoring, sequence
normalization, a leakage-report hash, frozen configuration, and final-test
access provenance where applicable. The existing dataset/vector capture-hash
anti-circularity check remains enforced.

These guards prevent known protocol violations when the provided interfaces
are used. They cannot prove that external data preparation or unrecorded manual
decisions were uncontaminated.

## 11. GDN2/ExFusion compressed basis extension

**Status: ESTABLISHED as engineering — NOT YET empirically qualified.**

The bundled `extensions/daph_gdn2_repobrain_v1_11_1` module constructs a
low-rank adapter basis, recomputes teacher coefficients by least squares
against the actual compressed basis, and measures post-compression
reconstruction error. This repairs the previous mismatch where coefficients
were fitted to the pre-compression SVD basis.

The implementation and unit tests do not establish downstream task quality,
compression superiority, or production readiness.

## 12. Latent Memory v0.5.2 and COCONUT

**Status: OUT OF SCOPE.**

No runnable DAPH Latent Memory v0.5.2 training subsystem is included.
Functional mismatch loss, hard-negative curricula, skill/instance
disentanglement, CLUE verification, and anti-anchoring experiments remain
design work.

COCONUT is a research reference for continuous-thought recurrence. It is not
vendored or represented as an external query-bound memory bank.

## 13. Production and cross-model claims

**Status: NOT YET.**

v0.3.8 does not license claims of:

- real-model causal steering benefit;
- generalization across model families or tokenizers;
- out-of-distribution benefit;
- autonomous policy improvement;
- latent-memory causality;
- production qualification.

Those claims require preregistered, family-disjoint, multi-seed experiments
with identical matched controls, paired uncertainty, complete manifests, and
a final split touched once after configuration freeze.
