# DAPH AutoLearn v0.3.10.6-alpha — Licensed Claims

This file is the authoritative claim boundary for the release. Tests can
establish that a mechanism is implemented and behaves as specified on covered
inputs. They do not by themselves establish a reproducible real-model effect,
out-of-distribution generalization, or production readiness.

Status tags (evidence levels — no claim may exceed its evidence level):

- `IMPLEMENTED` — code present; not yet unit-tested.
- `UNIT_TESTED` — behavior covered by repository tests.
- `SYNTHETIC_VALIDATED` — validated on synthetic data only; not real-model evidence.
- `REAL_MODEL_SMOKE` — exercised on a real Hugging Face model in a smoke run.
- `EXPERIMENTALLY_FAILED` — a real-model Gate A run was executed and failed a gate.
- `EXPERIMENTALLY_QUALIFIED` — a real-model Gate A run passed every preregistered gate.

Legacy tags still used for engineering-mechanism claims:

- `ESTABLISHED` — supported as an engineering mechanism by repository tests.
- `BOOTSTRAP` — implemented for developing later experiments; not a headline result.
- `PARTIAL` — some necessary components exist, but the broader claim does not.
- `NOT YET` — not supported by the supplied implementation or evidence.
- `OUT OF SCOPE` — deliberately excluded from this release.

## Gate A status: NOT YET REQUALIFIED

The earlier real-model Gate A run (`daph_gate_a_real_001_failed`, archived
under `artifacts/legacy/`) **FAILED** its group-aware confidence-bound gate
(LCB95% for P1−P0 = −0.041 < 0). It is retained for audit history only and
does NOT qualify the current source tree.

No new full real-model Gate A experiment has been executed for
`daph_gate_a_real_002`. The current pointer
(`artifacts/current/pointer.json`) declares `NOT_YET_REQUALIFIED` with
evidence level `IMPLEMENTED_AND_TESTED`. **No synthetic artifact is
presented as Gate A qualification evidence.**

A Gate A PASS will be claimed only when a full frozen real-model
experiment for `daph_gate_a_real_002` actually runs and every
preregistered gate passes, with a validated evidence bundle under
`artifacts/gate_a_qualified/`.

## 1. Release scope

**Status: ESTABLISHED as engineering.**

v0.3.10 is a counterfactual compute-selection learner release. It builds on the
v0.3.8 protocol-repair foundation (route scoring, residual steering hooks, a
bounded symbolic executor, typed verification, leakage audits, test-access
guards, empirical control inference, provenance manifests, wheel-safe commands)
and the v0.3.9 causal-chain repair (candidate steering vector drives candidate
routing during held-out evaluation). v0.3.10 adds a weighted soft-target
logistic router, regret as the primary metric, calibration, calibrated
abstention, OOD detection, causal intervention experiments, KL/capability
promotion gates, prioritized replay, contextual bandit logging, a
doubly-robust utility interface, an experimental low-rank multi-vector
controller, an immutable hashed `ExperimentConfig`, and the
`daph-autolearn-policy` CLI.

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
comparison, but a run using them is not protocol-eligible for a headline.

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

The main repository contains 1415 collected tests in the release build
(collected on macOS Darwin 25.2.0, Python 3.12.0, pytest 8.4.2; 1384 passed,
8 skipped). Tests that require explicitly enabled model downloads or
unavailable hardware may skip. The bundled GDN2/ExFusion extension has a
separate test suite.

The suite covers symbolic-executor safety, routing, full-sequence scoring,
typed verification, steering hooks and clamps, leakage checks, protocol
guards, manifests, command packaging, empirical null calculations,
regression behavior, the v0.3.9 counterfactual outcome-semantics,
frozen-utility, immutable-experience-record, promotion-gate, and
task-appropriate output-schema tests, and the 67 v0.3.10 release-gate tests
(G2–G16) for routing, calibration, abstention, OOD handling, intervention
effects, promotion constraints, and replay.

The test count is engineering evidence only. It is not a sample size and does
not qualify scientific claims.

## 7. AutoLearn

**Status: ESTABLISHED as engineering (mechanics) — NOT YET scientifically qualified.**

The v0.3.10 loop can route tasks, execute the symbolic backend, classify typed
outcomes, execute both backends counterfactually, compute per-task utility
gaps, train a weighted soft-target logistic router against continuous `ΔU`,
evaluate regret on held-out data, calibrate abstention thresholds, run causal
intervention experiments, and gate promotion via KL/capability constraints.
It uses full-sequence route scoring on model-mediated paths.

The v0.3.9 causal-chain repair ensures the candidate steering vector drives
candidate routing during held-out evaluation; the oracle Delta-U is used only
for the training target. The v0.3.10 release adds regret as the primary
metric, calibration, abstention, OOD detection, causal interventions, and
KL/capability promotion gates.

The loop does not yet demonstrate convergence on held-out **real-model** tasks,
and no qualifying real-model run is bundled. The low-rank multi-vector
controller is experimental.

Licensed wording:

> “DAPH includes a counterfactual compute-selection learner with calibrated
> abstention, OOD detection, causal intervention experiments, and
> KL/capability promotion gates. Regret is the primary metric. The mechanics
> are established by repository tests; no real-model scientific claim is
> licensed.”

Unlicensed wording:

> “DAPH autonomously learns a superior routing policy on real models.”

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
directional-causality result from that run is licensed by this release.

## 9. Matched random-direction controls

**Status: ESTABLISHED as engineering — NOT YET scientifically qualified.**

The control runner enforces the same ordered task fingerprint digest and
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

v0.3.10 does not license claims of:

- real-model causal steering benefit;
- generalization across model families or tokenizers;
- out-of-distribution benefit;
- autonomous policy improvement;
- latent-memory causality;
- production qualification.

Those claims require preregistered, family-disjoint, multi-seed experiments
with identical matched controls, paired uncertainty, complete manifests, and
a final split touched once after configuration freeze.
