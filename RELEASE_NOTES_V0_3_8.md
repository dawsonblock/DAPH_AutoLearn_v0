# DAPH AutoLearn v0.3.8 Release Notes

## Outcome

This release converts the audited v0.3.7 tree into a truthful protocol-repair
release. It fixes implementation defects that can be validated locally and
encodes safeguards needed before new real-model experiments are run.

It does not synthesize missing GPU evidence or implement roadmap systems under
new version labels.

## Implemented repairs

### Routing

- Added teacher-forced complete-label scoring in
  `routing/logit_router.py`.
- Supports explicit `mean` and `sum` label-score normalization.
- Uses one combined forward pass for both candidates across the batch.
- Fails closed if prompt tokenization is not a strict prefix of
  prompt-plus-label tokenization.
- Makes sequence scoring the default in tuning, random controls, the
  model-mediated AutoLearn path, and the generation routing path.
- Retains the first-token scorer only as a legacy comparison mode.

### Verification

- Added typed `VerificationStatus` and `VerificationResult`.
- Added a verifier protocol plus numeric and route-suitability verifiers.
- Numeric correctness accepts only a bare integer or `FINAL: <integer>`.
- Raw route text can recover a route but can never be reused as an answer.
- Fixed the false-positive case where expected `12` matched output `312`.

### Dataset and split protocol

- Added exact, normalized, task-fingerprint, task-ID, family/template, and
  within-split duplicate detection.
- Added grouped family-aware splitting.
- Added a deterministic 13-category generator with disjoint template slots
  for train, development, calibration, and final test.
- Added purpose-aware split access checks.
- Added an atomic one-shot final-test access ledger.

### Statistical controls

- Removed normal-theory z-score inference from the active random-control
  implementation.
- Added finite-sample empirical p-values with the plus-one correction.
- Enforced identical ordered task fingerprint digests and scorer
  configurations for real and random directions.
- Marked runs with fewer than 500 random controls or legacy token scoring as
  protocol-ineligible.

### Steering safety

- Added optional effective-alpha clamps for relative perturbation and cosine
  shift.
- Added requested/effective alpha and clamp reasons to telemetry.
- Applied the same limits to single- and multi-vector route scorers.
- Wired cosine decay into reasoning-policy generation, with 16 steps and a
  0.1 floor as the CLI defaults.

### Provenance and packaging

- Extended headline manifest validation with sequence-scoring and frozen
  protocol requirements.
- Added leakage-report and final-test access provenance.
- Moved command implementations into the installed package so ordinary
  wheels no longer depend on repository-level scripts.
- Added the installed `daph-build-protocol-dataset` command.

### GDN2/ExFusion extension

- Bundled `daph-gdn2-repobrain` v0.2.1 under `extensions/`.
- Constructs the actual rank-truncated basis before coefficient fitting.
- Recomputes teacher coefficients by least squares against that compressed
  basis.
- Reports post-compression reconstruction errors, explained energy, fit
  method, and condition diagnostics.

## Evidence quarantine

The supplied v0.3.7 Qwen experiment is preserved under
`experiments/legacy_contaminated_v0_3_7/`. Its previous headline
interpretation is withdrawn. See that directory's README and `CLAIMS.md` for
the exact reasons.

## Validation performed

- Main source suite: **394 passed, 22 skipped**.
- Skips: CUDA timing and explicitly model-backed tests whose model artifacts
  were not available locally.
- GDN2/ExFusion extension: **105 passed**.
- Main wheel built and installed cleanly:
  `daph_autolearn-0.3.8-py3-none-any.whl`.
- All six installed console commands were present; protocol-dataset and
  evaluation help paths executed from the clean wheel install.
- GDN2/ExFusion wheel built:
  `daph_gdn2_repobrain-0.2.1-py3-none-any.whl`.
- A clean-install smoke run generated all four protocol splits and produced a
  leakage report with `is_clean=true`.

## Deferred work

- Counterfactual AutoLearn v2 with both-backend execution, independent
  verification, replay weighting, and immutable promotion lineage.
- Real-model multi-seed qualification on frozen family-disjoint splits.
- Conditional low-rank steering and learned routing coefficients.
- Latent Memory v0.5.2 implementation and causal swap/corruption controls.
- Non-route KL drift caps and production qualification.
