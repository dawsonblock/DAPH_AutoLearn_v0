# DAPH AutoLearn v0.3.10-alpha

**Causal learning loop repair for auditable LLM tool-routing research.**

DAPH connects capability assessment, bounded symbolic execution, residual
activation steering, and outcome evaluation. Version 0.3.9 repairs the
counterfactual AutoLearn learning loop so that the candidate steering vector
actually determines candidate routing decisions during held-out evaluation —
fixing the critical defect where the oracle Delta-U was used to choose the
candidate route.

> The v0.3.9 causal chain is now genuine: experience -> counterfactual
> utility -> target -> candidate update -> changed policy -> changed
> decisions -> changed held-out utility -> promotion/rollback. See
> [CLAIMS.md](CLAIMS.md) and [AUDIT_REPORT_V0_3_9.md](AUDIT_REPORT_V0_3_9.md).

## What changed in v0.3.9

| Area | v0.3.9 behavior |
|---|---|
| **Candidate evaluation** | Held-out promotion evaluation now uses the candidate steering vector for candidate routing and the incumbent vector for incumbent routing, via an injected `RoutePolicyFn`. The oracle Delta-U is used only for the training target. |
| **Routing policy contract** | New `SteeringPolicyConfig`, `PolicyRouteDecision`, and `RoutePolicyFn` interface decouple the loop from any specific model implementation. |
| **Trust-region bootstrap** | Zero-vector incumbents now initialize from the candidate direction. Added `min_vector_norm`, `max_vector_norm`, `max_update_norm` bounds. Fail-closed on NaN/Inf. |
| **Capture alignment** | `CapturedActivation` and `CaptureResult` structs join activations and weights by `task_id`, never by positional index. |
| **Lineage** | Steering lineage generated from `SteeringPolicyConfig` (no more hard-coded `layer=24`, `alpha=1.0`). |
| **Dataset hashes** | Training and development dataset hashes tracked independently. |
| **Promotion gate** | Added `min_sample_count` and per-capability regression gates. |
| **CLI** | `daph-autolearn --engine counterfactual` (default) or `--engine legacy` (deprecated). |
| **Versioning** | All surfaces agree on `0.3.9`. |

## What changed in v0.3.8

| Area | v0.3.8 behavior |
|---|---|
| Route scoring | Teacher-forced full-label log-probability scoring is the protocol default. It handles multi-token labels and fails closed on unstable prompt/label boundaries. |
| Verification | Typed verification accepts only an exact integer or `FINAL: <integer>` for numeric tasks. Substring matches such as expected `12` in output `312` are rejected. |
| Dataset discipline | Family-aware splitting, exact/normalized/fingerprint leakage scans, and a deterministic 13-category protocol dataset generator. |
| Test access | Adaptive work cannot use `test` or `final_test`. Final evaluation requires frozen configuration and a one-shot access ledger. |
| Random controls | Identical task sets and scorer configurations are enforced. Inference uses the empirical Monte Carlo p-value \((1+k)/(N+1)\); headline eligibility requires at least 500 controls. |
| Steering safety | Optional per-forward clamps bound relative perturbation and cosine shift. Reasoning steering defaults to cosine decay in the generation CLI. |
| Provenance | Headline manifests require full-label scoring, a frozen protocol configuration, leakage-report provenance, and final-test access provenance where applicable. |
| Packaging | Console commands now ship their implementations inside the wheel and work without the repository-level `scripts/` directory. |
| GDN2/ExFusion | The bundled extension recomputes teacher coefficients after low-rank basis compression and reports post-compression reconstruction diagnostics. |

## Deliberate boundaries

Version 0.3.8 does **not** contain or claim:

- a counterfactual AutoLearn v2 optimizer;
- both-backend execution with immutable replay and promotion lineage;
- a Latent Memory v0.5.2 training system;
- scientifically qualified Qwen2.5 gains;
- cross-model or production qualification.

The historical v0.3.7 Qwen experiment is retained only under
`experiments/legacy_contaminated_v0_3_7/`. It used unequal task sets for the
real and random directions, only five random controls, test-set tuning, and
leaky splits. Its reported z-score is not a licensed result.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install .
```

For model-backed experiments and tests:

```bash
python -m pip install '.[full]'
```

Installed commands:

- `daph-autolearn`
- `daph-evaluate-routes`
- `daph-build-oracles`
- `daph-tune-steering`
- `daph-random-control`
- `daph-build-protocol-dataset`

## Build a leakage-audited protocol dataset

The generator creates 2,500 train, 1,000 development, 500 calibration, and
1,000 final-test tasks by default. Template slots are disjoint across splits.

```bash
daph-build-protocol-dataset \
  --output-dir data/protocol_v038 \
  --seed 1337
```

The output directory includes the four JSONL splits, a leakage report, and a
dataset manifest with SHA-256 hashes. This is a synthetic protocol benchmark;
it is not automatically an OOD or scientifically qualified benchmark.

## Full-sequence route scoring

Protocol runs score complete route labels:

\[
S(\ell\mid x)=\frac{1}{|\ell|}\sum_j \log p(t_j\mid x,t_{<j})
\]

and route by:

\[
M(x)=S(\text{SYMBOLIC}\mid x)-S(\text{LLM}\mid x).
\]

Mean normalization avoids mechanically favoring the shorter label. The
legacy single-token scorer remains available only for backward comparison.

Example tuning run:

```bash
daph-tune-steering \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --val-tasks data/protocol_v038/dev.jsonl \
  --vector-bundle vectors/tool_policy.json \
  --label-scoring sequence \
  --sequence-normalization mean \
  --output runs/tuning.jsonl
```

## Test and final-test discipline

Reserve the final split only after configuration is frozen:

```python
from pathlib import Path
from daph_learning.evaluation.protocol import reserve_final_test

record = reserve_final_test(
    Path("runs/final_test_access.json"),
    run_id="qualified-run-001",
    configuration_sha256="<sha256-of-frozen-config>",
)
```

The file is created atomically and cannot be reserved twice. Final evaluation
commands require `--protocol-purpose final_evaluation`,
`--configuration-frozen`, and the access-record SHA-256.

## Random-direction controls

The control runner evaluates every direction on the same tasks and scoring
configuration as the real direction:

```bash
daph-random-control \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --tasks data/protocol_v038/dev.jsonl \
  --vector vectors/tool_l20.npz \
  --n-random 999 \
  --label-scoring sequence \
  --sequence-normalization mean \
  --output runs/random_control.json
```

This command produces an empirical null summary. A result with fewer than 500
controls is marked protocol-ineligible.

## Steering safety

`SteeringSafetyLimits` can clamp the effective alpha when either bound would
be exceeded:

\[
\frac{\|\alpha v\|_2}{\|h\|_2}\le 0.65,\qquad
1-\cos(h,h+\alpha v)\le 0.25.
\]

The generation script also defaults reasoning-policy steering to a 16-step
cosine decay with a 0.1 floor. These are conservative engineering defaults,
not empirically optimal universal constants.

## Bounded symbolic executor

The arithmetic executor does not use `eval`, `exec`, unrestricted
`sympy.sympify`, or arbitrary calls. Its AST fallback accepts bounded integer
arithmetic and rejects names, attributes, subscripts, comprehensions, floats,
booleans, true division, and exponentiation. Resource limits constrain input
length, AST size/depth, integer digits, and result bits.

## Project layout

```text
src/daph_learning/
  autolearn/          bootstrap iterative loop
  cli/                wheel-safe command implementations
  data/               protocol generator and leakage audit
  evaluation/         typed verification, manifests, nulls, split guards
  execution/          bounded plans and symbolic executor
  routing/            full-sequence and legacy route scorers
  steering/           capture, extraction, hooks, safety clamps
extensions/
  daph_gdn2_repobrain_v1_11_1/
experiments/
  legacy_contaminated_v0_3_7/
```

## Testing

```bash
python -m pytest -q
```

The bundled GDN2/ExFusion extension has its own test suite:

```bash
cd extensions/daph_gdn2_repobrain_v1_11_1
python -m pytest -q
```

Some real-model tests skip when model downloads are not explicitly enabled.
Passing unit and integration tests establish mechanics, not scientific
effects.

## Documentation

- [CLAIMS.md](CLAIMS.md) — authoritative claim license
- [RELEASE_NOTES_V0_3_8.md](RELEASE_NOTES_V0_3_8.md) — upgrade details
- [CHANGELOG.md](CHANGELOG.md) — version history
- [docs/RUN_MANIFEST.md](docs/RUN_MANIFEST.md) — manifest schema
- [ROADMAP.md](ROADMAP.md) — deferred v0.3.9–v0.4.0 work

## Changelog

### v0.3.10

- **Architecture shift**: counterfactual compute-selection learner. Weighted
  centroid is now a baseline; the primary learner is a weighted soft-target
  logistic router (`P(S|h) = σ(w^T h + b)`) trained on continuous `ΔU`.
- **Primary metric: regret** (`max_a U(a) - U(π(x))`), not routing accuracy.
- **Weighted centroid**: `w_i = clip(|ΔU_i|·c_i)` with gap-threshold tie
  truncation.
- **Soft targets**: `q = σ(ΔU/τ)` retains continuous preference information.
- **Calibrated abstention**: `max(p, 1-p) < τ_conf → ABSTAIN`.
- **Calibration metrics**: Brier, ECE, reliability bins, selective-risk curve.
- **OOD detection**: Mahalanobis distance with regularized covariance.
- **Feature reduction**: PCA fitted on TRAIN only.
- **Causal intervention experiments**: dose-response `+v/0/-v` with
  direction reversal test.
- **KL/capability promotion gates**: utility gain + neutral KL + capability
  drop constraints.
- **Prioritized replay**, contextual bandit logging, DR interface.
- **Low-rank multi-vector controller** (experimental): `h' = h + Vα(h)`.
- **Immutable `ExperimentConfig`** (frozen, hashed).
- **67 new release-gate tests** (G2-G16).

### v0.3.9

- **Critical fix**: held-out promotion evaluation now uses the candidate
  steering vector for candidate routing and the incumbent vector for
  incumbent routing, via an injected `RoutePolicyFn`. The oracle Delta-U is
  used only for the training target, never for the held-out routing decision.
- Added `SteeringPolicyConfig`, `PolicyRouteDecision`, `CapturedActivation`,
  `CaptureResult` for proper provenance and task-ID-aligned capture.
- Fixed trust-region bootstrap, norm bounds, and fail-closed NaN/Inf handling.
- Fixed dataset hash lineage (separate training/dev hashes).
- Added capability regression gates to the promotion gate.
- CLI `daph-autolearn run` now supports `--engine counterfactual` (default).

### v0.3.8

- Repaired route scoring, typed verification, split leakage detection, and
  final-test access discipline.
- Replaced invalid five-control z-score reporting with empirical null
  inference and equal-task enforcement.
- Added steering clamps and mandatory default decay for reasoning generation.
- Made installed console commands wheel-safe.
- Corrected post-compression coefficient fitting in the GDN2/ExFusion
  extension.
- Quarantined the contaminated v0.3.7 experiment and narrowed all claims.

See [CHANGELOG.md](CHANGELOG.md) for the complete history.
