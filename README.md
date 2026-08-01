# DAPH AutoLearn v0.3.10.5-alpha

<div align="center">

**Counterfactual compute-selection learning for auditable LLM tool-routing research.**

`v0.3.10.5-alpha` · Python ≥ 3.10 · MIT-style research software

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.3.10.5--alpha-orange.svg)](./CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-1269%2B-brightgreen.svg)](#testing)
[![Status](https://img.shields.io/badge/status-research%20alpha-lightgrey.svg)](./CLAIMS.md)

</div>

---

> **Gate A status: FAIL (daph_gate_a_real_007_harder).** This repository implements an
> experimental counterfactual routing learner. The latest Gate A run uses a
> **harder magnitude-decoupled benchmark** with **Qwen2.5-7B-Instruct**.
> The primary endpoint passes (P1 - best_fixed = 0.129, LCB = 0.074 > 0),
> and the **hidden-state contribution claim is SUPPORTED** (P1 - surface_only
> = 0.222, LCB = 0.173 > 0). The gate fails only on positive-group-fraction
> (50% < 60% threshold). Only a validated bundle under
> `artifacts/gate_a_qualified/` constitutes current Gate A evidence.
> The earlier real-model run (`daph_gate_a_real_001_failed`) is archived under
> `artifacts/legacy/` after failing its group-aware confidence-bound gate
> (LCB95% for P1−P0 = −0.041 < 0); it is retained for audit history only.
> No synthetic artifact is presented as Gate A qualification evidence.

DAPH AutoLearn connects **capability assessment**, **bounded symbolic
execution**, **residual activation steering**, and **outcome evaluation** into a
single auditable learning loop. Starting with v0.3.10, the system is a
**counterfactual compute-selection learner**: it learns a calibrated contextual
policy that routes each task to `SYMBOLIC`, `LLM`, or `ABSTAIN` by minimizing
**regret** against counterfactual utility, not by maximizing routing accuracy.

> **What "auditable" means here.** Every run emits a hashed, immutable
> `ExperimentConfig`; every promotion decision is gated by KL/capability
> constraints and paired bootstrap statistics; every headline claim is
> licensed in [`CLAIMS.md`](./CLAIMS.md) with an explicit status tag
> (`ESTABLISHED`, `BOOTSTRAP`, `PARTIAL`, `NOT YET`, `OUT OF SCOPE`).

---

## Table of contents

- [Why DAPH](#why-daph)
- [Key capabilities](#key-capabilities)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Command reference](#command-reference)
- [Experiment protocol](#experiment-protocol)
- [Safety and verification](#safety-and-verification)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Documentation](#documentation)
- [Licensed claims](#licensed-claims)
- [Changelog](#changelog)
- [Boundaries and caveats](#boundaries-and-caveats)

---

## Why DAPH

Most LLM tool-routing work reports accuracy on a single split and stops there.
DAPH is built around three commitments that make the resulting research
inspectable and falsifiable:

1. **Counterfactual supervision.** Both backends (`SYMBOLIC`, `LLM`) are executed
   on every training task, outcomes are independently verified, and the
   per-task utility gap `ΔU = U(S) − U(L)` becomes the learning signal. No
   oracle ever silently chooses the candidate action during held-out
   evaluation — that was the v0.3.8 defect, repaired in v0.3.9.
2. **Regret, not accuracy.** The primary objective is
   `regret = max_a U(a) − U(π(x))`. A policy that abstains on hard cases and
   routes easy cases correctly can beat an accuracy-maximizing policy on
   regret without over-claiming.
3. **Causal evidence, not correlation.** Every candidate steering direction
   must survive dose-response (`+v / 0 / −v`), direction-reversal, and
   matched-random controls before it enters the policy. KL and capability
   preservation gates must hold before any policy is promoted.

## Key capabilities

| Area | What v0.3.10 provides |
|---|---|
| **Policy learner** | Weighted soft-target logistic router `P(S|h) = σ(wᵀh + b)` trained on continuous `ΔU`, with utility-weighted centroid as a transparent baseline. |
| **Primary metric** | Per-task regret with paired-bootstrap mean and confidence intervals. |
| **Calibration** | Brier score, ECE, reliability bins, selective-risk curve. |
| **Abstention** | Calibrated `ABSTAIN` route: `max(p, 1−p) < τ_conf → ABSTAIN`. |
| **OOD detection** | Mahalanobis distance with regularized covariance, fit on TRAIN only. |
| **Feature reduction** | PCA fitted on TRAIN only (no leakage from evaluation splits). |
| **Causal interventions** | Dose-response, direction-reversal, matched-random control experiments. |
| **Promotion gate** | Neutral-KL ceiling + per-capability drop tolerance + abstain-rate floor. |
| **Replay & bandits** | Prioritized experience replay, contextual bandit logging, doubly-robust utility interface. |
| **Low-rank controller** | Experimental `h' = h + Vα(h)` with orthogonality loss and interference matrix. |
| **Route scoring** | Teacher-forced full-label log-probability scoring with mean normalization. |
| **Symbolic executor** | Bounded AST evaluator — no `eval`/`exec`/`sympify`, fail-closed on overflow. |
| **Dataset discipline** | Family-aware splits, exact/normalized/fingerprint leakage scans, SHA-256 manifests. |
| **Test access** | One-shot final-test access ledger; adaptive work cannot touch `test`/`final_test`. |
| **Random controls** | Empirical Monte Carlo p-value `(1+k)/(N+1)`; ≥500 controls required for headline eligibility. |
| **Steering safety** | Per-forward clamps on relative perturbation (≤ 0.65) and cosine shift (≤ 0.25). |
| **Provenance** | Immutable, hashed `ExperimentConfig`; run manifests with full lineage. |

## Installation

DAPH AutoLearn ships as a standard Python wheel. Python 3.10 or newer is
required.

```bash
# Core package (numpy + packaging only)
python -m pip install .

# With model-backed experiment support (torch, transformers, accelerate, safetensors)
python -m pip install '.[full]'

# Test dependencies only
python -m pip install '.[test]'
```

The following console commands become available on `PATH` after install:

| Command | Purpose |
|---|---|
| `daph-autolearn` | Run the iterative AutoLearn loop (`--engine counterfactual` default, `--engine legacy` deprecated). |
| `daph-autolearn-policy` | v0.3.10 policy learner CLI: `train`, `evaluate`, `intervene`, `calibrate`. |
| `daph-evaluate-routes` | Score routes on a task set with a frozen configuration. |
| `daph-build-oracles` | Build empirical utility oracles from executed backends. |
| `daph-tune-steering` | Tune a steering vector against full-sequence route scoring. |
| `daph-random-control` | Run matched random-direction controls with empirical null inference. |
| `daph-build-protocol-dataset` | Generate the 13-category leakage-audited protocol dataset. |

## Quick start

### 1. Build a leakage-audited protocol dataset

The generator produces 2,500 train / 1,000 dev / 500 calibration / 1,000
final-test tasks by default, with disjoint template slots across splits.

```bash
daph-build-protocol-dataset \
  --output-dir data/protocol_v038 \
  --seed 1337
```

The output directory contains the four JSONL splits, a leakage report, and a
dataset manifest with SHA-256 hashes. This is a **synthetic protocol
benchmark** — it is not automatically an OOD or scientifically qualified
benchmark.

### 2. Train a policy on the synthetic closed-loop environment

```bash
daph-autolearn-policy train \
  --synthetic \
  --policy logistic \
  --soft-targets \
  --target-temperature 1.0 \
  --weight-mode gap \
  --gap-threshold 0.0 \
  --confidence-threshold 0.70 \
  --seed 0 \
  --output artifacts/policy_train.json
```

The CLI prints dev metrics including mean candidate/incumbent utility, mean
candidate/incumbent regret, Brier score, and ECE.

### 3. Run a causal intervention experiment

```bash
daph-autolearn-policy intervene \
  --vector-file artifacts/vector.npy \
  --output artifacts/intervention.json
```

This runs the `+v / 0 / −v` dose-response schedule and prints a
direction-reversal summary. A genuine causal direction should produce
opposite-sign utility changes for `+v` and `−v`.

### 4. Tune a steering vector against full-sequence route scoring

Protocol runs score complete route labels:

```
S(ℓ | x) = (1/|ℓ|) Σ_j log p(t_j | x, t_<j)
M(x)     = S(SYMBOLIC | x) − S(LLM | x)
```

Mean normalization avoids mechanically favoring the shorter label.

```bash
daph-tune-steering \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --val-tasks data/protocol_v038/dev.jsonl \
  --vector-bundle vectors/tool_policy.json \
  --label-scoring sequence \
  --sequence-normalization mean \
  --output runs/tuning.jsonl
```

### 5. Reserve the final test (one-shot, frozen configuration)

```python
from pathlib import Path
from daph_learning.evaluation.protocol import reserve_final_test

record = reserve_final_test(
    Path("runs/final_test_access.json"),
    run_id="qualified-run-001",
    configuration_sha256="<sha256-of-frozen-config>",
)
```

The access record is created atomically and cannot be reserved twice. Final
evaluation commands require `--protocol-purpose final_evaluation`,
`--configuration-frozen`, and the access-record SHA-256.

### 6. Run matched random-direction controls

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

The runner reports the empirical null summary
`p = (1 + #{T_null ≥ T_real}) / (N + 1)`. Results with fewer than 500
controls are marked protocol-ineligible.

## Architecture

v0.3.10 upgrades AutoLearn from a "weighted steering-vector extractor" into a
**counterfactual compute-selection learner**. The weighted centroid remains as
a **transparent baseline**; the primary learner is a calibrated contextual
policy model.

```
state / activation
    → action policy
    → computation choice (SYMBOLIC | LLM | ABSTAIN)
    → verified outcome
    → counterfactual utility / regret
    → policy improvement
```

### Architecture progression (P0 → P10)

```
P0  Fix actual candidate/incumbent evaluation          [v0.3.9, done]
 ↓
P1  Counterfactual utility + regret                    [v0.3.9 + v0.3.10]
 ↓
P2  Weighted centroid baseline                         [v0.3.10]
 ↓
P3  Weighted soft-target logistic router               [v0.3.10, CRITICAL]
 ↓
P4  Calibration + abstention                           [v0.3.10]
 ↓
P5  OOD detection                                      [v0.3.10]
 ↓
P6  +v / -v causal intervention experiments            [v0.3.10]
 ↓
P7  KL + capability-preservation promotion gates       [v0.3.10]
 ↓
P8  Prioritized replay                                 [v0.3.10]
 ↓
P9  Low-rank multi-vector controller                   [v0.3.10, experimental]
 ↓
P10 Partial-feedback / off-policy learning             [v0.3.10 interface only]
```

### End-to-end data flow

```
1.  Tasks arrive (train / dev / calibration / final)
2.  Execute BOTH backends counterfactually (execute_both_backends)
3.  Verify outcomes independently
4.  Compute U(S), U(L), ΔU (compute_backend_utility)
5.  Sample weight w_i = clip(|ΔU_i| · c_i) with gap truncation
6.  Optional feature reduction (PCA on TRAIN only)
7.  Fit OOD detector (Mahalanobis on TRAIN only)
8.  Train weighted logistic router p = σ(wᵀh + b) against q = σ(ΔU/τ)
9.  Evaluate on dev: candidate vs incumbent, compute regret
10. Calibrate abstention threshold on calibration split
11. Run causal intervention experiments (+v/0/−v dose-response)
12. KL/capability promotion gate
13. Final test evaluation (once, frozen)
```

### Split discipline

| Split | May fit | May select |
|---|---|---|
| **TRAIN** | centroids, logistic weights, PCA, covariance, feature selectors | — |
| **DEV** | — | algorithm, regularization, layer, temperature, steering params |
| **CALIBRATION** | — | probability calibration, abstention threshold, OOD threshold |
| **FINAL TEST** | — | used once for final frozen evaluation |

Any protocol run that touches FINAL TEST more than once is invalidated and
must be re-run with a fresh FINAL TEST split.

See [`ARCHITECTURE_V0_3_10.md`](./ARCHITECTURE_V0_3_10.md) for the full
component map and design principles.

## Command reference

### `daph-autolearn`

Runs the iterative AutoLearn loop. v0.3.9 introduced an explicit engine
selector; the counterfactual engine is the default and the legacy engine is
retained only for regression comparison.

```bash
daph-autolearn run \
  --engine counterfactual \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train-tasks data/protocol_v038/train.jsonl \
  --val-tasks   data/protocol_v038/dev.jsonl \
  --output      runs/autolearn.jsonl
```

| Flag | Values | Notes |
|---|---|---|
| `--engine` | `counterfactual` (default), `legacy` (deprecated) | `legacy` will be removed after feature parity is verified. |
| `--model` | HuggingFace model ID | Required for real-model runs. |
| `--train-tasks`, `--val-tasks` | JSONL paths | Family-aware splits from the protocol generator. |
| `--output` | path | JSONL stream of loop iterations. |

### `daph-autolearn-policy`

The v0.3.10 policy learner CLI. Four subcommands:

```bash
daph-autolearn-policy train      --synthetic --output artifacts/policy.json
daph-autolearn-policy evaluate   --policy-file artifacts/policy.pt --test-tasks data/test.jsonl
daph-autolearn-policy intervene  --vector-file artifacts/vector.npy
daph-autolearn-policy calibrate  --calibration-tasks data/cal.jsonl
```

Common policy flags:

| Flag | Default | Purpose |
|---|---|---|
| `--policy` | `logistic` | `centroid` (baseline) or `logistic` (primary learner). |
| `--soft-targets` / `--no-soft-targets` | on | Use `q = σ(ΔU/τ)` (default) or hard labels for ablation. |
| `--target-temperature` | `1.0` | Soft-target temperature τ. Smaller ⇒ sharper preference. |
| `--weight-mode` | `gap` | `gap` (zero weight below threshold) or `snr`. |
| `--gap-threshold` | `0.0` | Tasks with `|ΔU| ≤ threshold` get zero weight. |
| `--layer` | `24` | Transformer layer index. |
| `--alpha` | `1.0` | Steering alpha. |
| `--confidence-threshold` | `0.70` | Abstention threshold τ_conf. |
| `--ood-threshold` | `∞` | OOD abstention threshold. |
| `--seed` | `0` | Random seed. |
| `--model`, `--model-revision`, `--tokenizer-revision` | — | Provenance metadata for real-model runs. |

### `daph-build-protocol-dataset`

```bash
daph-build-protocol-dataset --output-dir data/protocol_v038 --seed 1337
```

### `daph-tune-steering`

```bash
daph-tune-steering \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --val-tasks data/protocol_v038/dev.jsonl \
  --vector-bundle vectors/tool_policy.json \
  --label-scoring sequence \
  --sequence-normalization mean \
  --output runs/tuning.jsonl
```

### `daph-random-control`

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

## Experiment protocol

The v0.3.10 experiment is a **four-way design** crossing two steering modes
(hard / soft) with two policy classes (centroid / logistic), yielding the
cells: Hard Centroid, Soft Centroid, Hard Logistic, Soft Logistic. Each cell
is evaluated against the baseline matrix (A–I) defined in
[`EXPERIMENT_PROTOCOL_V0_3_10.md`](./EXPERIMENT_PROTOCOL_V0_3_10.md).

### Baseline matrix

| ID | Method | Role |
|---|---|---|
| A | Base LLM | No steering, no routing. Lower bound. |
| B | Hand Router | Human-designed heuristic. Non-learned control. |
| C | Unweighted Centroid | Isolates the effect of utility weighting. |
| D | Weighted Centroid | Primary centroid variant. |
| E | Weighted Logistic | Primary logistic variant. |
| F | Soft Logistic | Tests the soft/hard axis for the logistic class. |
| G | Random Steering | Matched-norm random vectors. Null control for direction. |
| H | Incumbent | Prior best AutoLearn version. Regression bar. |
| I | AutoLearn v0.3.10 | The full proposed system. |

### Causal intervention protocol

To establish that learned steering directions **cause** the observed utility
changes (not merely correlate with them), three interventions are performed:

1. **Dose-response.** Apply scaled interventions with
   `α ∈ {−1.0, −0.5, 0.0, 0.5, 1.0}`. `α = 0.0` is the no-intervention
   control. Utility should increase monotonically with `|α|` in the learned
   direction and decrease in the anti-direction.
2. **Direction reversal.** Apply `α = +1.0` and `α = −1.0`. A genuine causal
   direction should produce utility changes of opposite sign. Same-sign
   movements flag the vector as non-causal and exclude it from the policy.
3. **Matched random control.** Generate a random vector of identical norm and
   apply the same `α` schedule. The random vector should produce no
   significant utility change relative to `α = 0.0`. Learned vectors must
   outperform their matched random controls to be retained.

### KL / capability gate

Every candidate policy must pass the capability gate before final
evaluation:

- **Neutral KL** — KL divergence between steered and unsteered model
  distributions on a neutral held-out task set, bounded by a configured
  ceiling.
- **Capability drops** — performance change on standard capability benchmarks
  when steering is active. No benchmark may drop by more than the configured
  tolerance.
- **Constraint enforcement** — all constraints must hold simultaneously. A
  policy that passes one constraint but violates another fails the gate and
  is not eligible for FINAL TEST.

### Multi-seed protocol

Every experiment cell is run across at least 5 seeds. For each cell, report
mean, standard deviation, and per-seed values. A result is reported as
significant only if the mean effect exceeds the standard deviation by the
configured factor and the sign is consistent across all seeds.

### Required report table

Every experiment report must include the following table, populated for each
method in the baseline matrix (A–I). Metrics are reported as
mean ± standard deviation across seeds.

| Method | Utility ↑ | Regret ↓ | Accuracy ↑ | Brier ↓ | ECE ↓ | Abstain | Neutral KL ↓ |
|---|---|---|---|---|---|---|---|
| A. Base LLM | | | | | | | |
| B. Hand Router | | | | | | | |
| C. Unweighted Centroid | | | | | | | |
| D. Weighted Centroid | | | | | | | |
| E. Weighted Logistic | | | | | | | |
| F. Soft Logistic | | | | | | | |
| G. Random Steering | | | | | | | |
| H. Incumbent | | | | | | | |
| I. AutoLearn v0.3.10 | | | | | | | |

Methods failing the KL/capability gate are marked `gate-failed` and excluded
from headline comparisons but retained in the table for transparency.

## Safety and verification

### Bounded symbolic executor

The arithmetic executor does **not** use `eval`, `exec`, unrestricted
`sympy.sympify`, or arbitrary calls. Its AST fallback accepts bounded integer
arithmetic and rejects names, attributes, subscripts, comprehensions, floats,
booleans, true division, and exponentiation. Resource limits constrain input
length, AST size/depth, integer digits, and result bits.

### Steering safety

`SteeringSafetyLimits` clamps the effective alpha when either bound would be
exceeded:

```
‖αv‖₂ / ‖h‖₂ ≤ 0.65,        1 − cos(h, h + αv) ≤ 0.25
```

The generation script also defaults reasoning-policy steering to a 16-step
cosine decay with a 0.1 floor. These are conservative engineering defaults,
not empirically optimal universal constants.

### Typed verification

Typed verification accepts only an exact integer or `FINAL: <integer>` for
numeric tasks. Substring matches such as expected `12` in output `312` are
rejected.

### Test access discipline

Adaptive work cannot use `test` or `final_test`. Final evaluation requires
frozen configuration and a one-shot access ledger. The access file is created
atomically and cannot be reserved twice.

## Project layout

```text
src/daph_learning/
  autolearn/          bootstrap iterative loop (counterfactual + legacy engines)
  bandit/             contextual bandit logging, IPS, doubly-robust utility
  cli/                wheel-safe command implementations
    commands/         per-command argument parsing and orchestration
  data/               protocol generator and leakage audit
  environment.py      task environment interface
  environment_synthetic.py  synthetic closed-loop environment
  evaluation/         typed verification, manifests, nulls, split guards
  execution/          bounded plans and symbolic executor
  experiments/        experiment drivers
  interventions/      causal intervention experiments, KL/capability gate
  latent_verifier.py  latent-space verifier experiment
  lowrank/            experimental low-rank multi-vector controller
  memory/             episodic, procedural, vector-library memory
  policy/             the learned policy system
    types.py          BackendOutcome, CounterfactualExperience, Route, ...
    confidence.py     OutcomeConfidence with explicit provenance
    weighting.py      utility_weight, WeightConfig, gap-threshold truncation
    centroid.py       weighted / unweighted contrastive mean (BASELINE)
    logistic.py       WeightedLogisticRouter, soft_preference_target
    abstention.py     choose_route with calibrated confidence threshold
    calibration.py    Brier, ECE, reliability bins, selective-risk curve
    ood.py            MahalanobisOOD with regularized covariance
    features.py       PCAFeatureReducer fitted on TRAIN only
    config.py         ExperimentConfig (frozen, hashed)
    regret.py         per_task_regret, paired_bootstrap_mean_ci
    learner.py        train_policy_learner — integrated pipeline
  replay/             PrioritizedReplayBuffer, replay_priority
  routing/            full-sequence and legacy route scorers
  steering/           capture, extraction, hooks, safety clamps
  telemetry.py        run telemetry and provenance
  tools/              tool interface definitions
extensions/
  daph_gdn2_repobrain_v1_11_1/   bundled GDN2/ExFusion extension
experiments/
  legacy_contaminated_v0_3_7/    archived, not licensed as headline evidence
docs/
  EXPERIMENT_PLAN.md
  RUN_MANIFEST.md
```

## Testing

```bash
# Full repository test suite
python -m pytest -q

# Bundled GDN2/ExFusion extension
cd extensions/daph_gdn2_repobrain_v1_11_1
python -m pytest -q
```

The main repository contains 1229+ collected tests in the release build,
covering symbolic-executor safety, routing, full-sequence scoring, typed
verification, steering hooks and clamps, leakage checks, protocol guards,
manifests, command packaging, empirical null calculations, regression
behavior, the v0.3.9 counterfactual outcome-semantics / frozen-utility /
immutable-experience-record / promotion-gate tests, and the 67 v0.3.10
release-gate tests (G2–G16) for routing, calibration, abstention, OOD
handling, intervention effects, promotion constraints, and replay.

Some real-model tests skip when model downloads are not explicitly enabled.
Passing unit and integration tests establish **mechanics**, not scientific
effects — see [`CLAIMS.md`](./CLAIMS.md) §6.

## Documentation

| Document | Purpose |
|---|---|
| [`CLAIMS.md`](./CLAIMS.md) | Authoritative claim license with status tags. |
| [`ARCHITECTURE_V0_3_10.md`](./ARCHITECTURE_V0_3_10.md) | Component map, data flow, design principles. |
| [`EXPERIMENT_PROTOCOL_V0_3_10.md`](./EXPERIMENT_PROTOCOL_V0_3_10.md) | Four-way experiment design, baseline matrix, intervention protocol. |
| [`AUTOLEARN_MATH.md`](./AUTOLEARN_MATH.md) | Formal definitions of utility, regret, soft targets, calibration. |
| [`REAL_MODEL_QUALIFICATION.md`](./REAL_MODEL_QUALIFICATION.md) | Path to qualified real-model results. |
| [`CHANGELOG.md`](./CHANGELOG.md) | Full version history. |
| [`CHANGELOG_V0_3_10.md`](./CHANGELOG_V0_3_10.md) | v0.3.10 changes only. |
| [`CHANGELOG_V0_3_9.md`](./CHANGELOG_V0_3_9.md) | v0.3.9 changes only. |
| [`ROADMAP.md`](./ROADMAP.md) | Deferred v0.3.11–v0.4.0 work. |
| [`docs/RUN_MANIFEST.md`](./docs/RUN_MANIFEST.md) | Run manifest schema. |
| [`docs/EXPERIMENT_PLAN.md`](./docs/EXPERIMENT_PLAN.md) | Experiment planning template. |
| [`AUDIT_REPORT_V0_3_9.md`](./AUDIT_REPORT_V0_3_9.md) | Independent audit of the v0.3.9 causal-chain repair. |
| [`TEST_REPORT_V0_3_10.md`](./TEST_REPORT_V0_3_10.md) | v0.3.10 release-gate test report. |
| [`release_gates.json`](./release_gates.json) | Machine-readable release-gate results (source-tree hashed). |
| [`experiment_results.json`](./experiment_results.json) | Machine-readable experiment results (source-tree hashed). |

## Licensed claims

DAPH AutoLearn is research software. Tests establish that a mechanism is
implemented and behaves as specified on covered inputs. They do **not** by
themselves establish a reproducible real-model effect, out-of-distribution
generalization, or production readiness. The full claim boundary is in
[`CLAIMS.md`](./CLAIMS.md); the headline summary:

| Claim | Status |
|---|---|
| Route scoring, typed verification, leakage audits, manifests, wheel-safe CLI | `ESTABLISHED` (engineering) |
| Synthetic 13-category protocol benchmark | `BOOTSTRAP` |
| Multi-token route scoring with mean normalization | `ESTABLISHED` (engineering) |
| Matched random-direction controls with empirical p-value | `ESTABLISHED` (engineering) — `NOT YET` scientifically qualified |
| Split discipline, provenance, final-test access ledger | `ESTABLISHED` (engineering) |
| GDN2/ExFusion compressed basis extension | `ESTABLISHED` (engineering) — `NOT YET` empirically qualified |
| Counterfactual AutoLearn learning loop (v0.3.9 repair) | `ESTABLISHED` (mechanics) |
| Weighted soft-target logistic router, regret, calibration, abstention, OOD, interventions, gates | `ESTABLISHED` (mechanics, v0.3.10) |
| Real-model causal steering benefit | `NOT YET` |
| Cross-model / cross-tokenizer generalization | `NOT YET` |
| Out-of-distribution benefit | `NOT YET` |
| Autonomous policy improvement on real-model held-out tasks | `NOT YET` |
| Latent Memory v0.5.2 / COCONUT | `OUT OF SCOPE` |
| Production readiness | `NOT YET` |

The historical v0.3.7 Qwen2.5 experiment is retained only under
`experiments/legacy_contaminated_v0_3_7/` for auditability. It used unequal
task sets for the real and random directions, only five random controls,
test-set tuning, and leaky splits. **Its reported z-score is not a licensed
result.**

## Changelog

### v0.3.10.5-alpha

- **Gate A statistical correctness repair** — the prior Gate A PASS
  (daph_gate_a_real_002) is **INVALIDATED**. Primary CIs were placeholders,
  P1-minus-sham used the wrong interval, and P1 utility was computed from
  soft probabilities rather than hard routing actions.
- **Hard routing**: P1 utility uses selected actions, not probability-weighted
  surrogates.
- **Real group bootstrap**: 20,000 iterations with group-weighted estimand.
- **P1-minus-sham**: Nested bootstrap of the actual difference distribution.
- **Frozen policy evaluation**: Final stage loads, hashes, and executes the
  frozen policy artifact — no retraining.
- **Operational calibration**: Frozen thresholds applied to raw probabilities.
- **Precondition gates**: Checked before statistical gates (NOT_EVALUABLE).
- **Prediction artifacts**: final_predictions, final_task_metrics, sham_predictions.
- **Independent validator**: Recomputes all metrics from task-level records.
- **Portable pointer**: Relative paths only, no machine-local paths.
  `daph_gate_a_real_001_failed` under `artifacts/legacy/` with
  `LEGACY_NOTICE.md`; new experiment ID `daph_gate_a_real_002`.
- `eval()` removed from symbolic execution paths; bounded AST evaluator
  `safe_eval_int_expr` extended with explicitly enumerated permitted nodes.
- Canonical `FINAL_ANSWER: <integer>` verifier
  (`parse_canonical_integer_answer`); qualification fails closed on
  ambiguity; legacy permissive extraction can no longer award credit.
- New `artifacts/` layout + `validate_artifact_bundle()` recursive validator
  + `tests/test_artifact_integrity.py` CI gate.
- **Gate A status: NOT YET REQUALIFIED.** No new full real-model run
  executed; no synthetic artifact presented as qualification evidence.

### v0.3.10.3.2-alpha

- **Qualification integrity release** (not architecture expansion). Mission:
  prove or falsify that AutoLearn can choose the better computation between
  two available backends for individual tasks that share the same family and
  subtype. See `CHANGELOG_V0_3_10_3_2.md`.
- **Within-subtype crossover** (Sections 12-17): at least 1 subtype must
  individually contain both symbolic-preferred and LLM-preferred examples,
  with the optimal backend emerging from actual executed utilities — not
  taxonomy classification. The benchmark now has 8 subtypes (A-H) including
  unit conversion (G) and number theory (H).
- **Canonical source-tree hash** (Section 2): one
  `compute_source_tree_sha256()` implementation; all components delegate to
  it. Full 64-char SHA-256 in artifacts.
- **Artifact integrity** (Sections 3-4): current artifacts must match current
  source tree; stale artifacts archived to `artifacts/archive/`.
- **32-gate framework executed** (Sections 48-50): G01-G32 actually run
  against the current source tree.
- **Real CLI paths completed** (Sections 22-27): evaluate, calibrate,
  intervene use real frozen pipelines with zero fitting on final.
- **Real steering utility** (Sections 28-31): `ΔU(α)` measured via executed
  backend utility, not symbolic probability. Beneficial/harmful flip
  analysis; matched random controls.

### v0.3.10.3.1-alpha

- **Qualification repair release** (not architecture expansion). Mission:
  make the evidence trustworthy. See `CHANGELOG_V0_3_10_3_1.md`.
- **Within-family crossover benchmark** (Section 10-12): structured +
  natural-language mathematics where both symbolic and LLM win on different
  instances inside the same family. The central scientific question is
  whether AutoLearn can choose the better computation for an individual
  task, not merely classify the task family.
- **Steering optimizes verified utility** (Section 21-25): `ΔU(α)` not
  `P(symbolic)`. Per-alpha beneficial/harmful flip analysis; random
  direction controls; neutral KL release gate.
- **Frozen evaluation** (Section 14-19): `ExperimentStage` access control,
  final-test access ledger, zero fitting on final, real
  evaluate/calibrate/intervene CLIs.
- **Source-hash enforcement** (Section 32-35): current artifacts must share
  `source_tree_sha256`; mismatched artifacts archived and excluded from
  headline claims.

### v0.3.10.3-alpha

- **P0 repairs**: verified symbolic output (not just execution success),
  expanded `BackendOutcome` with execution/error semantics, fixed
  confidence semantics (unsupported ≠ zero confidence), removed silent
  zero-class-weight fallback, true unweighted ablation (w=1), ΔU-based
  calibration targets, disjoint word pools across splits, deferred final
  set execution until freeze, `max_vector_norm` only shrinks, one
  canonical `backend_utility()` function, verified steering utility,
  G16/G21/G23/G25 literally test their names, tie-aware routing accuracy.
- **Real-model loop completion**: actual symbolic + LLM backend
  execution, real verification, real utility, real regret. No more
  `symbolic_correct` / `llm_correct` placeholder labels.
- **Release-gate integrity**: G10 is now a true superiority gate
  (multi-seed); G16-G25 actually execute what they claim.
- **Calibration repair**: uses `policy.predict_proba`, not `p=0.5`.
- **Near-tie env redesign**: weighting now has a real expected
  advantage (decisive vs ambiguous with nuisance direction).
- **Real steering utility test**: dose-response with verified utility,
  route-flip analysis, random controls.
- **Evidence labels**: finer categories
  (latent/behavioral/utility intervention).
- **Source-tree hash + pytest collection hash** in all artifacts.

### v0.3.10.1-alpha

- **Correctness repair pass**: focused on implementation integrity and
  scientific validity. The release answers whether AutoLearn reduces
  held-out regret and improves held-out utility on non-trivial tasks.
- **P0-1 / G1**: `soft_targets: bool` replaced by explicit
  `TargetMode.SOFT` | `TargetMode.HARD` with validity mask; hard-mode
  ties ignored, not coerced to 0.5.
- **P0-2 / G2**: `weight_mode` made real — four modes
  (`UNIFORM`, `ABSOLUTE_GAP`, `CLIPPED_GAP`, `SNR`) via
  `compute_weight(...)`. Clean break: old `"gap"` string raises.
- **P0-3 / G3**: `policy_type` made real — `centroid` / `logistic` /
  `mlp_experimental` select different implementations.
- **P0-4 / G4**: held-out evaluation fails closed on missing
  `utility_fn`; no zero-utility synthesis.
- **P0-5/6 / G5**: strict task-ID alignment — `FeatureRecord` +
  `join_by_task_id`; no silent zip truncation.
- **P0-7 / G6**: `weighted_mean` rejects negative weights, NaN, Inf,
  all-zero effective weights.
- **P0-8 / G7**: calibration math fixed — `preference_brier_soft` and
  `action_confidence_ece` replace the wrong soft-label ECE.
- **P0-9 / G8**: dev early stopping selectable
  (`dev_loss` | `dev_regret` | `dev_utility`), default `dev_regret`.
- **P10 / G9-G13**: synthetic benchmark redesigned — four environments
  (linear, near-tie/heteroskedastic, multimodal, XOR) + random control.
- **P11/G14, P22/G21, P24/G22**: comparative gates literally compare;
  promotion uses real policy decisions; capability + neutral KL gates.
- **P13-20 / G16-G20**: real intervention pipeline + real
  train/evaluate/calibrate CLI + OOD threshold calibration + PCA safety.
- **P30 / G13**: small MLP router for nonlinear diagnostic.
- **P43 / G24**: version unified to `0.3.10.3-alpha` across all
  surfaces.

### v0.3.10

- **Architecture shift**: counterfactual compute-selection learner. The
  weighted centroid is now a baseline; the primary learner is a weighted
  soft-target logistic router `P(S|h) = σ(wᵀh + b)` trained on continuous
  `ΔU`.
- **Primary metric: regret** (`max_a U(a) − U(π(x))`), not routing accuracy.
- **Weighted centroid**: `w_i = clip(|ΔU_i| · c_i)` with gap-threshold tie
  truncation.
- **Soft targets**: `q = σ(ΔU/τ)` retains continuous preference information.
- **Calibrated abstention**: `max(p, 1−p) < τ_conf → ABSTAIN`.
- **Calibration metrics**: Brier, ECE, reliability bins, selective-risk curve.
- **OOD detection**: Mahalanobis distance with regularized covariance.
- **Feature reduction**: PCA fitted on TRAIN only.
- **Causal intervention experiments**: dose-response `+v/0/−v` with direction
  reversal test.
- **KL/capability promotion gates**: utility gain + neutral KL + capability
  drop constraints.
- **Prioritized replay**, contextual bandit logging, doubly-robust interface.
- **Low-rank multi-vector controller** (experimental): `h' = h + Vα(h)`.
- **Immutable `ExperimentConfig`** (frozen, hashed).
- **67 new release-gate tests** (G2–G16).
- New `daph-autolearn-policy` CLI with `train` / `evaluate` / `intervene` /
  `calibrate` modes.

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

See [`CHANGELOG.md`](./CHANGELOG.md) for the complete history.

## Boundaries and caveats

DAPH AutoLearn v0.3.10-alpha does **not** contain or claim:

- a scientifically qualified real-model steering benefit on any model family;
- cross-model or cross-tokenizer generalization;
- out-of-distribution benefit;
- autonomous policy improvement on real-model held-out tasks;
- a Latent Memory v0.5.2 training system;
- COCONUT or any external continuous-thought memory bank;
- production readiness.

The v0.3.10 release explicitly defers (Section 50 of the changelog): full SAE
training, COCONUT, GDN2 architectural changes, model merging, TIES, DARE,
multi-agent debate, large-scale PPO/GRPO, and new external database
infrastructure.

If you find a bug or a claim that is not licensed by [`CLAIMS.md`](./CLAIMS.md),
please open an issue with the smallest reproducer you can construct.
