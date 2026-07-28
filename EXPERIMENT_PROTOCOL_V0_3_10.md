# EXPERIMENT_PROTOCOL_V0_3_10.md — DAPH AutoLearn v0.3.10-alpha

## Overview

This protocol defines the experiment plan for DAPH AutoLearn v0.3.10-alpha. The core study is a four-way experiment crossing two steering modes (hard / soft) with two policy classes (centroid / logistic), yielding the cells: Hard Centroid, Soft Centroid, Hard Logistic, and Soft Logistic. Each cell is evaluated against the baseline matrix defined in Section 25. The four-way design isolates the contribution of (a) discrete vs. continuous activation steering and (b) geometric vs. probabilistic policy fitting, while holding the data, splits, and evaluation harness constant.

## Split Discipline (Section 34)

The data is partitioned into four disjoint splits, each with a single, non-overlapping purpose:

- **TRAIN**: Used only to fit policy parameters (vector weights, centroid positions, logistic coefficients). No model selection occurs here.
- **DEV**: Used to select the algorithm variant and hyperparameters (e.g., regularization strength, aggregation method, soft-steering temperature). Selection is performed by optimizing the DEV objective; TRAIN is never consulted for this.
- **CALIBRATION**: Used to select decision thresholds (abstain cutoffs, routing boundaries, confidence thresholds). Thresholds are fit here only; they must not be tuned on DEV or FINAL TEST.
- **FINAL TEST**: Used exactly once, after all selection and calibration is complete, to report the primary metrics. No iteration back into model selection or threshold tuning is permitted after FINAL TEST is consulted.

Any protocol run that touches FINAL TEST more than once is invalidated and must be re-run with a fresh FINAL TEST split.

## Baseline Matrix (Section 25)

All methods are evaluated on identical splits and identical metrics. The baselines are:

- **A. Base LLM** — No steering, no routing. Raw model completions serve as the lower bound.
- **B. Hand Router** — A human-designed routing heuristic using handcrafted rules. Non-learned control for the value of learned policy.
- **C. Unweighted Centroid** — Centroid policy in which each training template contributes equally (no utility weighting). Isolates the effect of utility weighting.
- **D. Weighted Centroid** — Centroid policy weighted by per-template utility. The primary centroid variant.
- **E. Weighted Logistic** — Logistic policy trained with utility-weighted examples. The primary logistic variant.
- **F. Soft Logistic** — Logistic policy with soft (continuous) steering rather than hard activation replacement. Tests the soft/hard axis for the logistic class.
- **G. Random Steering** — Steering vectors drawn at random, matched in norm to learned vectors. Null control for direction.
- **H. Incumbent** — The current best-performing prior AutoLearn version at the time of the run. Establishes the regression bar.
- **I. AutoLearn v0.3.10** — The full proposed system: learned utility-weighted policy, selected on DEV, calibrated on CALIBRATION, evaluated once on FINAL TEST.

## Causal Intervention Protocol (Sections 14-16)

To establish that learned steering directions cause the observed utility changes — and are not merely correlated with them — three interventions are performed:

1. **Dose-response (Section 14).** For each learned steering vector, apply scaled interventions with alpha in {-1.0, -0.5, 0.0, 0.5, 1.0}. Alpha = 0.0 is the no-intervention control. Utility should increase monotonically with |alpha| in the learned direction and decrease in the anti-direction (negative alpha). A monotonic or near-monotonic dose-response curve is required to claim a causal effect.

2. **Direction reversal test (Section 15).** Apply the steering vector with alpha = +1.0 and alpha = -1.0. A genuine causal direction should produce utility changes of opposite sign across the two conditions. If both signs move utility in the same direction, the vector is flagged as non-causal and excluded from the policy.

3. **Random matched direction control (Section 16).** For each learned vector, generate a random vector of identical norm and apply it with the same alpha schedule. The random vector should produce no significant utility change relative to alpha = 0.0. Learned vectors must outperform their random matched controls to be retained.

## KL/Capability Gate Protocol (Section 17)

Every candidate policy must pass the capability gate before final evaluation. The gate measures:

- **Neutral KL**: KL divergence between steered and unsteered model distributions on a neutral held-out task set. Neutral KL must remain below the configured ceiling.
- **Capability drops**: Performance change on standard capability benchmarks (e.g., MMLU, reasoning, code) when steering is active. No benchmark may drop by more than the configured tolerance.
- **Constraint enforcement**: All constraints (neutral KL ceiling, per-benchmark capability tolerance, abstain-rate floor) must be satisfied simultaneously. A policy that passes any single constraint but violates another fails the gate and is not eligible for FINAL TEST.

Policies that fail the gate are reported but excluded from the primary results table.

## Multi-Seed Protocol (Section 23)

To bound variance from stochastic training and evaluation, every experiment cell is run across at least 5 seeds. For each cell, report:

- Mean across seeds for every metric.
- Standard deviation across seeds for every metric.
- Per-seed values in an appendix or supplementary table.

A result is reported as significant only if the mean effect exceeds the standard deviation by the configured factor and the sign is consistent across all seeds.

## Family-Disjoint Evaluation (Section 24)

Evaluation uses the existing `split_family` infrastructure to guarantee that test task families are disjoint from training template families. Specifically:

- Each task is tagged with a family identifier.
- The split assigns entire families to TRAIN, DEV, CALIBRATION, or FINAL TEST; families are never split across partitions.
- Test families must not duplicate training templates. A family appearing in FINAL TEST must have no template overlap with any family in TRAIN, DEV, or CALIBRATION.
- Family leakage is checked programmatically before each run; any detected overlap aborts the run.

This ensures that reported metrics reflect generalization to unseen task families rather than memorization of seen templates.

## CLI Commands

The experiment is driven by the `daph-autolearn-policy` CLI. Canonical invocations:

```bash
# Train a policy on synthetic data
daph-autolearn-policy train --synthetic --output artifacts/result.json

# Evaluate a trained policy on the FINAL TEST split
daph-autolearn-policy evaluate --policy-file artifacts/policy.pt --test-tasks data/test.jsonl

# Run a causal intervention using a stored steering vector
daph-autolearn-policy intervene --vector-file artifacts/vector.npy

# Calibrate decision thresholds on the CALIBRATION split
daph-autolearn-policy calibrate --calibration-tasks data/cal.jsonl
```

## Required Report Table (Section 48)

Every experiment report must include the following table, populated for each method in the baseline matrix (A-I). Metrics are reported as mean +/- standard deviation across seeds. Arrows indicate the preferred direction.

| Method | Utility up | Regret down | Accuracy up | Brier down | ECE down | Abstain | Neutral KL down |
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

All cells must be filled; methods failing the KL/capability gate are marked as "gate-failed" and excluded from headline comparisons but retained in the table for transparency.
