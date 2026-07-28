# CHANGELOG_V0_3_10.md — DAPH AutoLearn v0.3.10-alpha

Date: 2026-07-28

Upgrades AutoLearn from a weighted steering-vector extractor into a counterfactual compute-selection learner. The weighted centroid remains as a transparent baseline; the primary policy learner is a weighted soft-target logistic router. Regret is the primary metric, not routing accuracy.

## Added
- `daph_learning/policy/` package: the learned policy system (types, confidence, weighting, centroid baseline, logistic router, abstention, calibration, OOD, features, config, regret, learner). This package is the new home for all policy-learning components and provides the module surface imported by the CLI and release-gate tests.
- Weighted soft-target logistic router (`WeightedLogisticRouter`, `soft_preference_target`, `weighted_policy_loss`) — the primary policy learner. It optimizes a weighted soft-target cross-entropy objective so that higher-utility routes receive proportionally more gradient signal.
- Utility-weighted centroid baseline with gap-threshold tie truncation (`WeightConfig`, `utility_weight`, `weighted_contrastive_mean`). The centroid remains as a transparent, non-learned baseline for comparison against the logistic router and truncates near-ties using a configurable gap threshold.
- Calibrated abstention (`Route` enum, `choose_route` with confidence threshold). The router can decline to act when its confidence falls below a threshold, returning an explicit abstain route instead of forcing a low-confidence decision.
- Calibration metrics: Brier score, ECE, reliability bins, selective-risk curve. These metrics quantify how well predicted confidences match empirical outcomes and characterize the risk-coverage tradeoff of the abstention policy.
- OOD detection via Mahalanobis distance (`MahalanobisOOD`). Inputs are flagged as out-of-distribution when their Mahalanobis distance from the training centroid exceeds a learned threshold, enabling conservative routing on unfamiliar inputs.
- PCA feature reduction fitted on TRAIN only (`PCAFeatureReducer`). Dimensionality is reduced using principal components estimated exclusively on the training split to prevent leakage from evaluation data.
- Causal intervention experiments: dose-response `+v/0/-v`, direction reversal test (`run_intervention_experiment`, `direction_reversal_test`). These experiments verify that interventions produce monotonic dose-response effects and that reversing the steering direction flips the outcome as expected.
- KL/capability promotion gates (`PromotionConstraints`, `evaluate_kl_capability_gate`). A candidate policy is only promoted if it satisfies both a KL divergence bound against the prior and a capability preservation check on held-out tasks.
- Paired bootstrap confidence intervals and promotion statistics. Bootstrap resampling over paired comparisons produces confidence intervals on regret and utility deltas used to decide promotion.
- Prioritized experience replay (`PrioritizedReplayBuffer`, `replay_priority`). Transitions are sampled with probability proportional to a priority weight, increasing the learner's exposure to high-regret decisions.
- Contextual bandit logging (`PolicyDecision`, `log_policy_decision`). Each routing decision is logged with its context, chosen route, predicted utility, and observed outcome to support offline evaluation and replay.
- Doubly-robust utility interface (`doubly_robust_utility`). Combines an inverse-propensity-weighted estimate with a learned reward model to produce a lower-variance, debiased estimate of route utility.
- Low-rank multi-vector controller (experimental: `LowRankSteeringController`, `orthogonality_loss`, `interference_matrix`). An experimental controller that maintains multiple low-rank steering directions with an orthogonality regularizer and an interference matrix measuring cross-direction coupling.
- Immutable experiment configuration (`ExperimentConfig`, frozen and hashed). Experiment configurations are frozen dataclasses whose hash is recorded with every run to guarantee reproducibility and detect configuration drift.
- Synthetic closed-loop environment (`make_synthetic_tasks`, `synthetic_execute_fn`). A self-contained synthetic environment generates tasks and executes routes so the learner can be evaluated end-to-end without external dependencies.
- Latent verifier experiment (`LatentVerifier`). A verifier operating in latent space is used to check whether a route's internal representation satisfies task-specific constraints before promotion.
- Integrated policy learner (`train_policy_learner`). Orchestrates feature reduction, router training, calibration, abstention, replay, and promotion gates into a single training loop.
- CLI: `daph-autolearn-policy` with train/evaluate/intervene/calibrate modes. The CLI exposes the full learner workflow as subcommands for training, evaluation, intervention experiments, and calibration analysis.
- 67 new release-gate tests (G2-G16). These tests gate the release by checking routing behavior, calibration, abstention, OOD handling, intervention effects, promotion constraints, and replay correctness.

## Changed
- Version bumped to 0.3.10. The package version string and metadata now reflect the v0.3.10-alpha release.
- README, CLAIMS, CHANGELOG updated for v0.3.10. Documentation was revised to describe the counterfactual compute-selection learner, the role of the centroid baseline, and the primacy of regret as the evaluation metric.
- pyproject.toml: added `daph-autolearn-policy` entry point. The new CLI is installed as a console script so it is available on `PATH` after package installation.

## Out of Scope (Section 50)
- Full SAE training, COCONUT, GDN2 architectural changes, model merging, TIES, DARE, multi-agent debate, large-scale PPO/GRPO, new external database infrastructure. These items are explicitly excluded from the v0.3.10-alpha scope and are deferred to later releases.
