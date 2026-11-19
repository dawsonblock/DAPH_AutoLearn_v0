# DAPH GDN2 + RepoBrain v1.4 — Empirical Qualification Remediation

v1.4 removes conclusions that were not supported by real checkpoint experiments.

## Removed assumptions

- No universal `KL=0.035` safety boundary.
- No default 12.5%/15%/20% Fisher pruning recommendation.
- No controller action may both say `ACCEPT` and silently transform an adapter.
- Synthetic tensor calculations are explicitly tagged `data_origin=synthetic`.

## Added

- Qwen checkpoint architecture adapter/registry.
- Exact-top-k hard Fisher correction (no quantile tie ambiguity).
- Per-tensor Fisher normalization.
- Soft Fisher attenuation.
- SVD retained-energy and reconstruction-error metrics.
- Evidence-first controller with `EVALUATE_CORRECTIONS` state.
- Mandatory revalidation flag for transformed candidates.
- Experiment provenance contract separating empirical and synthetic outputs.
- Empirical Qwen v1.4 config with KL threshold unset until calibrated.

## Qualification order

1. Prove dynamic RepoLoRA == dense delta application.
2. Train Repo2LoRA-Lite on immutable train/validation repository splits.
3. Measure real repository gain and generic capability retention.
4. Measure real KL and determine whether it predicts regression.
5. Compare scalar scaling against hard/soft Fisher corrections.
6. Select only from validation Pareto frontier.
7. Freeze policy and run untouched test repositories once.
8. Qualify external GDN2 kernels independently from RepoBrain.

A passing unit test validates software behavior, not model efficacy.
