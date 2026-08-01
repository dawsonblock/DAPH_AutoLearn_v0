# DAPH GDN2 + RepoBrain v0.1.3 remediation

This revision closes seven edge cases found in the v0.1.2 audit.

## Repairs

1. `RepoEvolutionState` normalizes unbatched `[D]` embeddings/state to `[1,D]` for `GRUCell`, preserves batched execution, and validates dimensional consistency.
2. `ValidationLossProbe` now places inputs on the model device and masks padding when labels are inferred from `input_ids`.
3. `LogitKLDriftProbe` independently places inputs for base/candidate models, transfers candidate logits only after forward execution, and excludes padding tokens from KL aggregation.
4. `patch_repo_lora_linears` raises on zero matches instead of returning an apparently valid empty patch set.
5. Empirical Fisher tracks gradient occurrence counts per parameter and normalizes each diagonal by the number of estimator steps in which that parameter actually received a gradient. The counts are exposed as `FisherDiagonal.parameter_samples`.
6. Truncated SVD factor balancing floors singular values at `1e-12` before square root to remove the explicit infinite derivative at zero.
7. `RepoPooler` performs weighted mean/max aggregation in FP32 and casts the final pooled vector back to the input dtype.

## Additional hardening

The probe changes go slightly beyond the audit: KL aggregation now honors `attention_mask`, and validation metadata reports actual valid token counts instead of the number of batches.

## Qualification boundary

Passing these unit tests does not establish heterogeneous-cluster production readiness. FSDP/DeepSpeed/Accelerate sharded placement, tensor parallelism, real CUDA GDN2 kernels, checkpoint-family patch maps, BF16/FP16 training, and distributed failure recovery still require hardware-backed qualification.
