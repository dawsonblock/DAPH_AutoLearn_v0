# DAPH GDN2 + RepoBrain v1.10.1 Corrective Release

This corrective release addresses defects exposed by the v1.10 qualification audit without adding new research mechanisms.

## Corrected

1. **FSDP scalar-parameter incompatibility**
   - `_FactorHead.log_scale` is now stored as a one-element FP32 parameter (`shape == [1]`) rather than a zero-dimensional scalar.
   - This preserves broadcasting semantics while satisfying FSDP's parameter contract.
   - The FSDP qualification script now performs an explicit preflight check for zero-dimensional parameters.

2. **Distributed cleanup on failure**
   - `qualify_fsdp_repobrain.py` now tears down a process group in a `finally` block when it initialized that group, including exception paths.

3. **Batched adapter generation expansion**
   - Request-scoped adapter contexts now accept optional `batch_indices` mapping runtime rows to generated repository-adapter rows.
   - Common contiguous generation expansion (for example beam/return-sequence repeat-interleave) is handled automatically when the runtime batch is an integer multiple of the factor batch.
   - Reordered fan-out can be represented explicitly with `batch_indices`, avoiding adapter/request misalignment.

4. **Scientific Fisher default**
   - `empirical_fisher_diagonal()` now defaults to `per_sample=True`.
   - The batch-gradient-square approximation remains available only as an explicit opt-in (`per_sample=False`).

## Qualification status

These changes repair software and qualification contracts. They do **not** claim that GDN2 improves a pretrained Qwen checkpoint or that RepoBrain learns useful repository-to-adapter mappings. CUDA/Triton parity, real-checkpoint FSDP/ZeRO-3, multi-node, and the six-arm RepoBrain mechanism experiment must still be run on target hardware/data.
