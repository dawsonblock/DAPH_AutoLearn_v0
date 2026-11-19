# DAPH GDN2 + RepoBrain v1.5 — Deep Audit Remediation

## Critical fixes

1. **Real GDN2 cache contract**
   The uploaded upstream `GatedDeltaNet2.forward` returns
   `(o, None, past_key_values)`. v1.4 incorrectly treated tuple element 1 as
   recurrent state. v1.5 extracts tuple element 2 when present and preserves
   opaque framework cache objects.

2. **Reference recurrence readout**
   The upstream recurrent kernel documents `o = S^T q`. The old scaffold's
   reference implicitly read with `k`, preventing honest kernel-output
   qualification unless `q == k`. v1.5 accepts the true query tensor.

3. **Controller alias collision**
   Substring matching of the alias `ce` could classify names such as
   `kl_divergence` as validation loss. Short aliases now require token-level
   matches.

4. **Silent partial patch/application protection**
   Patching can now require all requested target suffixes, and factor
   application is strict by default. A treatment arm cannot silently omit one
   requested projection.

5. **Empirical provenance and split integrity**
   Empirical artifacts require a checkpoint, immutable dataset hash, and seed.
   Split manifests reject duplicates, blanks, non-string IDs, and overlap.

6. **Fisher behavior**
   Hard Fisher selection does not delete arbitrary zero-Fisher coordinates.
   Ranking is stable. Fisher accumulation can run in deterministic eval mode;
   per-sample empirical Fisher is distinguished from the batch-gradient-square
   approximation.

7. **Runtime safety checks**
   RepoPooler validates finite values and nonzero weight mass. LoRA factors,
   configuration values, vision feature contracts, and cache operations have
   stronger guards.

## Remaining empirical boundaries

- Real external GDN2 CUDA/Triton kernels still need hardware-backed numerical
  comparison with the corrected q/k/v reference.
- Repo2LoRA-Lite still needs real frozen-checkpoint training/evaluation.
- Fisher correction still has no promoted default; it must beat scalar scaling
  on validation data.
- No universal KL threshold is asserted.
- Multi-node/FSDP/DeepSpeed qualification is not claimed.
