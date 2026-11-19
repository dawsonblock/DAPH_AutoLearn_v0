# DAPH GDN2 + RepoBrain v1.8 Production Qualification

v1.8 hardens the qualification scaffold. It does **not** mark any hardware gate as passed unless the corresponding command is executed on target infrastructure.

## Corrections over v1.6

- Cache migration preserves integer/bool metadata dtypes instead of casting every tensor to model dtype.
- Generic beam reorder only touches tensors whose leading dimension matches the known cache batch size.
- GDN2 cache qualification now compares prefill+decode against a fresh full-sequence forward, not merely `cache is not None`.
- Hardware parity results record timing and peak CUDA allocation, validate shapes, and execute under `torch.inference_mode()`.
- Hugging Face repository embedding uses chunk micro-batches to avoid materializing all 4096-token chunks in one forward.
- RepoBrainTrainer now masks inferred-label padding, moves single-device inputs consistently, and rejects non-finite loss/gradient norm.
- Immutable split tooling accepts either an explicit split mapping or a raw repository list and partitions the latter deterministically by seed+repository hash.
- KL calibration requires a minimum number of safe validation adapters and records safe/total counts.
- Docker sandbox defaults to a read-only host mount copied into ephemeral tmpfs so tests can write without modifying the host repository.
- Cache checkpoints carry a schema version and can restore tensors to a requested device/dtype without corrupting integer tensors.
- FSDP and ZeRO-3 scripts now exercise an end-to-end graph: hypernetwork -> dynamic factors -> patched frozen base -> loss -> backward.
- Empirical artifact validation has a stricter hardware contract.
- Fisher correction can run in strict mode, raising if retained-energy or reconstruction gates fail.

## Required external qualification

The following remain empirical gates and are not represented as passed by the source tree:

1. CUDA/Triton GDN2 parity on each target GPU family.
2. Incremental GDN2 cache parity on each deployed kernel mode.
3. Real repository training and held-out validation/test results.
4. Validation-only controller calibration and Pareto selection.
5. Hardened sandbox image execution.
6. FSDP and ZeRO-3 multi-rank execution on target cluster software versions.
7. Restart/recovery testing using real opaque GDN2 cache codecs.
8. Final empirical artifact contract validation and release sign-off.
