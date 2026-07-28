# DAPH v1.6 production-qualification scaffold

v1.6 implements the missing *qualification infrastructure* from the v1.5 production-readiness audit. It does not claim that production qualification has been achieved: hardware-backed GDN2 parity, real repository training, validation calibration, sandbox image qualification, and distributed-cluster runs must still be executed in the target environment.

## Workstream 1 — GDN2 hardware parity

`scripts/qualify_gdn2_hardware.py` consumes an external low-level kernel adapter with the contract:

```python
kernel(q, k, v, erase, write, log_decay, initial_state) -> (outputs, final_state)
```

It compares against `gdn2_recurrence_reference` across dtype, context length, and gate regimes. Default acceptance tolerances are FP32 `1e-5`, BF16 `5e-3`, FP16 `1e-2`. Length 16384/32768 should be run on sufficiently large target GPUs rather than in CI.

## Workstream 2 — real Repo2LoRA training

`HuggingFaceFileEmbedder` and `RepositoryEmbeddingPipeline` provide the frozen Qwen3-Embedding path with tokenizer-aware 4096/512 overlapping chunks and FP32 RepoPooler aggregation. `RepoBrainTrainer` freezes the base checkpoint, trains only the hypernetwork, and reports target-module dense-delta Frobenius norms.

## Workstream 3 — empirical calibration

`calibrate_kl_envelope` derives the KL envelope from validation adapters that already satisfy generic-capability and execution gates and demonstrate positive repository benefit. `configs/daph_qwen06b_production_template.yaml` intentionally fails validation while `max_kl_drift` remains null.

Fisher correction acceptance requires both retained SVD energy and reconstruction error. The default qualification gates are retained energy >= 0.95 and relative reconstruction error <= 0.10.

## Workstream 4 — sandbox

`DockerPytestRunner` invokes Docker with no network, dropped Linux capabilities, `no-new-privileges`, read-only repository/root filesystem, resource limits, pids limit, and a bounded timeout. The runner never executes repository tests in the ML process. Production deployment should prefer a hardened runtime such as gVisor or Firecracker where available.

## Workstream 5 — distributed state

`AdapterPlacementManager` creates device/dtype-local factor copies for the actual owner of every patched RepoLoRALinear. `cache_state_dict`/`load_cache_state_dict` serialize tensor-backed HybridSequenceCache state; framework-owned opaque caches require an explicit codec rather than unsafe generic pickling.

FSDP guidance: patch `nn.Linear` modules before FSDP wrapping. Generated LoRA factors are runtime tensors/activations, not trainable Parameters. The Repo2LoRA hypernetwork is the trainable module and should be wrapped/sharded separately. For model-parallel inference, use `AdapterPlacementManager.apply` after factor generation.

## Remaining external gates

- H100/A100/L40S GDN2 kernel parity report.
- Real Qwen2.5-Coder/DAPH Repo2LoRA training on immutable 20/5/5+ split.
- Controller calibration on validation only; final test evaluated once after freeze.
- Hardened sandbox image and policy review.
- 1/2/4 GPU DDP and FSDP/ZeRO-3 qualification, followed by multi-node failure/restart testing.

## Calibration promotion path

1. Produce validation-only adapter records.
2. Run `scripts/calibrate_controller.py validation_records.json kl_calibration.json`.
3. Run `scripts/apply_kl_calibration.py configs/daph_qwen06b_production_template.yaml kl_calibration.json configs/daph_qwen06b_production.yaml`.
4. The generated YAML is parsed by `DAPHIntegrationConfig` before it is written as qualified input to later runs.

The repository embedder chunks token IDs directly; it does not decode and re-tokenize each window, avoiding boundary drift in the 4096/512 chunk contract.
