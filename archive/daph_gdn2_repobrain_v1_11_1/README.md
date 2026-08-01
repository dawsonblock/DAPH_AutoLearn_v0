# DAPH GDN2 + RepoBrain Integration v1.11.1

Research scaffold for three deliberately separate adaptation timescales:

- **ExFusion** — permanent capability consolidation.
- **Repo2LoRA-Lite** — repository-specific parametric adaptation.
- **GDN2** — recurrent token-sequence working memory.

v1.10 is the empirical-mechanism qualification release; v1.9 remains the production-qualification hardening baseline. It retains v1.6's
qualification tooling and closes additional cache, calibration, sandbox,
embedding-memory, and distributed end-to-end gaps.

## v1.6 foundation retained

- GDN2 low-level hardware parity harness with an FP32 mathematical oracle,
  FP32/BF16/FP16 tolerances, cache-contract qualification, adversarial gate
  regimes, and context lengths through 32K tokens.
- Frozen Hugging Face repository embedder integration using direct token-ID
  windows (4096 tokens, 512 overlap) and FP32 mean+max RepoPooler aggregation.
- Real RepoBrain hypernetwork training utility that freezes the base checkpoint,
  trains Repo2LoRA only, and records dense-delta Frobenius norms by target.
- Immutable split tooling with default 20/5/5 pilot minimums.
- Validation-only KL envelope calibration, correction-grid generation, and Pareto
  utilities; no production KL value is invented in this package.
- Fisher refactoring now requires both retained SVD energy and bounded relative
  reconstruction error.
- Hardened Docker pytest runner with no network, a read-only host repository
  copied into ephemeral tmpfs, dropped capabilities, non-root execution,
  resource/process limits, and timeout.
- Rank/device-aware dynamic factor placement plus tensor/opaque-cache checkpoint
  contracts for distributed execution and RepoEvolution hidden state.
- FSDP and DeepSpeed ZeRO-3 qualification scripts for target GPU clusters.
- Architecture discovery for Qwen, LLaMA, Mistral/Mixtral, and Gemma-style
  decoder checkpoints.
- Strict production YAML template that refuses to load until a validation-derived
  KL threshold has been applied.

## v1.5 corrections retained

- Correctly handles upstream GDN2's three-tuple return
  `(hidden_states, None, past_key_values)` instead of dropping the cache.
- Allows opaque FLA/Transformers recurrent cache objects.
- Corrects GDN2 reference readout to use `q` (`o = S^T q`).
- Fixes probe alias collisions and requires actual repository-task benefit.
- Uses deterministic Fisher ranking and avoids pruning zero-Fisher coordinates.
- Enforces empirical provenance, immutable splits, strict patching, finite pooling,
  Fisher estimator semantics, and stronger tensor/config validation.

## Required interpretation

Passing the unit suite proves only the tested software contracts. It does **not**
prove that Repo2LoRA improves a real checkpoint, Fisher correction helps, or
GDN2 improves long-context performance. Those remain empirical qualification
gates in `docs/EMPIRICAL_PROTOCOL.md`.

## Quick validation

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/validate_install.py
python -m compileall -q src tests scripts
```

The package does not redistribute NVIDIA GatedDeltaNet-2 or Huawei Ascend
triangular-inverse kernels; see `THIRD_PARTY_NOTICES.md` and
`LICENSE_STATUS.md`.

## v1.8 qualification hardening

v1.8 closes additional production-qualification gaps in cache dtype handling, incremental GDN2 cache parity, repository-embedding memory use, deterministic raw-list splitting, calibration sample sufficiency, sandbox workspace isolation, checkpoint schema/versioning, and end-to-end distributed smoke tests, plus a fail-closed release sign-off script. See `docs/PRODUCTION_QUALIFICATION_V1_7.md`.


## v1.8 serving-safety rule
Production inference must bind adapters with `daph_ext.repobrain.adapter_context.use_adapter_context`. Mutating `RepoLoRALinear.current_factors` is retained only for single-request research compatibility and is not concurrency-safe.

## v1.9 qualification hardening

v1.9 adds executable adversarial sandbox, restart/recovery, multi-node and FSDP+ZeRO aggregation harnesses, integrates repository lineage checking into split creation, and makes final sign-off validate the semantic content of those artifacts rather than only checking a top-level `passed` flag. See `docs/PRODUCTION_QUALIFICATION_V1_9.md`.


## v1.10 empirical mechanism qualification

v1.10 freezes architecture expansion and makes the core research hypotheses explicit. Before additional controller or mixer complexity is promoted, the package now requires a real-model comparison matrix: frozen base, RAG, shared LoRA, independently trained per-repository LoRA, Repo2LoRA-Lite, and Repo2LoRA-Lite + RAG.

The per-repository LoRA arm is the adaptation ceiling: if direct repository adaptation does not help, generated repository adaptation is not expected to help. If direct LoRA helps but Repo2LoRA does not, the failure is localized to the repository-to-adapter mapping rather than the base model's adaptability.

Pilot split minimums remain 20/5/5 for plumbing. Generalization qualification requires at least 100 train / 20 validation / 20 untouched test repositories, with lineage, fork-family, exact-content and available near-duplicate checks.

Production serving remains request-scoped through `use_adapter_context`. Batched repository factors are supported only when the factor batch dimension exactly matches the model input batch dimension.

## v0.2.1 — compressed-basis coefficient repair

v0.2.1 recomputes every teacher's coefficients after the SVD basis blocks are
rank-truncated. Coefficients are solved by least squares against the actual
compressed whole-model basis, and diagnostics now report post-compression
relative reconstruction error, explained energy, fit method, and coefficient
condition number.

This replaces the v0.2.0 mismatch where coefficients were calculated against
the orthonormal pre-compression basis and then reused after each basis block
had changed through independent low-rank refactoring.

## v0.2.0 — ExFusion Geometry Basis → RepoBrain Coefficients → Residual GDN2

This build adds the constrained adaptation path:

```text
Direct/ExFusion teacher adapters
        ↓
fit_joint_adapter_basis()
        ↓
whole-model capability basis U = {U₁ … Uₖ}
        ↓
repository embedding
        ↓
BasisRepoBrain / RepoCoefficientPredictor
        ↓
signed coordinates c(repo)
        ↓
ΔW_repo = Σₖ cₖ(repo) Uₖ
        ↓
request-scoped RepoLoRA execution
        ↓
ResidualGDN2Mixer for token-time recurrent memory
```

The basis is learned jointly across all target projection types and layer groups from independently trained LoRA teachers. SVD finds principal whole-model adapter directions; each direction is then low-rank refactored for efficient runtime composition. RepoBrain no longer has to invent arbitrary LoRA factors from scratch: it predicts coordinates in the learned capability geometry.

`ResidualGDN2Mixer` is identity-preserving at initialization:

```text
y = x + tanh(gate) * (GDN2(x) - x)
```

With `gate=0`, inserting the mixer does not perturb the pretrained hidden-state distribution. The gate can be learned during continued training.

### Build a basis from direct-LoRA teachers

```bash
PYTHONPATH=src python scripts/build_exfusion_adapter_basis.py \
  teachers/repo_001.pt teachers/repo_002.pt teachers/repo_003.pt \
  --basis-size 16 --basis-rank 8 \
  --output artifacts/capability_basis.pt
```

The saved artifact includes teacher coordinates and basis-fit diagnostics. Use those teacher coordinates to supervise `BasisRepoBrainTrainer` in addition to downstream language-model loss.

### Scientific gate

Do not treat this build as validated merely because unit tests pass. The decisive comparison remains:

1. frozen base
2. base + RAG
3. shared LoRA
4. direct per-repository LoRA
5. basis RepoBrain
6. basis RepoBrain + RAG

Report Adapter Amortization Efficiency (AAE):

```text
AAE = (M_repoBrain - M_base) / (M_directLoRA - M_base)
```

A high unseen-repository AAE is the evidence that repository → capability-coordinate inference actually works.
