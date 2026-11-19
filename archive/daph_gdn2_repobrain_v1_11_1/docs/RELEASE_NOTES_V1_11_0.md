# DAPH v1.11.0 — Geometry-Basis Architecture

This release changes RepoBrain from unconstrained LoRA synthesis to constrained capability-coordinate prediction.

## Added

- Joint whole-model adapter basis fitted from direct-LoRA / ExFusion teacher deltas.
- Exact low-rank composition of `ΔW_repo = Σ c_k U_k` without dense runtime materialization.
- Repository coefficient predictor with signed coordinates and FSDP-safe scale parameter.
- BasisRepoBrain trainer supporting direct teacher-coordinate distillation plus task CE.
- Identity-preserving ResidualGDN2Mixer initialized to the original pretrained path.
- Basis serialization helpers, basis-building CLI, AAE calculator, and regression tests.

## Preserved

The v1.10.1 repairs remain in place: FSDP-safe scalar parameters, adapter batch expansion/index mapping, per-sample Fisher default, distributed cleanup, and existing qualification infrastructure.

## Validation

104 unit/integration tests pass in the build environment. `validate_install.py` and `compileall` pass.

This is an experimental research build. It has not yet demonstrated positive unseen-repository AAE or real-Qwen GDN2 gains.
