# DAPH v0.2.0: Capability-Basis Architecture

## Core hypothesis

Instead of predicting unconstrained LoRA matrices from repository embeddings, learn a reusable whole-model adapter geometry from direct specialist adapters and restrict repository adaptation to that subspace.

For teacher adapter i, materialize its scaled parameter delta across all target modules and layer groups into one vector d_i. Stack teachers into D and compute SVD:

    D = U Σ Vᵀ

The first K rows of Vᵀ define joint whole-model capability directions. Each direction is split back into module/group blocks and rank-r refactored. At runtime RepoBrain predicts signed coordinates c(repo) and composes:

    ΔW_repo = Σ_k c_k(repo) U_k

The implementation concatenates the low-rank factors of all K basis blocks, so the weighted sum is applied without dense weight materialization.

## Why this is a better constraint

- Direct-LoRA teachers define an empirically useful adaptation manifold.
- RepoBrain solves a lower-dimensional coordinate inference problem instead of arbitrary matrix synthesis.
- Basis indices are shared across module types, preserving whole-model co-adaptation geometry.
- Teacher SVD coordinates provide a direct supervised target for amortized adapter learning.
- The basis can be frozen for strict generalization tests or fine-tuned later as an ablation.

## GDN2 insertion

GDN2 is wrapped with an identity-preserving interpolation:

    y = x + tanh(g) [GDN2(x) - x]

At g=0 the pretrained path is exactly preserved. This eliminates the previous failure mode where an untrained sequence mixer could immediately corrupt hidden-state distributions.

## Required experiments

Use repository-disjoint and clone-screened splits. Compare base, RAG, shared LoRA, direct LoRA, basis RepoBrain, and basis RepoBrain+RAG. Report paired confidence intervals, per-repository outcomes, AAE, generic-regression metrics, latency, VRAM, and basis reconstruction diagnostics.

The architecture is successful only if unseen-repository AAE is materially positive and the confidence interval excludes trivial gain.
