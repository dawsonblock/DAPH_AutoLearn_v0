# Ablations and acceptance gates

## A. GDN2

Ablations:

- attention-only
- current attention/Mamba
- GDN2 every 2 layers
- GDN2 every 3 layers
- GDN2 with and without short conv if upstream supports both

Acceptance:

1. Correctness harness passes dtype-specific numerical gates.
2. No state leakage across packed examples.
3. No sparse token compaction.
4. Long-context/retrieval metric improves materially or equal quality is achieved at lower cost.
5. Short-context quality does not regress beyond a predeclared tolerance.

## B. Repo2LoRA-Lite

Ablations:

- 1 vs 4 vs per-layer groups
- rank 4/8/16
- q,v vs q,v,o,down vs all seven projection families
- mean only vs mean+max repo pooling
- frozen embedder choices

Acceptance:

1. Beats no-repo-context baseline on unseen repositories.
2. Beats or matches a reasonable retrieval baseline on the target workload, or wins decisively on latency/context-token cost.
3. Beats a shared LoRA on cross-repository holdout.
4. Protected generic capabilities stay within declared regression bounds.
5. Adapter generation is deterministic for a fixed repo snapshot and model version.

## C. Controller

Ablations:

- no controller
- geometry warning only
- scaling only from functional probes
- scaling + localized projection

Acceptance:

1. Geometry-only signals never directly mutate weights.
2. Controller reduces harmful-adapter rate on held-out calibration repositories.
3. Controller preserves most gains of safe adapters.
4. Every project/scale action is re-probed after modification.
5. False rejection rate is reported, not hidden.

## D. Evolution

Acceptance:

1. Strict chronological data lineage.
2. State replay reproduces the same adapter trajectory.
3. Recurrent update beats stale Static after repository drift.
4. It either beats full snapshot regeneration or offers a compelling update-cost advantage.

## E. Vision

Acceptance:

1. Structural branch improves predeclared structural tasks.
2. Semantic-only tasks do not materially regress.
3. Added latency/memory is measured.
