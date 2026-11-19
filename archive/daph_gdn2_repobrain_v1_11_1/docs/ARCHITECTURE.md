# Architecture

## 1. Separation of timescales

The design enforces three different mechanisms instead of pretending they solve the same problem.

| Mechanism | Update timescale | Purpose | Persistence |
|---|---|---|---|
| ExFusion | offline / release | consolidate transferable capabilities | checkpoint |
| Repo2LoRA | repo snapshot or commit | repository-specific parametric context | adapter/state |
| GDN2 | token step | recurrent working memory | inference state |

This separation is non-negotiable for v1. Merging repository-specific knowledge permanently into the universal checkpoint creates unnecessary interference and invalidates clean evaluation.

## 2. DAPH block integration

The first experiment is deterministic layer routing, not token routing.

```text
L0  GDN2
L1  attention/SWA
L2  GDN2
L3  attention/SWA
...
```

A recurrent mixer processes every causal step. Arbitrary compaction such as selecting tokens `[0,3,4]` and presenting them as a length-3 sequence changes the recurrence semantics. Therefore sparse compaction is prohibited.

## 3. Repo2LoRA-Lite

Repository representation:

```text
file tokens -> chunks -> frozen embedding model -> file vectors
                                         |
                       weighted mean + max pooling
                                         |
                                  repository vector
```

Default repository vector shape: `[1, 2048]` when file embedding dimension is 1024 and mean+max pooling is used.

Generator:

```text
[1,2048]
   -> Linear(2048,512)
   -> GELU + LayerNorm
   -> Linear(512,512)
   -> GELU + LayerNorm
   -> module-specific factor heads
```

For each target module and layer group, produce:

- `A[g]`: `[rank, in_features]`
- `B[g]`: `[out_features, rank]`
- effective delta `DeltaW = alpha/rank * B @ A`

Four groups over 28 layers map approximately to 7 layers/group.

## 4. ExFusion diagnostics

Generated adapters are treated as candidate task vectors. Geometry is computed per module/layer group against permanent expert deltas or the final ExFusion delta.

Metrics:

- cosine similarity
- norm ratio
- sign conflict fraction
- projection overlap
- Fisher-weighted candidate energy when a trustworthy Fisher diagonal exists

These metrics identify suspicious regions. They do not prove functional conflict.

## 5. Functional controller

Decision hierarchy:

1. Code execution / behavioral correctness.
2. Held-out repository loss and task score.
3. KL/logit drift on protected base capabilities.
4. Geometry warning signals.

The controller may accept, scale, layer-gate, project a specific conflicting component, or reject. Projection is only justified when functional probes regress and geometry identifies a plausible conflict direction.

## 6. Structural vision

DINO is a structural branch, not the sole semantic vision system.

```text
image -> semantic VLM tokens -----------\
                                        -> alignment -> gated fusion -> DAPH
image -> DINO patch/global features ----/
```

The `VisionEncoder` protocol keeps DINOv2 replaceable by DINOv3, SigLIP-family encoders, or another structural encoder.
