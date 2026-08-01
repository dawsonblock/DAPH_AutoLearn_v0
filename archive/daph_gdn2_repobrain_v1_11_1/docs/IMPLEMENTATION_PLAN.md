# Full implementation plan

## Phase 0 — Baseline freeze

Before integrating anything, record the exact DAPH baseline commit, tokenizer, model configuration, training corpus version, evaluation harness version, seeds, dtype, and hardware. Every subsequent result must be comparable to this snapshot.

Required baselines:

- current DAPH attention/Mamba configuration
- attention-only or closest dense control
- current ExFusion universal checkpoint without repository adaptation

No architecture claim is valid until the harness reproduces these baselines.

## Phase 1 — GDN2 mixer integration

### 1.1 Add a mixer boundary

DAPH blocks should depend on a generic mixer interface, not import GDN2 directly. The wrapper lives at `daph_ext.mixers`.

Expected block pseudocode:

```python
residual = x
x = pre_mixer_norm(x)
mix = self.sequence_mixer(x, recurrent_state=state, use_cache=use_cache)
x = residual + mix.hidden_states

residual = x
x = residual + self.ffn(self.pre_ffn_norm(x))
```

Do not change the FFN or ExFusion parameter path in the first GDN2 experiment.

### 1.2 Fixed schedule

Start with `GDN2, Attention, GDN2, Attention, ...` because it isolates the mixer variable. Dynamic routing comes later.

### 1.3 State handling

Training: chunk mode if the external implementation supports it.

Autoregressive decode: recurrent/cache mode.

The state must be reset exactly at sequence boundaries. Packed examples require correct sequence IDs or separate states. Never carry state across unrelated documents.

### 1.4 No token compaction

Token-level routes may mask writes in a future implementation, but may not delete time steps. v1 should not expose token-level recurrent routing at all.

## Phase 2 — Numerical qualification

Compare the external implementation with the slow recurrence reference.

Test grid:

- dtype: FP32, BF16, FP16 where supported
- T: 1, 16, 64, 256, 1024, 4096
- gate regimes: low erase, high erase, low/high decay, adversarial near-boundary values
- state: zero and non-zero initial state

Metrics:

- max absolute output error
- relative L2 output error
- max absolute final-state error
- relative L2 final-state error
- NaN/Inf count

Initial research thresholds, to be calibrated empirically:

- FP32 relative L2: <= 1e-5 for reference-equivalent path
- BF16: <= 5e-3
- FP16: <= 1e-2

If the actual upstream kernel uses a mathematically equivalent but reordered algorithm, use a precision-aware threshold and verify downstream loss, not bitwise equality.

## Phase 3 — Repo2LoRA-Static-Lite

### 3.1 Dataset

Build a repository-level benchmark from your own DAPH development repositories plus permissively licensed public repositories. Split by repository, not random assertion/example alone.

You need:

- train repositories
- cross-repository validation repositories never seen by the hypernetwork
- cross-repository test repositories
- optional in-repository holdout

Keep test files/tasks separate from repository-conditioning source when the task design requires it.

### 3.2 Repository encoding

First reproduce the simple approach:

- chunk source files
- frozen code/text embedding model
- mean-pool chunks per file
- weighted mean + max pool across files

Cache embeddings. Hypernetwork training should not repeatedly run the repository encoder unless you are explicitly training it.

### 3.3 Generator

Train only Repo2LoRA-Lite; freeze DAPH.

Start:

- rank 8
- alpha 16
- 4 layer groups
- targets `q_proj`, `v_proj`, `o_proj`, `down_proj`

Loss:

- normal next-token/task loss for repo-conditioned examples
- optional adapter-norm regularizer
- optional protected-domain KL regularizer against the universal model

### 3.4 Baselines

At minimum:

- universal model, no repo context
- retrieval/context injection
- one shared LoRA across all repositories
- per-repository LoRA where data budget makes it meaningful
- Repo2LoRA-Lite

Do not call the generated adapter useful if it only beats the no-context model but loses to a cheap retrieval baseline.

## Phase 4 — Functional interference controller

### 4.1 Geometry collection

For each generated factor pair, materialize the dense delta only offline. Compare with relevant permanent deltas per group/module.

### 4.2 Protected probes

A candidate repository adapter must be tested on:

- repository-specific held-out task
- generic coding set
- general language set
- math/reasoning set if those are protected DAPH capabilities
- tool-format/function-calling set if deployed

Record:

- loss delta
- task-score delta
- logit KL
- execution correctness where possible

### 4.3 Controller policy

v1 policy is conservative:

- pass all gates -> accept
- mild loss/KL drift -> search scale grid
- localized functional regression + strong geometry conflict -> test projection on only that module/group
- execution regression -> reject unless a lower scale demonstrably repairs it

Every intervention must be re-evaluated after modification.

## Phase 5 — Repo2LoRA-Evolution

Only start when Static passes Phase 3 acceptance gates.

Use chronological commit splits. Encode production-code diffs and update a repository state with a GRU or equivalent recurrent state.

Critical leakage rule: state at commit t may use only snapshot/history <= t. No future files, tests, or diffs.

Compare:

- frozen static snapshot adapter
- re-encoded latest snapshot adapter
- recurrent diff-state adapter
- shared LoRA

The recurrent state is justified only if it beats a simple re-encode strategy at an acceptable update cost.

## Phase 6 — Structural vision

Add the vision branch only after text/code changes are stable.

Start with frozen encoders. Evaluate tasks where structural features should matter:

- UI element localization/matching
- screenshot change detection
- repeated-pattern or layout reasoning
- image retrieval by structural similarity

A DINO branch that does not improve structural tasks is dead weight.

## Phase 7 — Kernel optimization

Profile end to end. Only optimize triangular inversion/solve if it is a material portion of wall time. Do not infer a bottleneck from another hardware platform's profile.

Required before kernel work:

- GPU profiler trace on your target NVIDIA hardware
- percentage of GDN2 forward/backward time in triangular solve
- arithmetic intensity and memory-bandwidth evidence
- numerical baseline from Phase 2

If the solve is not a top bottleneck, spend engineering time elsewhere.
