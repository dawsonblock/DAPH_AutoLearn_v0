# DAPH GDN2 + RepoBrain v1.2 Remediation

This release implements the six issues from the second line-by-line audit and extends the proposed fixes where necessary to preserve end-to-end tensor contracts.

## 1. External GDN2 kwargs dispatch

`ExternalGDN2Adapter` now detects `inspect.Parameter.VAR_KEYWORD`. Explicit upstream parameter names remain preferred; generic wrappers receive the conventional `past_key_values`, `attention_mask`, and `use_cache` keywords. Recurrent state is normalized to the hidden-state device/dtype before dispatch.

## 2. Cross-family layer-path patching

The default layer regex now recognizes `layers`, `h`, `blocks`, and `layer`. `_set_child` also handles `ModuleDict`, allowing paths such as `transformer.h.0.q_proj` to be replaced correctly.

## 3. Batched Repo2LoRA

The hypernetwork accepts `[B,E]` embeddings for any `B >= 1`. For `B=1`, generated factors remain `[G,R,In]` / `[G,Out,R]` for backward compatibility. For `B>1`, the contract is `[B,G,R,In]` / `[B,G,Out,R]`.

The audit's minimal fix changed only generation. v1.2 also updates `RepoLoRALinear`, dense materialization, projection, and truncated-SVD refactorization so batched factors do not break downstream execution.

## 4. Fisher slicing

Per-sample Fisher accumulation now slices tensors, lists, tuples, and nested mappings when their leading length matches the inferred tensor batch size. Scalar or unrelated metadata is preserved.

## 5. Vision token reconciliation

`StructuralSemanticFusion` accepts mismatched token counts and linearly resamples structural tokens to the semantic sequence length. This is a 1-D sequence alignment fallback, not a substitute for true 2-D spatial registration when both patch grids are known.

## 6. Probe aliases

Controller categories recognize common execution, KL/drift, and validation-loss aliases. Functional evidence still dominates geometry.

## Verification

Executed in this package build:

```text
PYTHONPATH=src python -m pytest -q
29 passed

PYTHONPATH=src python scripts/validate_install.py
validation OK
```

## Remaining qualification work

This remains a research integration package. Before deployment, qualify concrete LLaMA/Qwen/Mistral/Gemma checkpoints, verify each model family's target-module names, run BF16/CUDA forward/backward tests, compare external GDN2 kernels against the recurrence reference, and execute code probes only through a hardened external sandbox.
