# DAPH patch guide

This is the minimum invasive integration path for your existing DAPH repository.

## 1. Find the block class

Locate the code that currently selects attention/Mamba/FNet/cheap-path. Replace direct branching with a `sequence_mixer` object created at model construction.

Pseudo-diff:

```diff
- if route == "mamba":
-     y = self.mamba(x)
- elif route == "attention":
-     y = self.attn(x, mask)
+ y = self.sequence_mixer(
+     x,
+     attention_mask=mask,
+     recurrent_state=cache.get(self.layer_idx),
+     use_cache=use_cache,
+ )
```

For v1, `route` is determined once per layer from `alternating_gdn2_attention`. Remove token-level recurrent routing from the experimental path.

## 2. Add cache namespace

Do not overload attention KV cache with GDN2 state. Use separate fields:

```python
cache.attention[layer_idx]
cache.recurrent[layer_idx]
```

Reset both at sequence boundaries.

## 3. Add RepoBrain outside the block

Repo2LoRA generation should happen once per repository snapshot/state update, not once per token. Attach the resulting factors to target modules through your existing LoRA/PEFT mechanism.

Recommended lifecycle:

```text
repo version changes
    -> compute/update repo embedding
    -> generate factors
    -> run controller qualification
    -> cache approved adapter
    -> serve many token requests
```

## 4. ExFusion integration point

Do not merge generated repo adapters into the canonical checkpoint. Export permanent expert deltas and final ExFusion deltas to a diagnostics registry. RepoBrain consumes that registry for comparison only.

Suggested key:

```text
(layer_idx, module_name) -> permanent_delta / fisher_diag / protected metadata
```

## 5. Feature flags

Add independent flags:

- `enable_gdn2`
- `enable_repobrain`
- `enable_adapter_controller`
- `enable_structural_vision`

Never make one flag implicitly enable another. This keeps ablations honest.
