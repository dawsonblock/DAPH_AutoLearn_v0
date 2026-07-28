# Run manifest schema

A run manifest captures the provenance required to reinterpret an experimental
output. The implementation in
`src/daph_learning/evaluation/manifest.py` is authoritative.

## Schema overview

```json
{
  "manifest_version": "daph.run.v1",
  "run_id": "qualified-run-001",
  "created_at": "2026-07-28T12:00:00Z",
  "git_commit": "40-character commit SHA or null",
  "git_dirty": false,
  "daph_version": "0.3.10",
  "model": {
    "repo": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "immutable model revision",
    "config_hash": "sha256",
    "hidden_size": 1536,
    "num_layers": 28,
    "dtype": "float32",
    "device_map": "cuda:0",
    "attention_impl": "sdpa"
  },
  "tokenizer": {
    "repo": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "immutable tokenizer revision",
    "chat_template_hash": "sha256",
    "pad_token_id": 151643,
    "pad_side": "left"
  },
  "environment": {
    "python_version": "3.12.0",
    "torch_version": "2.4.0",
    "transformers_version": "4.45.0",
    "cuda_version": "12.4",
    "gpu_model": "NVIDIA H100 80GB",
    "os": "linux"
  },
  "dataset": {
    "path": "data/protocol_v038/final_test.jsonl",
    "sha256": "sha256",
    "num_tasks": 1000,
    "split": "final_test",
    "label_field": "utility_oracle",
    "label_oracle_kind": "utility"
  },
  "decoding": {
    "seed": 42,
    "prompt_format": "chat",
    "max_new_tokens": 64,
    "do_sample": false,
    "temperature": 0.0,
    "top_p": 1.0,
    "route_decision_mode": "logit",
    "route_token_resolver": "contextual",
    "label_scoring": "sequence",
    "sequence_normalization": "mean",
    "batch_size": 16,
    "route_batch_size": 32
  },
  "protocol": {
    "purpose": "final_evaluation",
    "configuration_frozen": true,
    "leakage_report_sha256": "sha256",
    "final_test_access_sha256": "sha256"
  },
  "vectors": [
    {
      "path": "vectors/tool_l20.npz",
      "vector_id": "tool-v1",
      "family": "tool_policy",
      "behavior": "invoke_symbolic_tool",
      "layer": 20,
      "alpha": 3.0,
      "anchor": "ACTION:",
      "token_scope": "last",
      "hidden_size": 1536,
      "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
      "extraction_method": "contrastive_mean_difference",
      "normalization": "none",
      "positive_n": 200,
      "negative_n": 200,
      "capture_anchor": "ACTION:",
      "capture_prompt_format": "chat",
      "capture_dataset_sha256": "sha256",
      "vector_sha256": "sha256"
    }
  ],
  "pytorch_determinism": {
    "seed": 42,
    "cudnn_deterministic": true,
    "cudnn_benchmark": false,
    "use_deterministic_algorithms": true
  },
  "outputs": [
    {
      "path": "runs/final.routes.jsonl",
      "kind": "routes",
      "sha256": "sha256"
    }
  ]
}
```

## Validation tiers

`REQUIRED` fields identify the run, package, model, tokenizer, dataset, seed,
prompt format, token resolver, and at least one output.

`REQUIRED_FOR_HEADLINE` additionally requires:

- immutable model and tokenizer revisions;
- model config and chat-template hashes;
- dtype and installed torch/transformers versions;
- a clean, identified code revision;
- complete vector and capture provenance;
- `decoding.label_scoring="sequence"`;
- `decoding.sequence_normalization` equal to `mean` or `sum`;
- `protocol.configuration_frozen=true`;
- a protocol purpose;
- `protocol.leakage_report_sha256`;
- for test/final-test runs, `protocol.purpose="final_evaluation"` and
  `protocol.final_test_access_sha256`.

Unknown required-for-headline values are recorded as `null` and make the
manifest ineligible; they must not be invented.

## Split discipline

Allowed split labels include:

- `train` / `capture` — representation capture or training;
- `dev` / `validation_alpha` / `validation_threshold` — adaptive selection;
- `calibration` — frozen calibration procedures;
- `test` / `final_test` — one-shot final evaluation.

The protocol guard rejects test/final-test use for capture, tuning,
calibration, model selection, and engineering evaluation. A final evaluation
must use a frozen configuration. The one-shot ledger is created atomically by
`reserve_final_test`.

The manifest validator also rejects a test/final-test run when a vector's
`capture_dataset_sha256` equals the evaluated dataset SHA-256.

## Route scoring honesty

`route_token_resolver` records how the route boundary was resolved:

- `isolated` tokenizes a label independently;
- `contextual` derives continuation tokens from the rendered prompt boundary.

`label_scoring` records the objective:

- `single_token` is the historical next-token contrast;
- `sequence` scores the complete label with teacher forcing.

Only `sequence` is headline-eligible. `sequence_normalization`
records whether complete-label log probabilities were summed or averaged.

## Vector provenance

Every vector entry records model/layer identity, capture data hash, extraction
method, normalization, sample counts, capture anchor and prompt format, and a
hash of the values. A missing field is a reproducibility failure, not an
invitation to infer it later.

## Output convention

The standard layout is:

```text
runs/final.routes.jsonl
runs/final.routes.jsonl.manifest.json
```

The caller prints the manifest SHA-256 after emission. Downstream tooling
should verify that hash before using the result.

## Versioning

The schema remains `daph.run.v1`; v0.3.10 adds fields compatibly. A future
incompatible change must bump the manifest version.
