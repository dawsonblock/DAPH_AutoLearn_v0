# Test Report — DAPH AutoLearn v0.3.10.5-alpha

## Environment

| Item | Value |
|------|-------|
| Platform | macOS Darwin 25.2.0 |
| Python | 3.12.0 (pyenv) |
| pytest | 8.4.2 |
| torch | 2.10.0 (MPS available, CUDA not available) |
| transformers | 5.13.0 |
| Repo root | `/Users/dawsonblock/Downloads/DAPH_AutoLearn_v0_3_8_protocol_repair` |

## Canonical source hash

```
c5191b73d4ca07d58c94d6ec31ac4a12d8f3de118640878cd34b27f8e3be2f32
```

Command: `python -m daph_learning.provenance source-hash`

## Executed commands and results

### 1. Full test suite

Command:
```
python -m pytest
```
Result:
```
1048 passed, 4 skipped, 4 warnings in 71.20s
exit code: 0
```

The 4 skips are hardware/model-download-gated tests
(`test_route_record_llm_task_has_answer_latency`,
`test_resolve_transformer_layers_real_model`, and two others) that skip
when explicit model-download enablement or unavailable hardware is not
present. No test failed. No test was deleted or weakened to achieve this
result.

### 2. New Priority 0 test files (subset run)

Command:
```
python -m pytest tests/test_artifact_integrity.py \
                 tests/test_current_artifact_tree_contains_no_stale_source_hash.py \
                 tests/test_symbolic_safety_section6.py \
                 tests/test_canonical_verifier_section7.py -q
```
Result:
```
.......................................................................
71 passed
exit code: 0
```

Breakdown of new test files:

| File | Tests | Purpose |
|------|-------|---------|
| `tests/test_artifact_integrity.py` | 23 | Section 4.3 artifact-layout + bundle-validation CI gate. |
| `tests/test_current_artifact_tree_contains_no_stale_source_hash.py` | 2 | No stale source hash in `artifacts/current/`. |
| `tests/test_symbolic_safety_section6.py` | 36 | Section 6 eval-free symbolic execution + AST grammar. |
| `tests/test_canonical_verifier_section7.py` | 25 | Section 7 canonical FINAL_ANSWER verifier. |

### 3. Canonical source hash CLI

Command:
```
python -m daph_learning.provenance source-hash --json
```
Result:
```json
{
  "source_hash": "c5191b73d4ca07d58c94d6ec31ac4a12d8f3de118640878cd34b27f8e3be2f32",
  "repo_root": "/Users/dawsonblock/Downloads/DAPH_AutoLearn_v0_3_8_protocol_repair",
  "include_globs": ["src/**/*.py", "scripts/**/*.py", "tests/**/*.py"],
  "exclude_globs": ["**/__pycache__/**", "**/*.pyc", "artifacts/**",
                    ".git/**", ".pytest_cache/**", "build/**", "dist/**"],
  "hash_algorithm": "SHA-256",
  "hash_length": 64
}
```

### 4. Real-model smoke (opportunistic, locally-cached model)

Command:
```
python scripts/run_real_model_smoke.py
```
Result:
```
[smoke] source_hash=c5191b73d4ca07d5... run_id=smoke_a3416479f15f
[smoke] loading model Qwen/Qwen2.5-0.5B-Instruct (offline, from cache)
[smoke] device=mps n_layers=24
[smoke] task 0: expected=2408552 sym=UNVERIFIABLE llm=UNVERIFIABLE ...
[smoke] task 3: expected=32768 sym=UNVERIFIABLE llm=UNVERIFIABLE llm_out='4096 * 8 = 32768'
[smoke] bundle written: artifacts/real_model_smoke/smoke_a3416479f15f
[smoke] bundle valid=True errors=0 warnings=0
[smoke] OK — real-model smoke completed. NOT Gate A qualified.
exit code: 0
```

Findings (honest):
- The real model loaded from cache and generated text on MPS; hidden
  states were captured (shape recorded per task).
- The canonical `FINAL_ANSWER:` verifier **correctly failed closed**
  (UNVERIFIABLE) on every output because neither the symbolic backend's
  bare-integer output nor the LLM's prose output used the canonical
  `FINAL_ANSWER: <integer>` field. This is the verifier working as
  specified — it does not credit substring matches.
- This demonstrates a real Priority 1 integration requirement: the
  symbolic backend and the LLM generation prompt/format must emit the
  canonical `FINAL_ANSWER:` field to be credited under the new verifier.
- The smoke bundle validated with `validate_artifact_bundle`
  (`valid=True`) and is labeled `REAL_MODEL_SMOKE` / `status: SMOKE`. It
  is **not** Gate A evidence and is not under `gate_a_qualified/`.

## Summary

| Metric | Value |
|--------|-------|
| Collected tests | 1048 passed + 4 skipped = 1052 |
| Passed | 1048 |
| Failed | 0 |
| Skipped | 4 |
| Errors | 0 |
| Exit code | 0 |
| New test files | 4 (86 new tests) |
| Real-model smoke executed | Yes (Qwen2.5-0.5B-Instruct, MPS, offline cache) |
| Full Gate A experiment executed | No |
