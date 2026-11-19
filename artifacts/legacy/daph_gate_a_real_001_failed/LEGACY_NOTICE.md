# Legacy Notice — daph_gate_a_real_001_failed

This experiment was executed from a historical source state that does not
match the current repository.
The experiment produced a positive point estimate but failed its authoritative
group-aware confidence-bound gate because the 95% lower confidence bound for
P1 minus P0 crossed zero.
This artifact is retained for audit history only and does not qualify the
current source tree.

## Identity

- Experiment ID (assigned retrospectively): `daph_gate_a_real_001_failed`
- Release at time of run: `0.3.10.3.2-alpha`
- Historical source hash: `fd4d47e3b6af7373a70142469cce6532ee3a90eff505cd4c622f575b3ccb2374`
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (float16)
- Gate verdict: **FAILED** (criterion A2: LCB95%(P1 - P0) = -0.041 < 0)

## Why it failed

The grouped bootstrap (10,000 iterations, 14 groups) on the 1002-task final
split yielded a point estimate of +0.193 for P1 - P0 but a 95% lower
confidence bound of -0.041, crossing zero. The CI was wide because only 14
linguistic-template groups were resampled and the utility distribution was
heavy-tailed (most tasks were ties). See `GATE_A_RESULTS.md` and
`GATE_A_FIXES.md` for the full analysis.

## Evidence present in this archive

- `GATE_A_RESULTS.md` — human-readable report of the failed real-model run
- `GATE_A_FIXES.md` — root-cause analysis and proposed fixes
- `experiment_results.json` — synthetic dev-split results (180 tasks) that
  were co-located with the report at the historical source state
- `experiment_results_synthetic_dev.json` — duplicate of the synthetic
  dev artifact previously under `artifacts/current/`
- `current_source_manifest.json`, `freeze_manifest.json`, `release_gates.json`,
  `release_gates_current.json`, `test_report_*`, `junit.xml` — manifests and
  test reports from the historical source state
- `real_qwen_smoke_0.5b_result.json`, `smoke_real_model_result.json` —
  earlier real-model smoke results (0.5B) archived under the historical state

## Evidence explicitly MISSING (recorded honestly, not fabricated)

The following artifacts referenced by the Gate A protocol were NOT present in
the historical source tree and could not be archived:

- `raw_metrics.json` — per-task raw metrics for the real 1.5B final run
- `bootstrap_results.json` — stored bootstrap samples / CI details
- `environment.json` — recorded GPU/software environment for the real run

Their absence is one of the integrity defects corrected by the v0.3.10.4-alpha
repair: the new `daph_gate_a_real_002` protocol requires these artifacts to
exist and to validate before any qualification claim is made.

## Provenance

This directory is marked `legacy: true` and `not_current_source: true` in
`historical_source_manifest.json`. It MUST NOT be presented as current Gate A
evidence. It is retained for audit history only.
