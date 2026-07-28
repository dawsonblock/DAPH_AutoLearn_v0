# TEST REPORT V0.3.9 — Causal Learning Loop Repair

## Executed Results

**Command**: `python -m pytest tests/ --tb=no -q`

**Result**: 585 passed, 1 skipped, 0 failed, 4 warnings in 19.71s

**Platform**: macOS Darwin 25.2.0, Python 3.12.0, pytest 8.4.2

## Pytest Totals

| Metric   | Count |
|----------|-------|
| Collected| 586   |
| Passed   | 585   |
| Skipped  | 1     |
| Failed   | 0     |
| Errors   | 0     |

The 1 skip is a hardware-dependent test (`test_resolve_transformer_layers_real_model`)
that requires a real model download.

## Release Gate Results

| Gate | Description | Result |
|------|-------------|--------|
| GATE 1 | Full existing unit suite passes | PASS |
| GATE 2 | Candidate-vector sensitivity test | PASS |
| GATE 3 | Incumbent-vector sensitivity test | PASS |
| GATE 4 | Oracle-leakage test | PASS |
| GATE 5 | Capture/task-ID alignment tests | PASS |
| GATE 6 | Trust-region mathematical invariants | PASS |
| GATE 7 | Synthetic closed-loop AutoLearn improves | PASS |
| GATE 8 | Failed candidates reliably rollback | PASS |
| GATE 9 | Provenance fields reflect runtime config | PASS |
| GATE 10 | Dataset split hashes and leakage checks | PASS |
| GATE 11 | CLI executes corrected counterfactual engine | PASS |
| GATE 12 | Version surfaces agree | PASS |
| GATE 13 | Real-model smoke test | NOT RUN (requires model download) |
| GATE 14 | Candidate policy depends on learned vector | PASS |

## Synthetic Learning Result

**Test**: `test_synthetic_closed_loop_autolearn_improves_over_incumbent`

**Environment**: Deterministic synthetic with two task families:
- Family A: symbolic correct (utility=1), LLM wrong (utility=0)
- Family B: symbolic wrong (utility=0), LLM correct (utility=1)

**Initial incumbent**: Deliberately imperfect (`-axis0`, routes both families
wrong).

**Result**: After 5 iterations, the loop promoted at least one candidate. The
final incumbent:
- Differs from the initial bad incumbent (vector was actually updated).
- Has positive axis 0 (learned the optimal direction).
- Achieves higher utility than the bad incumbent on held-out tasks.
- Is correct on more held-out tasks than the bad incumbent.

**Conclusion**: AutoLearn learns. The causal chain is genuine.

## Vector Sensitivity Result

**Test**: `test_heldout_evaluation_is_sensitive_to_candidate_vector`

**Result**: Candidate `+v` routes family A -> symbolic, family B -> llm.
Candidate `-v` routes family A -> llm, family B -> symbolic. The routes
differ, proving the evaluator uses the candidate vector (not the oracle).

## Oracle Leakage Result

**Test**: `test_oracle_leakage_candidate_selects_llm_but_oracle_prefers_symbolic`

**Result**: When the oracle prefers symbolic but the candidate policy selects
LLM, the candidate evaluation scores the LLM action (incorrect), NOT
symbolic (correct). This proves ground-truth utility does not leak into
policy selection.

## Capture Alignment Result

**Test**: `test_capture_alignment_exactly_8_of_10_enter_update`

**Result**: With 10 attempted captures and 2 failures, exactly 8 aligned
examples enter the activation update. Each weight belongs to the correct
task (verified by `test_capture_alignment_weights_match_task_ids`). No
positional shifting occurs.

## New Test Files

- `tests/test_v039_causal_gates.py`: 20 tests covering GATEs 2-8, 14.
- Updated `tests/test_trust_region.py`: adapted to new `route_fn` and
  `CaptureResult` contract.
