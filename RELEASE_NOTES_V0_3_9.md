# Release Notes — DAPH AutoLearn v0.3.9

## Summary

v0.3.9 is a **causal learning loop repair** release. The central fix is that
the held-out promotion evaluation now uses the candidate steering vector to
produce candidate routing decisions and the incumbent vector to produce
incumbent routing decisions. Previously, the candidate route was derived from
the oracle Delta-U target and the incumbent route was hard-coded to
`"symbolic"`, making promotion invalid.

## Critical Fix

**Before (v0.3.8.1)**:
```
_evaluate_held_out(candidate_vector):
    c_route = derive_learning_target(outcome).target  # ORACLE decides route
    i_route = "symbolic"                                # HARDCODED
```

**After (v0.3.9)**:
```
_evaluate_held_out(candidate_vector, incumbent_vector):
    candidate_routes = route_fn(tasks, candidate_vector, config)  # POLICY decides
    incumbent_routes = route_fn(tasks, incumbent_vector, config)  # POLICY decides
```

## New Components

- `src/daph_learning/autolearn/route_policy.py` — routing policy contract
- `src/daph_learning/autolearn/adapters.py` — model-backed adapters
- `tests/test_v039_causal_gates.py` — 20 release-gate tests

## Modified Components

- `src/daph_learning/autolearn/trust_region.py` — rewritten evaluation, bootstrap, bounds
- `src/daph_learning/autolearn/experience.py` — added CapturedActivation, CaptureResult
- `src/daph_learning/autolearn/promotion.py` — added capability gates, route fields
- `src/daph_learning/cli/commands/autolearn.py` — added --engine flag
- `src/daph_learning/__init__.py` — version 0.3.9
- `pyproject.toml` — version 0.3.9

## Test Results

- 585 passed, 1 skipped, 0 failed
- All 14 release gates pass (GATE 13 real-model smoke not run in unit suite)
- 20 new causal-gate tests pass

## Compatibility

- The legacy `run_autolearn_loop` is preserved and accessible via
  `--engine legacy` (deprecated).
- The corrected counterfactual loop is the default via
  `--engine counterfactual`.
- Existing tests have been adapted to the new `route_fn` and `CaptureResult`
  contract.
