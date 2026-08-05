# Invalidation Notice — daph_gate_a_real_002

**Status: INVALIDATED**

This experiment executed a real model and produced useful raw evidence,
but its Gate A PASS is invalid.

The primary confidence interval was a placeholder, the sham comparison
used the wrong interval, and the reported P1 utility was not computed from
the actual routed actions of the frozen policy.

The raw outcomes are retained for audit and may be reanalyzed, but the
original gate decision must not be used as scientific evidence.

## Specific defects

1. The primary confidence interval was hardcoded as `point_estimate ± 0.1`.
2. The P1-minus-sham interval was actually the sham utility interval.
3. P1 utility was based on policy probabilities, not selected routing actions.
4. Positive-group fraction measured symbolic preference, not P1 improvement.
5. Crossover-subtype count was just the number of subtypes.
6. Final decisive fraction was calculated from training data.
7. Worst-subtype regression used the same soft surrogate instead of deployed actions.
8. Several freeze checks compared manifest values against themselves.
9. Policy and calibration hashes were not fully enforced during final execution.
10. The final policy was retrained instead of loaded from the frozen artifact.
11. Calibration was frozen but not operationally applied.
12. Representation selection was not fully loaded and verified at final time.
13. Model and tokenizer revisions may remain empty.
14. The current artifact pointer was machine-local and nonportable.
15. A synthetic integration-test bundle was mislabeled as experimentally qualified.
