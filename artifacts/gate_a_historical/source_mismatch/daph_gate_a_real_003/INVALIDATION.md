# Invalidation Notice — daph_gate_a_real_003

**Status:** INVALIDATED
**Reason:** Source hash mismatch
**Date:** 2026-07-31

## Original source hash
`51ad856027a33c21e4a477fbc2c7cf57ccd67c292602e42339f09c5fab933809`

## Current source hash (at invalidation time)
The current repository source tree no longer matches the hash recorded in
the original qualification bundle. The codebase has been repaired under
v0.3.10.5-alpha to fix policy semantics, representation selection, and
baseline reporting.

## Why qualification is no longer valid
Gate A qualification requires that the frozen source hash matches the
distributed source tree. After the v0.3.10.5-alpha repairs, the source
hash changed, invalidating this bundle.

Additionally, the original run used `always_llm` as P0 (the primary
comparator), but the repaired protocol requires `best_fixed` (selected
from development data) as the primary comparator.

## Historical evidence value
The numerical task records (final_predictions.json, final_experiences.json)
remain useful as historical evidence of benchmark-specific routing
improvements. They are NOT valid evidence for the current protocol.

## Replacement
The requalification run `daph_gate_a_real_005_requal` (or successor) under
the repaired protocol replaces this bundle as the current qualified
artifact.
