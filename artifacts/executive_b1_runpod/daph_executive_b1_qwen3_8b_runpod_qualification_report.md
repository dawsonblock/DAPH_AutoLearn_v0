# Executive Qualification Report

**Experiment:** `daph_executive_b1_qwen3_8b_runpod`
**Date:** 2026-08-01T22:34:00.741742+00:00
**Tasks:** 24  |  **Groups:** 8
**Actions:** action.reasoning.direct, action.retrieval.vector, action.reasoning.decompose

## Gate Decision: FAIL

## Primary Endpoint: P1 - P0

| Metric | Value |
|--------|-------|
| Point estimate | 0.1699 |
| 95% LCB | -0.0012 |
| 95% UCB | 0.4194 |
| Bootstrap mean | 0.1697 |
| Bootstrap std | 0.1033 |

## Utility Summary

| Policy | Mean Utility |
|--------|-------------|
| P1 (learned) | 0.8910 |
| P0 (baseline) | 0.7211 |
| Oracle | 0.9775 |

| Metric | Value |
|--------|-------|
| Oracle gap (oracle - P0) | 0.2564 |
| Oracle gap capture | 66.3% |
| Positive group fraction | 87.5% |
| Worst subtype regression | 0.0000 |

## Per-Action Utilities (always-action baseline)

| Action | Always-Action Utility |
|--------|---------------------|
| `action.reasoning.direct` | 0.7211 |
| `action.retrieval.vector` | 0.8910 |
| `action.reasoning.decompose` | 0.9396 |

## Per-Group Breakdown

| Group | N | P1 mean | P0 mean | P1-P0 |
|-------|---|---------|---------|-------|
| g0 | 3 | 0.9771 | 0.3056 | 0.6715 |
| g1 | 3 | 0.9793 | 0.3022 | 0.6772 |
| g2 | 3 | 0.9799 | 0.9746 | 0.0054 |
| g3 | 3 | 0.9759 | 0.9671 | 0.0088 |
| g4 | 3 | 0.9818 | 0.9715 | 0.0103 |
| g5 | 3 | 0.9740 | 0.9715 | 0.0026 |
| g6 | 3 | 0.2808 | 0.3045 | -0.0236 |
| g7 | 3 | 0.9792 | 0.9721 | 0.0071 |
