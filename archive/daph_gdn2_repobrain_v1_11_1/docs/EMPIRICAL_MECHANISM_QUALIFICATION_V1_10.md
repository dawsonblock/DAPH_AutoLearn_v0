# DAPH v1.10 Empirical Mechanism Qualification

## Purpose

Stop architecture growth until the two central mechanisms are measured on real checkpoints and real repositories. Local unit tests establish software correctness only.

## RepoBrain comparison matrix

Every locked evaluation must include:

1. frozen base
2. base + RAG
3. base + shared LoRA
4. base + independently trained per-repository LoRA
5. base + Repo2LoRA-Lite
6. base + Repo2LoRA-Lite + RAG

The per-repository LoRA arm is the adaptation ceiling.

## Metrics

- repository next-token cross entropy
- exact match / edit similarity when applicable
- unit-test pass rate
- generic capability cross entropy / benchmark score
- KL(base || adapted)
- adapter generation latency
- peak VRAM

## Splits

Pilot: >=20 train, >=5 validation, >=5 untouched test.
Qualification: >=100 train, >=20 validation, >=20 untouched test.

Split construction must check repository identity plus fork ancestry, repository family and exact content hashes. Near-duplicate code detection should be added when source metadata permits.

## Controller promotion rule

The controller may attempt correction only after an unmodified adapter is useful on the repository task but causes measurable generic or execution regression. Every correction must be re-evaluated. Fisher correction is promoted only if it is Pareto-superior to simple scalar scaling on validation repositories.

No universal KL threshold or fixed Fisher percentage is allowed before empirical calibration.

## GDN2 qualification

GDN2 remains an independent track: FP32-reference output parity, recurrent-state parity, cached-decode parity, long-horizon drift, functional memory tasks, latency and VRAM. If it provides no measurable advantage, remove it.

## Release interpretation

A negative result from a fully qualified mechanism experiment is a valid outcome. Do not add new heuristics simply to force a gate to pass.
