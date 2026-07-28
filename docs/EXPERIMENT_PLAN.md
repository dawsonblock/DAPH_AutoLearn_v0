# Corrected experimental plan

## Gate 0: integrity

1. Generate or ingest immutable task records and record every SHA-256.
2. Run `detect_leakage` and require zero exact, normalized, fingerprint, and
   generator/template-family overlap across splits.
3. Print the imported script/module paths and package versions.
4. Freeze decoding, scoring, safety, seeds, reward, and verifier versions.
5. Reserve final-test access atomically only after configuration freeze.
6. Preserve historical baseline/memory/sham outputs instead of overwriting them.

## Core ablation matrix

| Arm | Router | Memory | Steering | Symbolic |
|---|---:|---:|---:|---:|
| A baseline | no | no | no | no |
| B memory | no | real | no | no |
| C sham | no | sham | no | no |
| D symbolic | deterministic | no | no | yes |
| E symbolic+memory | deterministic | real | no | yes |
| F symbolic+steering | deterministic/learned | no | yes | yes |
| G full | deterministic/learned | real | yes | yes |

Interpretation:

- A vs B: procedural-memory effect.
- A vs D: symbolic-architecture effect.
- D vs F: incremental steering effect on routing/planning metrics, not arithmetic correctness alone.
- F vs G: incremental memory effect after exact execution exists.

## Steering research

Keep separate vector families:

- reasoning policy: `decompose`, `verify`, `reconsider`
- tool policy: `exact_symbolic_pressure`, `invoke_symbolic_tool`

Start with a residual-stream layer sweep. Treat any literature-based depth range as a prior, not a truth. After identifying useful layers, localize heads causally with patching/ablation. Do not rank heads by raw attention magnitude.

Primary tool-policy metrics:

- precision / recall / F1
- false-positive tool calls on easy/non-symbolic controls
- false-negative tool calls on hard exact-computation tasks
- malformed action rate
- final exact accuracy
- latency and generated-token cost

Required v0.3.8 controls:

- complete-label teacher-forced route scoring;
- identical ordered tasks for the real vector and every random direction;
- at least 500 matched-norm random directions for headline eligibility;
- finite-sample empirical p-values, not a five-sample z-score;
- paired confidence intervals clustered by generator family;
- steering clamp and decay configuration recorded in the manifest.

## OOD benchmark

The original five-item notebook set is a demonstration, not a statistical benchmark. Build and freeze a larger OOD set with deterministic seeds containing:

- easy addition/subtraction controls
- multi-digit multiplication
- signed arithmetic
- modular multiplication
- nested expressions
- non-symbolic controls

Never tune vector strength, layer, head set, routing thresholds, verifier
behavior, or stopping rules on the final test split. Repeated development
access turns a split into training data regardless of its filename.


## v0.3.1 connection rule

`steered_auto` is the only arm in which a tool-policy vector is allowed to influence the call/no-call decision. Reasoning-policy vectors are applied only after an LLM backend has been selected. This prevents the experimental confound where one generic vector simultaneously changes routing and answer generation. Malformed route outputs fall back to deterministic auto routing and must be counted separately.
