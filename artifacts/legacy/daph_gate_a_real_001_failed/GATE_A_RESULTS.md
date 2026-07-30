# Gate A — Real-Model Qualification Experiment: Results

## Overview

Gate A tests whether experience from real counterfactual execution causes
AutoLearn's policy (P1) to outperform the frozen incumbent (P0, always-LLM)
on unseen tasks. The experiment uses a within-subtype crossover benchmark
with 6 subtypes (A–F) so that instance-level routing is non-trivial: a
constant-action policy cannot achieve zero regret.

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-1.5B-Instruct (float16) |
| Device | NVIDIA RTX 5090 (32 GB VRAM) |
| Hidden state layer | 10 (residual stream, last token) |
| Total tasks | 3,996 (1998 train / 498 dev / 498 cal / 1002 final) |
| Replicates R | 1 (deterministic generation, do_sample=False) |
| Bootstrap iterations | 10,000 (grouped, 14 groups) |
| max_new_tokens | 128 |
| Semantic parser | Disabled (DAPH_DISABLE_SEMANTIC_PARSE=1) |
| Seed | 7319 (sham seed: 19037) |

### Utility Function (frozen)

```
U = quality_weight * quality
  - lambda_time * latency_sec
  - lambda_compute * normalized_cost
  - lambda_risk * risk
```

| Parameter | Value |
|-----------|-------|
| quality_weight | 1.0 |
| lambda_time | 0.001 |
| lambda_compute | 0.1 |
| lambda_risk | 1.0 |

### Subtypes

| Subtype | Description | Symbolic capable? | LLM capable? |
|---------|-------------|-------------------|--------------|
| A | Structured integer arithmetic | Yes | Partially |
| B | NL: crate/box counting (multiplication) | No | Partially |
| C | NL: warehouse inventory (multiplication) | No | Mostly |
| D | Structured modular multiplication | Yes | No |
| E | NL: seats/tickets (addition) | No | Partially |
| F | NL: abstract counting (addition) | No | Partially |

Semantic parser is disabled so the symbolic backend fails on NL tasks
(B, C, E, F), creating crossover: symbolic wins on structured tasks
(A, D), LLM wins on NL tasks (B, C, E).

---

## Policies Evaluated

| Policy | Description |
|--------|-------------|
| **P0** | Frozen incumbent: always-LLM (no routing) |
| **P1** | AutoLearn: logistic router trained on real counterfactual utility |
| **Psham** | Same learner/training as P1, but with shuffled winning-action labels |
| **Poracle** | Upper bound: best action per task from measured counterfactual utility |
| **Hand router** | Human-designed subtype-based router |
| **Centroid** | Nearest-centroid classifier on hidden states |
| **Unweighted logistic** | Logistic regression without clipped-gap weighting |
| **Always symbolic** | Always route to symbolic backend |

---

## Final Evaluation Results (1002 tasks)

### Policy Performance

| Policy | Mean Utility | Mean Regret | CI vs P0 | Oracle Capture | Safety |
|--------|-------------|-------------|----------|----------------|--------|
| P0 (always-LLM) | 0.3541 | 0.2198 | — | 0.000 | OK |
| Always symbolic | 0.3297 | 0.2442 | [-0.133, +0.108] | -0.077 | OK |
| Hand router | 0.5717 | 0.0022 | [+0.137, +0.288] | 0.984 | OK |
| Centroid | 0.5475 | 0.0264 | [-0.430, -0.255] | -1.611 | OK |
| Unweighted logistic | 0.4504 | 0.1235 | [-0.380, -0.222] | -1.381 | OK |
| Psham (shuffled) | 0.3334 | 0.2405 | [-0.041, 0.000] | -0.077 | OK |
| **P1 (AutoLearn)** | **0.5467** | **0.0272** | **[+0.057, +0.248]** | **0.876** | **OK** |
| Poracle (upper bound) | 0.5739 | 0.0000 | [+0.140, +0.291] | 1.000 | OK |

### Paired Bootstrap Comparisons (10,000 iterations, 14 groups)

| Comparison | Mean ΔU | Median ΔU | 95% CI | Wins | Ties | Losses |
|-----------|---------|-----------|--------|------|------|--------|
| **P1 - P0** | **+0.1925** | 0.000 | [-0.041, +0.469] | 480 | 501 | 21 |
| **P1 - Psham** | **+0.2132** | 0.000 | **[+0.004, +0.483]** | 475 | 527 | 0 |
| Psham - P0 | -0.0207 | 0.000 | [-0.077, 0.000] | 5 | 976 | 21 |
| Poracle - P0 | +0.2198 | 0.005 | [+0.008, +0.489] | 756 | 246 | 0 |
| Poracle - P1 | +0.0272 | 0.000 | [+0.001, +0.099] | 281 | 721 | 0 |
| P1 - Hand | -0.0251 | 0.000 | [-0.097, +0.002] | 167 | 809 | 26 |

### Oracle Gap Capture

| Metric | Value |
|--------|-------|
| Oracle headroom H = U(Poracle) - U(P0) | 0.2198 |
| P1 improvement = U(P1) - U(P0) | 0.1925 |
| **Oracle capture η** | **0.876** |

P1 recovers **87.6%** of the available routing value. The hand router
captures 98.4%. P1 has not yet surpassed the hand router, but it captures
the large majority of routing benefit.

---

## Crossover Analysis by Subtype

### Train Split (1998 tasks, 333 per subtype)

| Subtype | N | S wins | L wins | Ties | P(S wins) |
|---------|------|--------|--------|------|-----------|
| A | 333 | 125 | 0 | 208 | 0.375 |
| B | 333 | 0 | 156 | 177 | 0.000 |
| C | 333 | 0 | 261 | 72 | 0.000 |
| D | 333 | 302 | 0 | 31 | 0.907 |
| E | 333 | 0 | 50 | 283 | 0.000 |
| F | 333 | 0 | 0 | 333 | 0.000 |

### Final Split (1002 tasks, 167 per subtype)

| Subtype | N | S wins | L wins | Ties | Reg(S) | Reg(L) | Reg(Hand) | Reg(P1) |
|---------|------|--------|--------|------|--------|--------|-----------|---------|
| A | 167 | 60 | 0 | 107 | 0.000 | 0.368 | 0.000 | 0.156 |
| B | 167 | 0 | 88 | 79 | 0.524 | 0.002 | 0.002 | 0.002 |
| C | 167 | 0 | 132 | 35 | 0.786 | 0.001 | 0.001 | 0.001 |
| D | 167 | 155 | 0 | 12 | 0.000 | 0.937 | 0.000 | 0.000 |
| E | 167 | 0 | 26 | 141 | 0.155 | 0.004 | 0.004 | 0.004 |
| F | 167 | 0 | 0 | 167 | 0.000 | 0.005 | 0.005 | 0.000 |

**Crossover pattern confirmed:**
- Symbolic wins on **A** (structured arithmetic) and **D** (modular multiplication)
- LLM wins on **B**, **C**, **E** (NL arithmetic tasks where symbolic is unavailable)
- **F** is a tie (both backends produce the same utility)

P1 matches the hand router on B, C, D, E, F. The gap is on subtype A,
where P1 has regret 0.156 vs hand router's 0.000 — P1 sometimes routes
A tasks to LLM when symbolic would be better.

---

## Gate A Decision

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| A1: Practical effect | mean(P1-P0) > ε=0.01 | +0.193 | **PASS** |
| A2: Statistical LCB | LCB95% > 0 | -0.041 | **FAIL** |
| A3: Sham control | mean(P1-Psham) > 0, CI excludes 0 | +0.213, CI=[+0.004, +0.483] | **PASS** |
| A4: Safety | No safety violations | OK | **PASS** |
| A6: Real evidence | Real model, real execution | Confirmed | **PASS** |

### **Gate A: FAIL** (A2 fails)

The lower 95% confidence bound on P1-P0 is -0.041, crossing zero. The
point estimate is strongly positive (+0.193) and P1 decisively beats the
sham control (CI entirely above zero), but the grouped bootstrap CI is
wide due to:

1. **Only 14 groups** — the grouped bootstrap resamples at the group
   level (linguistic template groups), and with 14 groups the CI is
   inherently wide.
2. **Heavy-tailed utility distribution** — most tasks are ties (utility
   difference = 0), but a few tasks have large utility swings (±1.0),
   inflating the bootstrap variance.
3. **Backend co-availability = 0.333** — symbolic is only available on
   1/3 of tasks, limiting the number of decisive comparisons.

---

## Dev Selection (498 tasks)

| Method | Dev Regret |
|--------|-----------|
| Always LLM (P0) | 0.2082 |
| Always symbolic | 0.2098 |
| Hand router | 0.0024 |
| Centroid | 0.0328 |
| Unweighted logistic | 0.0178 |
| **P1 (logistic weighted)** | **0.0035** |
| Psham (shuffled) | 0.2082 |

P1's dev regret (0.0035) is very close to the hand router (0.0024),
confirming that the learned policy generalizes well to held-out data.

---

## Calibration

| Parameter | Value |
|-----------|-------|
| tau_ood (Mahalanobis OOD threshold) | 511.23 |
| tau_conf (confidence threshold) | 0.50 |
| Cal utility at tau_conf | 0.5283 |

---

## Effective Sample Size

| Metric | Value |
|--------|-------|
| ESS | 875.96 |
| Zero-weight fraction | 0.562 |
| N decisive (non-tie) | 876 |
| N ties | 1,122 |

56% of training tasks are ties (both backends produce equal utility),
which is expected given the crossover design: on F tasks both backends
tie, and on many A tasks both backends succeed.

---

## Interpretation

### What P1 Learned

P1 learned a routing policy that:
- Routes **structured arithmetic** (A, D) to the symbolic backend
- Routes **NL arithmetic** (B, C, E) to the LLM
- Routes **abstract counting** (F) to either (both tie)

This is essentially the same policy as the hand router, but learned
automatically from real execution data without human-designed rules.

### Why Gate A Fails on A2

The gate fails because the statistical lower confidence bound on
P1-P0 crosses zero. This is a **power issue**, not a signal issue:

- The **point estimate** is strongly positive: +0.193
- The **sham control** is decisively beaten: CI = [+0.004, +0.483]
- The **oracle capture** is high: 87.6%
- P1 wins **480 tasks** vs only **21 losses** against P0

The wide CI is driven by the grouped bootstrap structure (14 groups)
and the bimodal utility distribution (most ties, some large swings).

### What Would Make Gate A Pass

1. **More groups** — increasing the number of linguistic template
   groups would tighten the grouped bootstrap CI.
2. **More decisive tasks** — increasing the fraction of tasks where
   backends differ (currently 44% decisive) would reduce variance.
3. **Larger final set** — 1002 final tasks is substantial, but the
   grouped bootstrap resamples groups, not tasks, so more groups
   matter more than more tasks per group.

### P1 vs Hand Router

P1 has not yet surpassed the hand router:
- P1 regret = 0.027 vs Hand regret = 0.002
- P1-Hand CI = [-0.097, +0.002] — close but not quite crossing zero
- The gap is entirely on subtype A, where P1 misroutes some
  structured tasks to LLM

The hand router benefits from perfect subtype knowledge (it routes
by subtype). P1 must infer routing from hidden states alone, which
is harder on A tasks where both backends can succeed.

---

## Artifact

Results saved to:
`artifacts/gate_a_experiment_result.json`

Source hash: `cad335169d659e7b049f5cbf3dff7f6521df7f7b955bf0133263ddb3dc9aabca`
Model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
