# B4 v3 Qualification Report (harder tasks — FAILED iteration)

**Experiment:** `daph_executive_b4_qwen3_8b_hidden_states`
**Date:** 2026-08-02T09:03:27+0000
**Tasks:** 720 train / 288 dev / 432 final
**Actions:** action.reasoning.direct, action.retrieval.vector, action.reasoning.decompose
**Best representation:** `mean_prompt/layer_27/pca_64`
**Best fixed action:** `action.reasoning.direct`

## Gate Decision: FAIL (0/4 gates passed)

## Gates

| Gate | Value | Threshold | Result |
|------|-------|-----------|--------|
| Hidden > Fixed (LCB95) | -0.0748 | > 0 | FAIL |
| Hidden > Sham (LCB95) | -0.0403 | > 0 | FAIL |
| Gap capture | -36.7% | > 60% | FAIL |
| Positive group fraction | 22.9% | > 80% | FAIL |

## Ablation Results (Final Set)

| Policy | Regret | Utility | Gap Capture |
|--------|--------|---------|-------------|
| fixed_action.reasoning.direct | 0.1280 | 0.6761 | 0.0% |
| fixed_action.retrieval.vector | 0.2366 | 0.5674 | 0.0% |
| fixed_action.reasoning.decompose | 0.4044 | 0.3997 | 0.0% |
| subtype_only | 0.0485 | 0.7555 | 62.1% |
| logprob | 0.1213 | 0.6828 | 5.2% |
| hidden | 0.1749 | 0.6291 | -36.7% |
| hidden_plus_logprob | 0.2379 | 0.5662 | -85.9% |
| oracle | 0.0000 | 0.8041 | 100.0% |

## Sham Results

- Number of shams: 50
- Sham mean regret: 0.2594
- Sham std regret: 0.0694
- Hidden regret: 0.1749
- Hidden vs Sham (LCB95): -0.0403

## Analysis: Why v3 did worse than v2

v3 widened the oracle gap (0.128 vs 0.111) as intended, but:
1. **Subtype became too predictive**: subtype_only gap capture jumped from
   6.6% (v2) to 62.1% (v3). The harder tasks made the subtype label almost
   sufficient — hidden states had little left to add.
2. **Decompose got worse**: utility 0.400 (v3) vs 0.527 (v2). The harder
   multi-step problems caused decompose to fail more, adding noise.
3. **Hidden states overfit**: dev regret 0.172 (v3) vs 0.071 (v2). The
   more complex task distribution made the feature→action mapping harder
   to learn with 720 training examples.

**Conclusion**: v2's moderate difficulty was the sweet spot. The hidden
states could learn the within-subtype variation when the task distribution
wasn't too extreme. v3's harder tasks made the subtype label too dominant.
