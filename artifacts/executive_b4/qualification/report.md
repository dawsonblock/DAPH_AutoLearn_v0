# B4 Executive Qualification Report

**Experiment:** `daph_executive_b4_qwen3_8b_hidden_states`
**Date:** 2026-08-02T08:24:28+0000
**Tasks:** 720 train / 288 dev / 432 final
**Actions:** action.reasoning.direct, action.retrieval.vector, action.reasoning.decompose
**Best representation:** `mean_prompt/layer_18/pca_128`
**Best fixed action:** `action.reasoning.direct`

## Gate Decision: FAIL

## Gates

| Gate | Value | Threshold | Result |
|------|-------|-----------|--------|
| Hidden > Fixed (LCB95) | 0.0163 | > 0 | PASS |
| Hidden > Sham (LCB95) | 0.1271 | > 0 | PASS |
| Gap capture | 42.2% | > 60% | FAIL |
| Positive group fraction | 52.1% | > 80% | FAIL |

## Ablation Results (Final Set)

| Policy | Regret | Utility | Gap Capture |
|--------|--------|---------|-------------|
| fixed_action.reasoning.direct | 0.1104 | 0.7711 | 0.0% |
| fixed_action.retrieval.vector | 0.1657 | 0.7158 | 0.0% |
| fixed_action.reasoning.decompose | 0.3550 | 0.5265 | 0.0% |
| subtype_only | 0.1032 | 0.7783 | 6.6% |
| logprob | 0.1041 | 0.7774 | 5.7% |
| hidden | 0.0638 | 0.8177 | 42.2% |
| hidden_plus_logprob | 0.2374 | 0.6442 | -115.0% |
| oracle | 0.0000 | 0.8815 | 100.0% |

## Sham Results

- Number of shams: 50
- Sham mean regret: 0.2311
- Sham std regret: 0.0169
- Hidden regret: 0.0638
- Hidden vs Sham (LCB95): 0.1271

## Key Findings

1. **Hidden states beat fixed-direct** (LCB95=+0.016 > 0): Qwen3's internal
   representations contain information about which action will succeed,
   beyond what always picking the best fixed action provides.

2. **Hidden states strongly beat sham** (LCB95=+0.127 > 0): The signal is
   real, not an artifact of feature dimensionality or training noise.

3. **Hidden states beat subtype-only** (regret 0.064 vs 0.103): The hidden
   states capture more than just the discrete subtype label — they encode
   within-subtype difficulty variation that the subtype label cannot.

4. **Best representation**: mean_prompt/layer_18/pca_128 — mid-layer
   (layer 18 of 36), mean pooling over prompt tokens, 128 PCA dimensions.

5. **hidden+logprob hurts** (regret 0.237): Combining 128 hidden + 10 logprob
   features causes overfitting with 720 training examples.
