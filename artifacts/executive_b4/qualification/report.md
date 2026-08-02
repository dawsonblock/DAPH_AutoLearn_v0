# B4 Executive Qualification Report

**Experiment:** `daph_executive_b4_qwen3_8b_hidden_states`
**Date:** 2026-08-02T07:38:35+0000
**Tasks:** 640 train / 256 dev / 384 final
**Actions:** action.reasoning.direct, action.retrieval.vector, action.reasoning.decompose
**Best representation:** `mean_content/layer_36/pca_256`
**Best fixed action:** `action.reasoning.direct`

## Gate Decision: FAIL

## Gates

| Gate | Value | Threshold | Result |
|------|-------|-----------|--------|
| Hidden > Fixed (LCB95) | -0.0597 | > 0 | FAIL |
| Hidden > Sham (LCB95) | 0.0529 | > 0 | PASS |
| Gap capture | -1.4% | > 60% | FAIL |
| Positive group fraction | 35.4% | > 80% | FAIL |

## Ablation Results (Final Set)

| Policy | Regret | Utility | Gap Capture |
|--------|--------|---------|-------------|
| fixed_action.reasoning.direct | 0.1681 | 0.7567 | 0.0% |
| fixed_action.retrieval.vector | 0.2516 | 0.6733 | 0.0% |
| fixed_action.reasoning.decompose | 0.5278 | 0.3971 | 0.0% |
| subtype_only | 0.1284 | 0.7964 | 23.6% |
| logprob | 0.1681 | 0.7567 | 0.0% |
| hidden | 0.1704 | 0.7544 | -1.4% |
| hidden_plus_logprob | 0.2508 | 0.6740 | -49.2% |
| oracle | 0.0000 | 0.9249 | 100.0% |

## Sham Results

- Number of shams: 50
- Sham mean regret: 0.2966
- Sham std regret: 0.0422
- Hidden regret: 0.1704
- Hidden vs Sham (LCB95): 0.0529
