# Training strategy

## Stage A — GDN2 architectural training

Goal: determine whether the recurrent mixer improves DAPH at fixed parameter/compute budget.

Controls:

- match training tokens and optimizer steps
- report parameter count and FLOPs/token
- use same tokenizer/data order where feasible
- compare validation perplexity and long-context/retrieval tasks

Do not combine GDN2, Repo2LoRA, new routing, and a new dataset in one run. That destroys causal attribution.

## Stage B — Repo2LoRA-Static

Freeze the universal DAPH checkpoint. Cache repository embeddings. Train only the hypernetwork.

Suggested optimizer start point:

- AdamW
- LR search: 1e-4, 3e-4, 1e-3
- weight decay: 0.01
- gradient clipping: 1.0
- BF16 where stable

Adapter magnitude should start small. The scaffold initializes generated factor scales to 0.01.

Track per repository:

- task loss/score
- adapter Frobenius norm by layer group/module
- protected-domain KL
- geometry metrics

## Stage C — Controller calibration

The controller itself should not be tuned on the final test set. Build a calibration set of good and harmful adapters. Learn or hand-tune thresholds there, then freeze policy.

## Stage D — Evolution

Replay commits in order. Truncated BPTT is an implementation option, but verify that state detachment does not erase the long-horizon signal you care about. Compare against no-BPTT state updates and periodic full snapshot regeneration.
