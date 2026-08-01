# v1.6 Real-Model Experimental Protocol

## Immutable splits

Pilot: 20 train repositories, 5 validation repositories, 5 untouched test
repositories. Persist split manifests and SHA-256 hashes before training.

## Baselines

- Frozen base model.
- Simple RAG.
- Shared LoRA.
- Repo2LoRA-Lite.

## Primary metrics

- Repository next-token CE / perplexity.
- Generic held-out CE.
- Exact match / edit similarity.
- Unit-test execution pass rate.
- Forward KL(base || adapted), token weighted.

## Correction study

First compare scalar scaling `[1,.9,.75,.5,.25]`. Fisher earns a place only if
it produces a validation Pareto point that scalar scaling cannot dominate.

Hard Fisher: exact top-k at 5/10/15/20/30%.
Soft Fisher: gamma=1 initially and lambda in [.05,.1,.25,.5,1,2].

Every dense correction is refactored to rank-r and must report retained SVD
energy and relative reconstruction error. Low retained energy is a correction
failure, not an acceptable silent approximation.

## Statistical discipline

All hyperparameter/controller choices use validation repositories only. Test
repositories are evaluated once after policy freeze. Report paired bootstrap
confidence intervals and paired-discordant tests for binary outcomes where
appropriate. Null results are retained as final results rather than prompting
same-split rule search.
