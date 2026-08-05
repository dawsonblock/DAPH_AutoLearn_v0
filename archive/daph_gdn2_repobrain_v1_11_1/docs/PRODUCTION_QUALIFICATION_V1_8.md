# DAPH GDN2 + RepoBrain v1.8 Production Qualification

v1.8 closes the largest serving-safety gap in v1.7: generated RepoLoRA factors can now be bound to a request-local `ContextVar` instead of mutating shared module state. Production servers MUST use `use_adapter_context`; `set_factors` remains only for legacy single-request experiments.

Additional qualification changes:

- Typed `CacheTensor(batch_axis=...)` metadata for explicit beam-reorder semantics.
- KL calibration requires safe **and useful** adapters and reports a bootstrap 95% CI; production CLI defaults to >=30 safe adapters.
- Cross-split fork/family/content-hash leakage validation helper.
- Fisher/SVD reports now include spectral reconstruction error, norm ratio, numerical rank and target rank.
- Adapter-isolation concurrency qualification (`scripts/qualify_adapter_isolation.py`).
- Final sign-off requires Fisher superiority to scalar scaling, adapter isolation, recovery, and multi-node artifacts.
- GDN2 cache qualification tracks error at multiple decode horizons.

No hardware or multi-node gate is marked passed by this source release.
