# DAPH GDN2 + RepoBrain v1.9 — Production Qualification

v1.9 does not claim production readiness. It closes qualification-harness gaps in v1.8 and keeps release sign-off fail-closed.

## Added in v1.9

- Executable adversarial Docker sandbox qualification (`scripts/qualify_sandbox.py`).
- Recovery round-trip qualification for cache schema v2 and RepoEvolution state (`scripts/qualify_recovery.py`).
- Multi-node NCCL collective continuity harness (`scripts/qualify_multinode.py`).
- Separate FSDP and ZeRO-3 JSON artifacts plus a strict distributed aggregator (`scripts/aggregate_distributed_qualification.py`).
- Optional repository-lineage enforcement directly in `make_split_manifest.py --lineage-input`.
- Stronger final sign-off validation: bootstrap CI structure, 1000-request isolation, both FSDP and ZeRO-3, sandbox image fingerprint + adversarial case count, cache schema v2, and true multi-node evidence.

## Required external gates

1. GDN2 CUDA/Triton output and state parity.
2. GDN2 incremental cache parity across multiple horizons.
3. 1000+ request-scoped adapter-isolation forwards.
4. Immutable split + cross-split lineage validation.
5. Real Repo2LoRA held-out evaluation.
6. Safe+useful KL calibration with >=30 qualifying adapters and bootstrap CI.
7. Fisher Pareto superiority over scalar scaling before Fisher is promoted.
8. Adversarial sandbox containment using immutable image fingerprint.
9. FSDP **and** ZeRO-3 end-to-end backward passes.
10. Cache/evolution checkpoint recovery.
11. At least two-node distributed qualification.

`production_signoff.py` requires every mandatory artifact. No weighted release score exists.
