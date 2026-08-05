# DAPH GDN2 + RepoBrain v1.1 Remediation

Implemented from the audit:

1. Repo2LoRA factor scaling is applied once to B rather than independently to A and B.
2. `project_and_refactor_factors` materializes a group delta, performs guarded geometry correction, and uses balanced truncated SVD to recover A/B at the configured rank.
3. `RepoLoRALinear` applies generated factors dynamically and patch helpers target q_proj/v_proj/o_proj/down_proj-style modules by layer path.
4. `HybridSequenceCache` stores both attention KV tensors and GDN2 recurrent states.
5. `ExternalGDN2Adapter` normalizes recurrent-state device/dtype before dispatch.
6. `ValidationLossProbe`, `LogitKLDriftProbe`, and `CodeExecutionProbe` are concrete probe implementations. The execution probe delegates to an external sandbox runner rather than launching untrusted code itself.
7. `empirical_fisher_diagonal` accumulates squared gradients in FP32, with an optional per-sample mode to avoid the batch-gradient-square approximation.

## Qualification status

Local scaffold tests: 21 passed.
`validate_install.py`: passed.

Not yet claimed:

- Real Qwen checkpoint patching has not been executed in this container.
- External NVIDIA GDN2 kernels have not been numerically qualified here because they are not vendored and require their upstream runtime/dependencies.
- The code-execution probe provides the adapter contract; a production sandbox must be supplied by DAPH.
- BF16/CUDA end-to-end qualification still needs to run on the target GPU.
