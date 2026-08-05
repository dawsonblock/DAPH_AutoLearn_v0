# Source review summary

## GatedDeltaNet-2 uploaded archive

Observed architecture class: `lit_gpt/gdn2.py::GatedDeltaNet2`.

Constructor parameters include hidden size, value expansion, head dimension, number of heads, chunk/recurrent mode, optional short convolution, negative-eigenvalue option, layer index, and normalization epsilon.

The upstream config exposes `gdn2_per_layer`; the provided `swa_gdn2_1.3B` profile uses `gdn2_per_layer=2`, which supports the fixed alternating-layer experiment in this scaffold.

The upstream license in the uploaded archive is `Nvidia Source Code License-NC`, with non-commercial research/evaluation use limitation. Therefore this package does not copy those kernels.

## Code2LoRA uploaded paper

The paper's repository encoder uses frozen Qwen3-Embedding-0.6B, 4096-token chunks with 512-token overlap, file-level mean pooling, and repository-level weighted mean + max pooling. Its Static/Evo systems generate rank-16 LoRA for seven attention/MLP projection types. The paper reports a large hypernetwork (~720M Static, ~745M Evo) and H100-80GB training.

This scaffold keeps the architectural idea but intentionally reduces it to a 512-dimensional trunk, lower rank, selected targets, and layer groups.

## DINOv2 uploaded archive

The model implementation exposes `forward_features` with normalized class and patch tokens. The wrapper in this scaffold consumes `x_norm_clstoken` and `x_norm_patchtokens` only, keeping the backend replaceable.

## gdn-tri-inverse uploaded archive

The repository is explicitly a profiling project for Gated DeltaNet triangular inverse kernels around Huawei Ascend tooling/pto-kernels. Its direct kernel path is not used. The relevant transferable idea for current DAPH work is numerical validation and profiling before optimization.
