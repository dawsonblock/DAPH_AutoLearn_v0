# Third-party source and licensing notes

This package is a clean integration scaffold and does not bundle the uploaded third-party repositories.

## GatedDeltaNet-2

The uploaded NVIDIA repository states `Nvidia Source Code License-NC` and limits the Work and derivative works to non-commercial research/evaluation use. Do not copy its source into a commercial product without resolving licensing. The scaffold's `gdn2_adapter.py` loads an external implementation by Python module path; it does not redistribute NVIDIA code.

Relevant upstream files observed in the uploaded archive:

- `lit_gpt/gdn2.py`
- `lit_gpt/gdn2_ops/chunk_gdn2.py`
- `lit_gpt/gdn2_ops/fused_recurrent_gdn2.py`
- `lit_gpt/gdn2_ops/chunk_kda.py`

## DINOv2

The uploaded repository contains its own licenses and model cards. Use the upstream model/code under those terms. The scaffold only defines an interface/wrapper.

## gdn-tri-inverse

The uploaded repository is designed around Huawei Ascend NPU tooling and pto-kernels profiling. Its direct implementation is not bundled. The numerical-accuracy methodology is reflected only as generic tests.

## Code2LoRA paper

The design in this scaffold is inspired by the uploaded paper `Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution` (arXiv:2606.06492v1). This scaffold intentionally uses a smaller grouped-head hypernetwork instead of reproducing the paper's ~720M/~745M trainable hypernetwork.


## Optional runtime dependencies added in v1.6

- Hugging Face Transformers / model checkpoints: used only through optional integration paths and remain subject to their package/model licenses.
- Docker / gVisor / Firecracker: the included runner emits hardened Docker commands but does not bundle a container runtime or sandbox image.
- DeepSpeed: optional ZeRO-3 qualification script; DeepSpeed is not bundled in the ZIP.
