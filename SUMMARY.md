# Remediation Summary: glm_4_7_awq-causal_lm-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_awq/causal_lm/pytorch-single_device-inference]

## Result
XFAIL — GLM-4.7-AWQ is a 253B-parameter MoE model; AWQ 4-bit checkpoint is ~126 GB, far exceeding n150 DRAM (12 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-23 22:59:05.703 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=8) connects to a remote mmio device (assert.hpp:104)

## Root cause
The reported TT_FATAL is a well-known transient hardware initialization error (CI framework explicitly excludes "connects to a remote mmio device" from the tt_fatal failure category; 80+ tests hit this transiently and pass on retry). The underlying model cannot run on n150: GLM-4.7-AWQ is a 253B-parameter MoE model (92 layers, 5120 hidden size, 160 routed experts) whose AWQ 4-bit checkpoint is ~126 GB — far beyond n150 DRAM (12 GB). The previous remediation commit on this branch incorrectly marked SILICON_PASS by loading a tiny random model via from_config with heavily reduced dimensions (6 layers, hidden_size=1024, n_routed_experts=8) — a forbidden model-trimming workaround.

## Fix
- `glm_4_7_awq/causal_lm/pytorch/loader.py` (tt-forge-models): removed dimension trimming; switched from from_config to from_pretrained; kept the static MoE forward implementation (the grouped_mm histc-on-Int and batched_mm L1 CB overflow bugs are real and documented in glm4_7_awq_static_moe_experts.md).
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla): added KNOWN_FAILURE_XFAIL entry.

## Verification
- pytest exit: not-run
- Hardware:    n150
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `glm_4_7_awq/causal_lm/pytorch/loader.py` (tt-forge-models, commit fc16ff01cd)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | dfd3ef5282325eb15522c9d1cb8c52fdff0992ea |
| tt-xla          | (this branch HEAD) |
| tt-forge-models | fc16ff01cd |
