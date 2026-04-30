# Remediation Summary: glm_4_7_awq-causal_lm-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_awq/causal_lm/pytorch-single_device-inference]

## Result
SILICON_PASS — static per-expert masked matmul avoids grouped_mm histc error and batched_mm L1 CB overflow

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
glm4moe-grouped-mm-histc-int-and-batched-mm-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The reported failure (`TT_FATAL: Chip 0 logical eth core (x=0,y=8) connects to a remote mmio device`) was a transient hardware init error on the prior CI run. On reproduction, two real bugs were found:

1. `NotImplementedError: "histogram_cpu" not implemented for 'Int'` — `grouped_mm_experts_forward` calls `torch.histc` with an Int tensor which is unsupported on XLA.

2. After switching to `batched_mm`: `TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=9)] grow to 2208768 B which is beyond max L1 size of 1572864 B` — `batched_mm_experts_forward` gathers `self.gate_up_proj[expert_ids]` using a dynamic 1D tensor index, which MLIR lowers as an embedding lookup. The resulting embedding row (2*moe_intermediate_size × hidden_size) exceeds the 1.5 MB L1 CB budget.

## Root cause
`Glm4MoeNaiveMoe` is decorated with `@use_experts_implementation`, which dispatches `forward` to an implementation named by `config._experts_implementation`. The default for PyTorch 2.9+ is `"grouped_mm"`, which uses `torch.histc` with an Int-cast input — unsupported on XLA. Switching to `"batched_mm"` avoids histc but performs a dynamic 3D gather (`gate_up_proj[expert_ids]` where `expert_ids` is a device tensor), which tt-mlir lowers as a 2D embedding table whose row size overflows L1 CB (2.1 MB allocated vs 1.5 MB limit).

This is the same class of bug as GraniteMoeHybrid (`pjrt-device-to-host-transfer`/`embedding-rm-weight-row-exceeds-l1`) and GLM-5 (`grouped_mm_experts_forward`).

## Fix
Registered a custom `_tt_static_glm4_moe_forward` in `ALL_EXPERTS_FUNCTIONS["tt_static_glm4_moe"]` and set `config._experts_implementation = "tt_static_glm4_moe"` before `from_config`. The static implementation loops over `range(num_experts)` (Python ints), so dynamo unrolls it into 8 separate `F.linear` calls with constant weight slices — no dynamic gather, no histc. `top_k_weights` is cast to model dtype before accumulation to prevent float32 promotion from the router's `float32 e_score_correction_bias`.

File changed: `glm_4_7_awq/causal_lm/pytorch/loader.py` in `tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    69.89s
- Tier A attempts: N/A

## Files changed
- `glm_4_7_awq/causal_lm/pytorch/loader.py` (tt-forge-models, commit 31b84b7c1db51bf7a420405a07937ddc0be10c6b)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | dfd3ef5282325eb15522c9d1cb8c52fdff0992ea |
| tt-xla          | 04e6a540e65bc625599f7b368d982eaabca069a0 |
| tt-forge-models | 31b84b7c1db51bf7a420405a07937ddc0be10c6b |
