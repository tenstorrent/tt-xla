loader_path: third_party.tt_forge_models.kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16.causal_lm.pytorch.loader
variant_id: 9B_AutoRound_NVFP4_linearattn_BF16
arch: p150
status: DONE_FAIL
test_function: test_kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "hybrid model (linear attention + SSM Mamba layers) requires Mamba2Cache with has_previous_state attribute, but benchmark harness uses StaticCache: AttributeError: 'StaticCache' object has no attribute 'has_previous_state' in transformers/models/qwen3_5/modeling_qwen3_5.py:525"

# Benchmark added: test_kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16

## Test
tests/benchmark/test_llms.py::test_kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16

## Model
- HF name:    kaitchup/Qwen3.5-9B-autoround-NVFP4-linearattn-BF16
- Loader:     third_party.tt_forge_models.kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_3_5_9B_AUTOROUND_NVFP4_LINEARATTN_BF16

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Analysis
The model `kaitchup/Qwen3.5-9B-autoround-NVFP4-linearattn-BF16` is a **hybrid architecture**
combining standard Qwen3.5 transformer blocks with Mamba/SSM-style linear attention layers
(as indicated by `linearattn` in the model name and the presence of `linear_attn` submodules
with parameters like `A_log`, `conv1d`, `in_proj_qkv`, `dt_bias`, etc.).

The benchmark harness (`decode_utils.py`) initializes a `StaticCache` for the KV cache,
but the model's `linear_attn.forward()` checks `cache_params.has_previous_state`, an
attribute only present on `Mamba2Cache` (not on `StaticCache`).

Error traceback (abbreviated):
```
transformers/models/qwen3_5/modeling_qwen3_5.py:525: in forward
    and cache_params.has_previous_state
AttributeError: 'StaticCache' object has no attribute 'has_previous_state'
```

**Root cause:** The benchmark harness does not support hybrid transformer + linear attention
(SSM/Mamba) architectures that require a `Mamba2Cache`. This is an infrastructure
incompatibility, not a loader or compiler bug.

**Resolution needed:** The benchmark harness (or the model loader) would need to detect
hybrid architectures and provide an appropriate composite cache (`HybridCache` or
`Mamba2Cache` for the SSM layers) rather than `StaticCache`. This work belongs in the
benchmark infrastructure or tt-forge-models loader, not within the scope of this skill.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation or device execution.

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
