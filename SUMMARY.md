loader_path: third_party.tt_forge_models.qwen_3_5_9b_base_text_nvfp4.causal_lm.pytorch.loader
variant_id: 9B_Base_Text_NVFP4
arch: p150
status: DONE_FAIL
test_function: test_qwen_3_5_9b_base_text_nvfp4
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
failure_reason: "Qwen3.5 hybrid architecture uses Qwen3_5DynamicCache (is_compileable=False) for its 24/32 linear_attention (GatedDeltaNet/Mamba) layers; benchmark harness provides StaticCache which lacks has_previous_state — incompatible with static-graph XLA compilation"

# Benchmark added: test_qwen_3_5_9b_base_text_nvfp4

## Test
tests/benchmark/test_llms.py::test_qwen_3_5_9b_base_text_nvfp4

## Model
- HF name:    osoleve/Qwen3.5-9B-Base-Text-NVFP4
- Loader:     third_party.tt_forge_models.qwen_3_5_9b_base_text_nvfp4.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_3_5_9B_BASE_TEXT_NVFP4 = "9B_Base_Text_NVFP4"

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure Details
The model failed at the CPU prefill step (before any TT device compilation) with:

    AttributeError: 'StaticCache' object has no attribute 'has_previous_state'

Traceback root:
    transformers/models/qwen3_5/modeling_qwen3_5.py:525 — Qwen3_5GatedDeltaNetLayer.forward()

Root cause: Qwen3.5 (osoleve/Qwen3.5-9B-Base-Text-NVFP4) is a hybrid architecture
with 32 decoder layers:
- 24 × `linear_attention` (GatedDeltaNet / Mamba-like recurrent layers)
- 8  × `full_attention` (standard transformer attention)

The hybrid cache class `Qwen3_5DynamicCache` is explicitly marked `is_compileable = False`
in transformers. It uses `torch.cat` for dynamic KV concatenation and maintains
per-layer recurrent states (`conv_states`, `recurrent_states`). The benchmark harness
provides a `StaticCache`, which is fundamentally incompatible:

1. `StaticCache` has no `has_previous_state` attribute (required by GatedDeltaNet layers).
2. `Qwen3_5DynamicCache.is_compileable = False` — model authors explicitly mark it
   as incompatible with static graph compilation.
3. The dynamic `torch.cat` updates in the KV cache are incompatible with TT-XLA's
   requirement for static shapes.

This cannot be fixed by modifying just the test. The benchmark harness
(`llm_benchmark.py` / `decode_utils.py`) would need a new hybrid-cache path,
AND the model's dynamic shapes would need to be resolvable for TT-XLA compilation.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        N/A
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Compute
- total_flops:             N/A
- breakdown.matmul:        N/A
- breakdown.linear:        N/A
- breakdown.conv2d:        N/A
- breakdown.sparse_matmul: N/A

### Inputs
- count:        N/A
- memory_bytes: N/A

### KV cache
- count:        N/A
- memory_bytes: N/A
- memory_gb:    N/A

### Params
- count:                  N/A
- effective_count:        N/A
- memory_bytes:           N/A
- memory_gb:              N/A
- effective_memory_bytes: N/A
- effective_memory_gb:    N/A
- embedding_count:        N/A
- embedding_memory_bytes: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A
- dram_time_ms:             N/A
- compute_time_ms_lofi:     N/A
- compute_time_ms_hifi2:    N/A
- compute_time_ms_hifi3:    N/A
- compute_time_ms_hifi4:    N/A

## Files changed
- tests/benchmark/test_llms.py (test_qwen_3_5_9b_base_text_nvfp4 added)

## tt-forge-models submodule
no change
