loader_path: third_party.tt_forge_models.qwen_3_5_0_8b_coder_calude_full.causal_lm.pytorch.loader
variant_id: 0.8B_Coder_Calude_Full
arch: p150
status: DONE_FAIL
test_function: test_qwen_3_5_0_8b_coder_calude_full
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
failure_reason: "AttributeError: 'StaticCache' object has no attribute 'has_previous_state' — Qwen3.5 is a hybrid Mamba/attention model; the benchmark harness passes StaticCache to Mamba linear_attn layers which require MambaCache with has_previous_state; incompatible with current benchmark harness"

# Benchmark added: test_qwen_3_5_0_8b_coder_calude_full

## Test
tests/benchmark/test_llms.py::test_qwen_3_5_0_8b_coder_calude_full

## Model
- HF name:    rahul7star/Qwen3.5-0.8B-Coder-Calude-Full
- Loader:     third_party.tt_forge_models.qwen_3_5_0_8b_coder_calude_full.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_3_5_0_8B_CODER_CALUDE_FULL

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
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

## Failure details
The test fails during CPU golden generation (before any device compilation) with:

    AttributeError: 'StaticCache' object has no attribute 'has_previous_state'

Traceback location:
    transformers/models/qwen3_5/modeling_qwen3_5.py:525
    inside Qwen3_5MambaDecoder.forward → cache_params.has_previous_state

Qwen3.5 is a hybrid Mamba/attention architecture. Its linear_attn (Mamba) layers
expect a MambaCache object (with has_previous_state), but the benchmark harness
in decode_utils.py provides a StaticCache unconditionally. This is a structural
incompatibility between the harness and this model's hybrid cache requirements.
The fix would require the benchmark harness to supply MambaCache for Mamba layers
alongside StaticCache for attention layers — a non-trivial change to decode_utils.py.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150
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
- tests/benchmark/test_llms.py (test function added)
- SUMMARY.md

## tt-forge-models submodule
no change
