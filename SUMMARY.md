loader_path: third_party.tt_forge_models.autoglm_phone_9b_gguf.causal_lm.pytorch.loader
variant_id: 9B_Phone_Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_autoglm_phone_9b_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "GGUF architecture glm4 not supported by current transformers: ValueError: GGUF model with architecture glm4 is not supported yet. (transformers/modeling_gguf_pytorch_utils.py:478)"

# Benchmark added: test_autoglm_phone_9b_gguf

## Test
tests/benchmark/test_llms.py::test_autoglm_phone_9b_gguf

## Model
- HF name:    BlcaCola/AutoGLM-Phone-9B-GGUF
- Loader:     third_party.tt_forge_models.autoglm_phone_9b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.AUTOGLM_PHONE_9B_Q4_K_M ("9B_Phone_Q4_K_M")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (model loading failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure
The test fails at model loading time with:

    ValueError: GGUF model with architecture glm4 is not supported yet.

This error is raised in transformers 5.2.0 at
`transformers/modeling_gguf_pytorch_utils.py:478` when attempting to load
`AutoGLM-Phone-9B-Q4_K_M.gguf` from `BlcaCola/AutoGLM-Phone-9B-GGUF`.
The `glm4` GGUF architecture is not in the transformers GGUF support map.
This is a dependency issue outside the scope of this skill — the fix
requires a transformers upgrade that adds glm4 GGUF support.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (model did not load)
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
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change (submodule at 25ffef52dc)
