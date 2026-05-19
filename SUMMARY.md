loader_path: third_party.tt_forge_models.glm_z1_9b_0414_heretic_gguf.causal_lm.pytorch.loader
variant_id: Z1_9B_0414_heretic_GGUF
arch: p150
status: DONE_FAIL
test_function: test_glm_z1_9b_0414_heretic_gguf
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
failure_reason: "GGUF model with architecture glm4 is not supported by transformers 5.2.0; glm4 is not in GGUF_CONFIG_MAPPING"

# Benchmark added: glm_z1_9b_0414_heretic_gguf

## Test
tests/benchmark/test_llms.py::test_glm_z1_9b_0414_heretic_gguf

## Model
- HF name:    mradermacher/GLM-Z1-9B-0414-heretic-GGUF
- Loader:     third_party.tt_forge_models.glm_z1_9b_0414_heretic_gguf.causal_lm.pytorch.loader
- Variant:    Z1_9B_0414_heretic_GGUF

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

## Failure
The test failed during model loading with:

    ValueError: GGUF model with architecture glm4 is not supported yet.

The GLM4 GGUF architecture is not present in transformers 5.2.0's `GGUF_CONFIG_MAPPING`.
The installed transformers supports: llama, mistral, qwen2, qwen2_moe, qwen3, qwen3_moe,
falcon, phi3, bloom, t5, stablelm, gpt2, starcoder2, mamba, nemotron, gemma2, gemma3, etc.
but not glm4. This requires a newer transformers release or a fix in the tt-forge-models
loader to not rely on GGUF loading for the tokenizer configuration.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
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
no change
