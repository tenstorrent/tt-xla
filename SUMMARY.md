loader_path: third_party.tt_forge_models.glm_4_9b_0414_gguf.causal_lm.pytorch.loader
variant_id: 9B_0414_Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_glm_4_9b_0414_gguf
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
failure_reason: "GGUF architecture glm4 not supported in transformers==5.2.0; ValueError: GGUF model with architecture glm4 is not supported yet"

# Benchmark added: test_glm_4_9b_0414_gguf

## Test
tests/benchmark/test_llms.py::test_glm_4_9b_0414_gguf

## Model
- HF name:    unsloth/GLM-4-9B-0414-GGUF
- Loader:     third_party.tt_forge_models.glm_4_9b_0414_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GLM_4_9B_0414_Q4_K_M

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

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
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
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change

## Failure details
The GLM-4-9B-0414 GGUF model uses the `glm4` architecture type in its GGUF file,
which is not supported by transformers==5.2.0 (the version installed in the
tt-xla venv). The supported GGUF architectures in this version are: bloom,
deci, falcon, gemma2, gemma3, general, gpt2, lfm2, llama, mamba, mistral,
nemotron, phi3, qwen2, qwen2_moe, qwen3, qwen3_moe, stablelm, starcoder2, t5.
The error occurs before any device code runs:
  ValueError: GGUF model with architecture glm4 is not supported yet.
This is a loader/transformers compatibility issue — not a tt-xla or compiler
issue. To resolve, transformers would need to be updated to a version that
supports glm4 GGUF loading, or the loader would need to use the non-GGUF
path for this model.
