loader_path: third_party.tt_forge_models.autoglm_phone_gguf.causal_lm.pytorch.loader
variant_id: autoglm_phone_9b_q4_k_m
arch: p150
status: DONE_FAIL
test_function: test_autoglm_phone_9b_q4_k_m
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
failure_reason: "GGUF architecture glm4 not supported by transformers==5.2.0: ValueError: GGUF model with architecture glm4 is not supported yet. (transformers/modeling_gguf_pytorch_utils.py:478)"

# Benchmark added: test_autoglm_phone_9b_q4_k_m

## Test
tests/benchmark/test_llms.py::test_autoglm_phone_9b_q4_k_m

## Model
- HF name:    Luckybalabala/AutoGLM-Phone-9B-GGUF
- Loader:     third_party.tt_forge_models.autoglm_phone_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.AUTOGLM_PHONE_9B_Q4_K_M ("autoglm_phone_9b_q4_k_m")

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

## Failure details
The model loader fails during tokenizer/model loading with:
    ValueError: GGUF model with architecture glm4 is not supported yet.
    (venv/lib/python3.12/site-packages/transformers/modeling_gguf_pytorch_utils.py:478)

transformers==5.2.0 does not include support for the `glm4` architecture in its
GGUF loading utilities. The HuggingFace model `Luckybalabala/AutoGLM-Phone-9B-GGUF`
uses the glm4 GGUF architecture, which requires a newer version of transformers
that supports this architecture.

This is the same root cause as the previous n150 attempt with the older loader
`autoglm_phone_9b_gguf` (BlcaCola/AutoGLM-Phone-9B-GGUF). The fix belongs in
the tt-forge-models repo (either updating the loader to use a supported GGUF
model or waiting for transformers to add glm4 GGUF support).

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
