loader_path: third_party.tt_forge_models.gemma3_12b_glm_heretic_gguf.causal_lm.pytorch.loader
variant_id: 12B_GLM_Heretic_GGUF
arch: n150
status: DONE_FAIL
test_function: test_gemma3_12b_glm_heretic_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: null
trace_enabled: null
experimental_weight_dtype: null
failure_reason: "model size ~12B exceeds 10B single-chip capacity on n150"

# Benchmark added: test_gemma3_12b_glm_heretic_gguf

## Test
tests/benchmark/test_llms.py::test_gemma3_12b_glm_heretic_gguf

## Model
- HF name:    mradermacher/gemma-3-12b-it-vl-GLM-4.7-Flash-INSTRUCT-Thinking-Hybrid-Heretic-Uncensored-i1-GGUF
- Loader:     third_party.tt_forge_models.gemma3_12b_glm_heretic_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_12B_GLM_HERETIC_GGUF

## Early Exit Reason
Model size ~12B exceeds the 10B single-chip capacity on n150 (wormhole_b0).

At 12B parameters, this model will OOM during weight transfer on n150 regardless of
quantization knobs. This model requires a p150 (blackhole, ~25B cap) host to run as
a single-chip benchmark. On p150, the Q4_K_M GGUF quantization (~6.5 GB on disk)
provides enough headroom within the 25B single-chip limit.

## Test config landed
- optimization_level:        N/A (early exit)
- trace_enabled:             N/A (early exit)
- experimental_weight_dtype: N/A (early exit)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (early exit — no pytest run)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150
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
- SUMMARY.md (this file)

## tt-forge-models submodule
no change
