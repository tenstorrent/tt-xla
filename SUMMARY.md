loader_path: third_party.tt_forge_models.14b_qwen2_5_kunou_v1_gguf.causal_lm.pytorch.loader
variant_id: 14B_Q4_K_M
arch: n150
status: DONE_FAIL
test_function: test_14b_qwen2_5_kunou_v1_gguf
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
failure_reason: "model size ~14B exceeds 10B single-chip capacity"

# Benchmark added: test_14b_qwen2_5_kunou_v1_gguf

## Test
tests/benchmark/test_llms.py::test_14b_qwen2_5_kunou_v1_gguf

## Model
- HF name:    mradermacher/14B-Qwen2.5-Kunou-v1-GGUF
- Loader:     third_party.tt_forge_models.14b_qwen2_5_kunou_v1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN2_5_KUNOU_V1_14B_Q4_K_M (value: "14B_Q4_K_M")

## Test config landed
- optimization_level:        N/A (early exit — model rejected before test was added)
- trace_enabled:             N/A
- experimental_weight_dtype: N/A
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
Source JSON: N/A
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
None — early exit before any test code was added.

## tt-forge-models submodule
no change
