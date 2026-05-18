loader_path: third_party.tt_forge_models.bielik_11b_v3_gguf.causal_lm.pytorch.loader
variant_id: 11B_V3.0_INSTRUCT_GGUF
arch: p150
status: DONE_PASS
test_function: test_bielik_11b_v3_gguf
samples_per_second: 22.03
ttft_ms: 486.23
prefill_pcc: 0.999339
first_decode_pcc: 0.991223
top_perf_samples_per_sec: 28.8907
pct_of_target: 76.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: bielik_11b_v3_gguf

## Test
tests/benchmark/test_llms.py::test_bielik_11b_v3_gguf

## Model
- HF name:    speakleash/Bielik-11B-v3.0-Instruct-GGUF
- Loader:     third_party.tt_forge_models.bielik_11b_v3_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIELIK_11B_V3_0_INSTRUCT_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  22.03
- TTFT (ms):          486.23
- Prefill PCC:        0.999339
- First decode PCC:   0.991223
- Wall clock:         0:14:44
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bielik_11b_v3_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 76.3% (22.03 / 28.89)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           110
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  880000000000000
- hifi2: 440000000000000
- hifi3: 293333333333333
- hifi4: 220000000000000

### Compute
- total_flops:             706354348160
- breakdown.matmul:        706354348160
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        419430400
- memory_bytes: 838860800
- memory_gb:    0.78125

### Params
- count:                  11168796867
- effective_count:        11037200579
- memory_bytes:           11990606600
- memory_gb:              11.167
- effective_memory_bytes: 11727414024
- effective_memory_gb:    10.922
- embedding_count:        131596288
- embedding_memory_bytes: 263192576

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 28.8907
- top_perf_time_ms:         34.6132
- dram_time_ms:             23.0755
- compute_time_ms_lofi:     0.8027
- compute_time_ms_hifi2:    1.6054
- compute_time_ms_hifi3:    2.4080
- compute_time_ms_hifi4:    3.2107

## Files changed
- tests/benchmark/test_llms.py (added test_bielik_11b_v3_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (added hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
