loader_path: third_party.tt_forge_models.aisingapore_llama_sea_lion_v3_5_8b_r_gguf.causal_lm.pytorch.loader
variant_id: Llama_SEA_LION_v3_5_8B_R_GGUF
arch: n150
status: DONE_PASS
test_function: test_aisingapore_llama_sea_lion_v3_5_8b_r_gguf
samples_per_second: 18.13
ttft_ms: 677.05
prefill_pcc: 0.999051
first_decode_pcc: 0.998545
top_perf_samples_per_sec: 23.9513
pct_of_target: 75.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_aisingapore_llama_sea_lion_v3_5_8b_r_gguf

## Test
tests/benchmark/test_llms.py::test_aisingapore_llama_sea_lion_v3_5_8b_r_gguf

## Model
- HF name:    aisingapore/Llama-SEA-LION-v3.5-8B-R-GGUF
- Loader:     third_party.tt_forge_models.aisingapore_llama_sea_lion_v3_5_8b_r_gguf.causal_lm.pytorch.loader
- Variant:    Llama_SEA_LION_v3_5_8B_R_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  18.13
- TTFT (ms):          677.05
- Prefill PCC:        0.999051
- First decode PCC:   0.998545
- Wall clock:         ~20 min (including GGUF de-quantization ~1:17)
- Hardware:           n150 (n300 wormhole_b0 single-chip)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_aisingapore_llama_sea_lion_v3_5_8b_r_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 75.7% (18.13 / 23.95)

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             480298139776
- breakdown.matmul:        480298139776
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  8030261443
- effective_count:        7504924867
- memory_bytes:           9024905992
- memory_gb:              8.4050986841321
- effective_memory_bytes: 7974232840
- effective_memory_gb:    7.426583059132099
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 23.9513
- top_perf_time_ms:         41.7514
- dram_time_ms:             27.8343
- compute_time_ms_lofi:     1.8762
- compute_time_ms_hifi2:    3.7523
- compute_time_ms_hifi3:    5.6285
- compute_time_ms_hifi4:    7.5047

## Files changed
- tests/benchmark/test_llms.py (added test_aisingapore_llama_sea_lion_v3_5_8b_r_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: use hasattr for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added aisingapore_llama_sea_lion_v3_5_8b_r_gguf entry with gguf>=0.10.0)

## tt-forge-models submodule
no change
