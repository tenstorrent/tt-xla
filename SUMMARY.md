loader_path: third_party.tt_forge_models.cerebras_gpt_1_3b.causal_lm.pytorch.loader
variant_id: base
arch: n150
status: DONE_PASS
test_function: test_cerebras_gpt_1_3b
samples_per_second: 21.725
ttft_ms: 624.755919
prefill_pcc: 0.996849
first_decode_pcc: 0.999329
top_perf_samples_per_sec: 108.6126
pct_of_target: 20.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_cerebras_gpt_1_3b

## Test
tests/benchmark/test_llms.py::test_cerebras_gpt_1_3b

## Model
- HF name:    cerebras/Cerebras-GPT-1.3B
- Loader:     third_party.tt_forge_models.cerebras_gpt_1_3b.causal_lm.pytorch.loader
- Variant:    base

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  21.725
- TTFT (ms):          624.755919
- Prefill PCC:        0.996849
- First decode PCC:   0.999329
- Wall clock:         0:10:48
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_cerebras_gpt_1_3b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 20.0%

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
- total_flops:             83910852608
- breakdown.matmul:        6587285504
- breakdown.linear:        77323567104
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        402653184
- memory_bytes: 805306368
- memory_gb:    0.75

### Params
- count:                  1418649732
- effective_count:        1311529092
- memory_bytes:           1609228940
- memory_gb:              1.4987112395465374
- effective_memory_bytes: 1394987660
- effective_memory_gb:    1.2991834990680218
- embedding_count:        107120640
- embedding_memory_bytes: 214241280

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 108.6126
- top_perf_time_ms:         9.2070
- dram_time_ms:             6.1380
- compute_time_ms_lofi:     0.3278
- compute_time_ms_hifi2:    0.6556
- compute_time_ms_hifi3:    0.9833
- compute_time_ms_hifi4:    1.3111

## Files changed
- tests/benchmark/test_llms.py (added test_cerebras_gpt_1_3b)
- tests/benchmark/benchmarks/llm_benchmark.py (graceful handling of missing get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added cerebras_gpt_1_3b entry)

## tt-forge-models submodule
no change
