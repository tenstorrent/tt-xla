loader_path: third_party.tt_forge_models.bitnet_b1_58_large_tq2_0_gguf.causal_lm.pytorch.loader
variant_id: bitnet_b1_58_large_TQ2_0
arch: n150
status: DONE_PASS
test_function: test_bitnet_b1_58_large_tq2_0
samples_per_second: 22.39
ttft_ms: 547.9
prefill_pcc: 0.999132
first_decode_pcc: 0.999064
top_perf_samples_per_sec: 107.6419
pct_of_target: 20.8
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_bitnet_b1_58_large_tq2_0

## Test
tests/benchmark/test_llms.py::test_bitnet_b1_58_large_tq2_0

## Model
- HF name:    gianni-cor/bitnet_b1_58-large-TQ2_0
- Loader:     third_party.tt_forge_models.bitnet_b1_58_large_tq2_0_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BITNET_B1_58_LARGE_TQ2_0

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  22.39
- TTFT (ms):          547.9
- Prefill PCC:        0.999132
- First decode PCC:   0.999064
- Wall clock:         ~0:14:00
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bitnet_b1_58_large_tq2_0_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 20.8% (22.39 / 107.64)

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
- total_flops:             792751965792
- breakdown.matmul:        792751965792
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  777998003
- effective_count:        728842931
- memory_bytes:           872903560
- memory_gb:              0.8129547908902168
- effective_memory_bytes: 774593416
- effective_memory_gb:    0.7213963344693184
- embedding_count:        49155072
- embedding_memory_bytes: 98310144

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 107.6419
- top_perf_time_ms:         9.2901
- dram_time_ms:             3.6911
- compute_time_ms_lofi:     3.0967
- compute_time_ms_hifi2:    6.1934
- compute_time_ms_hifi3:    9.2901
- compute_time_ms_hifi4:    12.3867

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
