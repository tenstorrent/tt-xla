loader_path: third_party.tt_forge_models.gerpt2_large.causal_lm.pytorch.loader
variant_id: Gerpt2_Large
arch: p150
status: DONE_PASS
test_function: test_gerpt2_large
samples_per_second: 43.14557038478398
ttft_ms: 325.576053
prefill_pcc: 0.999090
first_decode_pcc: 0.999752
top_perf_samples_per_sec: 287.7655
pct_of_target: 15.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: gerpt2_large

## Test
tests/benchmark/test_llms.py::test_gerpt2_large

## Model
- HF name:    benjamin/gerpt2-large
- Loader:     third_party.tt_forge_models.gerpt2_large.causal_lm.pytorch.loader
- Variant:    ModelVariant.GERPT2_LARGE

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  43.14557038478398
- TTFT (ms):          325.576053
- Prefill PCC:        0.999090
- First decode PCC:   0.999752
- Wall clock:         0:08:44
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gerpt2_large_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 15.0% (43.15 / 287.77)

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
- total_flops:             49428807680
- breakdown.matmul:        4117053440
- breakdown.linear:        45311754240
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        377487360
- memory_bytes: 754974720
- memory_gb:    0.703125

### Params
- count:                  838359172
- effective_count:        772719492
- memory_bytes:           953687644
- memory_gb:              0.8881908319890499
- effective_memory_bytes: 822408284
- effective_memory_gb:    0.7659274004399776
- embedding_count:        65639680
- embedding_memory_bytes: 131279360

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 287.7655
- top_perf_time_ms:         3.4751
- dram_time_ms:             2.3167
- compute_time_ms_lofi:     0.0562
- compute_time_ms_hifi2:    0.1123
- compute_time_ms_hifi3:    0.1685
- compute_time_ms_hifi4:    0.2247

## Files changed
- tests/benchmark/test_llms.py (added test_gerpt2_large)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: getattr fallback for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added gerpt2_large entry)

## tt-forge-models submodule
no change
