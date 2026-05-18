loader_path: third_party.tt_forge_models.corianas_256_5epoch.causal_lm.pytorch.loader
variant_id: corianas_256_5epoch
arch: p150
status: DONE_PASS
test_function: test_corianas_256_5epoch
samples_per_second: 114.997
ttft_ms: 110.770
prefill_pcc: 0.997863
first_decode_pcc: 0.998778
top_perf_samples_per_sec: 874.4098
pct_of_target: 13.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_corianas_256_5epoch

## Test
tests/benchmark/test_llms.py::test_corianas_256_5epoch

## Model
- HF name:    Corianas/256_5epoch
- Loader:     third_party.tt_forge_models.corianas_256_5epoch.causal_lm.pytorch.loader
- Variant:    CORIANAS_256_5EPOCH

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  114.997
- TTFT (ms):          110.770
- Prefill PCC:        0.997863
- First decode PCC:   0.998778
- Wall clock:         0:07:37
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_corianas_256_5epoch_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 13.2%

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
- total_flops:             275935461376
- breakdown.matmul:        59491422208
- breakdown.linear:        216444039168
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        124780544
- memory_bytes: 249561088
- memory_gb:    0.232421875

### Params
- count:                  310656772
- effective_count:        253748932
- memory_bytes:           383886160
- memory_gb:              0.3575218468904495
- effective_memory_bytes: 270070480
- effective_memory_gb:    0.2515227347612381
- embedding_count:        56907840
- embedding_memory_bytes: 113815680

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 874.4098
- top_perf_time_ms:         1.1436
- dram_time_ms:             0.7624
- compute_time_ms_lofi:     0.3136
- compute_time_ms_hifi2:    0.6271
- compute_time_ms_hifi3:    0.9407
- compute_time_ms_hifi4:    1.2543

## Files changed
- tests/benchmark/test_llms.py (new test_corianas_256_5epoch function)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: use hasattr guard for get_weight_dtype_config_path to handle loaders that don't implement this method)
- .github/workflows/perf-bench-matrix.json (added corianas_256_5epoch entry)

## tt-forge-models submodule
no change
