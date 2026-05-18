loader_path: third_party.tt_forge_models.athena_1_3b_i1_gguf.causal_lm.pytorch.loader
variant_id: 3B_I1_GGUF
arch: p150
status: DONE_PASS
test_function: test_athena_1_3b_i1_gguf
samples_per_second: 41.613
ttft_ms: 226.481
prefill_pcc: 0.997069
first_decode_pcc: 0.999147
top_perf_samples_per_sec: 104.6961
pct_of_target: 39.7
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_athena_1_3b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_athena_1_3b_i1_gguf

## Model
- HF name:    mradermacher/Athena-1-3B-i1-GGUF
- Loader:     third_party.tt_forge_models.athena_1_3b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ATHENA_1_3B_I1_GGUF (value: "3B_I1_GGUF")

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 triggers a paged_update_cache sharding validation
error on p150 blackhole ("Expect input_tensor to be sharded"); optimization_level=1
is stable and passes PCC.

Also fixed: `benchmark_llm_torch_xla` harness now guards `get_weight_dtype_config_path`
with `hasattr` (mirrors the existing runner fix in dynamic_torch_model_tester.py)
so GGUF loaders that lack this method work correctly.

## Measured (full model, defaults)
- Sample per second:  41.613
- TTFT (ms):          226.481
- Prefill PCC:        0.997069
- First decode PCC:   0.999147
- Wall clock:         0:03:32
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_athena_1_3b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 39.7% (41.6 / 104.7 samples/sec)

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
- total_flops:             197487558784
- breakdown.matmul:        185405014144
- breakdown.linear:        12082544640
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  3397103811
- effective_count:        3085938883
- memory_bytes:           3901367048
- memory_gb:              3.633431203663349
- effective_memory_bytes: 3279037192
- effective_memory_gb:    3.053841359913349
- embedding_count:        311164928
- embedding_memory_bytes: 622329856

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 104.6961
- top_perf_time_ms:         9.5515
- dram_time_ms:             6.3676
- compute_time_ms_lofi:     0.2244
- compute_time_ms_hifi2:    0.4488
- compute_time_ms_hifi3:    0.6733
- compute_time_ms_hifi4:    0.8977

## Files changed
- tests/benchmark/test_llms.py (added test_athena_1_3b_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added athena_1_3b_i1_gguf entry)

## tt-forge-models submodule
no change
