loader_path: third_party.tt_forge_models.legion_coder.causal_lm.pytorch.loader
variant_id: Default
arch: p150
status: DONE_PASS
test_function: test_legion_coder
samples_per_second: 171.37
ttft_ms: 100.31
prefill_pcc: 0.998433
first_decode_pcc: 0.998554
top_perf_samples_per_sec: 981.7197
pct_of_target: 17.5
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_legion_coder

## Test
tests/benchmark/test_llms.py::test_legion_coder

## Model
- HF name:    dineth554/legion-coder-8m
- Loader:     third_party.tt_forge_models.legion_coder.causal_lm.pytorch.loader
- Variant:    ModelVariant.LEGION_CODER_8M ("Default")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  171.37
- TTFT (ms):          100.31
- Prefill PCC:        0.998433
- First decode PCC:   0.998554
- Wall clock:         0:02:33
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_legion_coder_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 17.5% (171.37 / 981.72)

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
- total_flops:             298795401216
- breakdown.matmul:        37748736000
- breakdown.linear:        261046665216
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        1584
- memory_bytes: 6336

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  110418564
- effective_count:        97344132
- memory_bytes:           129857036
- memory_gb:              0.12093878909945488
- effective_memory_bytes: 103708172
- effective_memory_gb:    0.09658576175570488
- embedding_count:        13074432
- embedding_memory_bytes: 26148864

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 981.7197
- top_perf_time_ms:         1.0186
- dram_time_ms:             0.3481
- compute_time_ms_lofi:     0.3395
- compute_time_ms_hifi2:    0.6791
- compute_time_ms_hifi3:    1.0186
- compute_time_ms_hifi4:    1.3582

## Files changed
- tests/benchmark/test_llms.py (added test_legion_coder)
- tests/benchmark/benchmarks/llm_benchmark.py (defensive getattr for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added legion_coder entry)

## tt-forge-models submodule
no change — submodule stays at 91a1a825cb
