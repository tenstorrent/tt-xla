loader_path: third_party.tt_forge_models.olmo3_1025_7b_logic.causal_lm.pytorch.loader
variant_id: 3_1025_7b_logic
arch: p150
status: DONE_PASS
test_function: test_olmo3_1025_7b_logic
samples_per_second: 22.97
ttft_ms: 447.23
prefill_pcc: 0.999758
first_decode_pcc: 0.998652
top_perf_samples_per_sec: 41.5763
pct_of_target: 55.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: olmo3_1025_7b_logic

## Test
tests/benchmark/test_llms.py::test_olmo3_1025_7b_logic

## Model
- HF name:    allenai/Olmo-3-1025-7B
- Loader:     third_party.tt_forge_models.olmo3_1025_7b_logic.causal_lm.pytorch.loader
- Variant:    ModelVariant.Olmo_3_1025_7B_Logic

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  22.97
- TTFT (ms):          447.23
- Prefill PCC:        0.999758
- First decode PCC:   0.998652
- Wall clock:         0:07:43
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_olmo3_1025_7b_logic_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 55.2% (22.97 / 41.58)

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
- total_flops:             440751620224
- breakdown.matmul:        440751620224
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        34
- memory_bytes: 134

### KV cache
- count:        1073741824
- memory_bytes: 2147483648
- memory_gb:    2

### Params
- count:                  7298011333
- effective_count:        6887272645
- memory_bytes:           8139700496
- memory_gb:              7.5806868225336075
- effective_memory_bytes: 7318223120
- effective_memory_gb:    6.81562639772892
- embedding_count:        410738688
- embedding_memory_bytes: 821477376

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 41.5763
- top_perf_time_ms:         24.0521
- dram_time_ms:             16.0348
- compute_time_ms_lofi:     0.5009
- compute_time_ms_hifi2:    1.0017
- compute_time_ms_hifi3:    1.5026
- compute_time_ms_hifi4:    2.0034

## Files changed
- tests/benchmark/test_llms.py (added test_olmo3_1025_7b_logic)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added olmo3_1025_7b_logic entry)

## tt-forge-models submodule
no change — submodule at a58679a387
