loader_path: third_party.tt_forge_models.mental_alpaca.causal_lm.pytorch.loader
variant_id: mental_alpaca
arch: p150
status: DONE_PASS
test_function: test_mental_alpaca
samples_per_second: 25.86
ttft_ms: 359.25
prefill_pcc: 0.999859
first_decode_pcc: 0.999867
top_perf_samples_per_sec: 43.0915
pct_of_target: 60.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_mental_alpaca

## Test
tests/benchmark/test_llms.py::test_mental_alpaca

## Model
- HF name:    NEU-HAI/mental-alpaca
- Loader:     third_party.tt_forge_models.mental_alpaca.causal_lm.pytorch.loader
- Variant:    ModelVariant.MENTAL_ALPACA

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  25.86
- TTFT (ms):          359.25
- Prefill PCC:        0.999859
- First decode PCC:   0.999867
- Wall clock:         0:08:15
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mental_alpaca_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 60.0% (25.86 / 43.09)

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
- total_flops:             422853214336
- breakdown.matmul:        422853214336
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1073741824
- memory_bytes: 2147483648
- memory_gb:    2

### Params
- count:                  6738424003
- effective_count:        6607347907
- memory_bytes:           7282709512
- memory_gb:              6.782551772892475
- effective_memory_bytes: 7020557320
- effective_memory_gb:    6.538403518497944
- embedding_count:        131076096
- embedding_memory_bytes: 262152192

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 43.0915
- top_perf_time_ms:         23.2064
- dram_time_ms:             15.4709
- compute_time_ms_lofi:     0.4805
- compute_time_ms_hifi2:    0.9610
- compute_time_ms_hifi3:    1.4415
- compute_time_ms_hifi4:    1.9221

## Files changed
- tests/benchmark/test_llms.py (added test_mental_alpaca)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added mental_alpaca entry)

## tt-forge-models submodule
no change
