loader_path: third_party.tt_forge_models.belgpt2.causal_lm.pytorch.loader
variant_id: belgpt2
arch: n150
status: DONE_PASS
test_function: test_belgpt2
samples_per_second: 73.35
ttft_ms: 205.82
prefill_pcc: 0.998861
first_decode_pcc: 0.995826
top_perf_samples_per_sec: 934.8777
pct_of_target: 7.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_belgpt2

## Test
tests/benchmark/test_llms.py::test_belgpt2

## Model
- HF name:    antoinelouis/belgpt2
- Loader:     third_party.tt_forge_models.belgpt2.causal_lm.pytorch.loader
- Variant:    ModelVariant.BELGPT2

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  73.35
- TTFT (ms):          205.82
- Prefill PCC:        0.998861
- First decode PCC:   0.995826
- Wall clock:         0:04:07
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_belgpt2_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 7.8% (73.35 / 934.88)

Note: The extract_perf_targets.py script uses a 1e12 FLOPs threshold to identify
decode vs prefill. For BelGPT2 (~163M params), even the prefill graph has only
~158B FLOPs (well below 1e12), so the script defaults to file 0 (prefill). The
true decode graph is file 1 (7.9B FLOPs), used here.

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
- total_flops:             7908704256
- breakdown.matmul:        2470232064
- breakdown.linear:        5438472192
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
- count:                  163037316
- effective_count:        123653508
- memory_bytes:           210429500
- memory_gb:              0.19597774371504784
- effective_memory_bytes: 131661884
- effective_memory_gb:    0.12261968478560448
- embedding_count:        39383808
- embedding_memory_bytes: 78767616

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 934.8777
- top_perf_time_ms:         1.0697
- dram_time_ms:             0.7131
- compute_time_ms_lofi:     0.0309
- compute_time_ms_hifi2:    0.0618
- compute_time_ms_hifi3:    0.0927
- compute_time_ms_hifi4:    0.1236

## Files changed
- tests/benchmark/test_llms.py (added test_belgpt2)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added belgpt2 entry)

## tt-forge-models submodule
no change
