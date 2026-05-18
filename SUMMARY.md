loader_path: third_party.tt_forge_models.amber.causal_lm.pytorch.loader
variant_id: Amber
arch: p150
status: DONE_PASS
test_function: test_amber
samples_per_second: 26.32972201581451
ttft_ms: 343.761233
prefill_pcc: 0.996232
first_decode_pcc: 0.979625
top_perf_samples_per_sec: 763.6281
pct_of_target: 3.4
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_amber

## Test
tests/benchmark/test_llms.py::test_amber

## Model
- HF name:    LLM360/Amber
- Loader:     third_party.tt_forge_models.amber.causal_lm.pytorch.loader
- Variant:    ModelVariant.AMBER

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  26.32972201581451
- TTFT (ms):          343.761233
- Prefill PCC:        0.996232
- First decode PCC:   0.979625
- Wall clock:         0:07:56
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_amber_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 3.4%

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
- total_flops:             384131139840
- breakdown.matmul:        384131139840
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        594
- memory_bytes: 2376

### KV cache
- count:        33554432
- memory_bytes: 67108864
- memory_gb:    0.0625

### Params
- count:                  464531651
- effective_count:        333459651
- memory_bytes:           616456968
- memory_gb:              0.5741202905774117
- effective_memory_bytes: 354312968
- effective_memory_gb:    0.32997966557741165
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 763.6281
- top_perf_time_ms:         1.3095
- dram_time_ms:             0.7392
- compute_time_ms_lofi:     0.4365
- compute_time_ms_hifi2:    0.8730
- compute_time_ms_hifi3:    1.3095
- compute_time_ms_hifi4:    1.7461

## Files changed
- tests/benchmark/test_llms.py (added test_amber)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: use hasattr check before calling get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added llm360_amber entry)

## tt-forge-models submodule
no change
