loader_path: third_party.tt_forge_models.internlm2_1_8b.causal_lm.pytorch.loader
variant_id: 1_8B
arch: p150
status: DONE_PASS
test_function: test_internlm2_1_8b
samples_per_second: 60.79
ttft_ms: 527.7
prefill_pcc: 1.000000
first_decode_pcc: 1.000000
top_perf_samples_per_sec: 174.12
pct_of_target: 34.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: internlm2_1_8b

## Test
tests/benchmark/test_llms.py::test_internlm2_1_8b

## Model
- HF name:    internlm/internlm2-1_8b
- Loader:     third_party.tt_forge_models.internlm2_1_8b.causal_lm.pytorch.loader
- Variant:    ModelVariant.INTERNLM2_1_8B (1_8B)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  60.79
- TTFT (ms):          527.7
- Prefill PCC:        1.000000
- First decode PCC:   1.000000
- Wall clock:         0:07:44
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_internlm2_1_8b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 34.9% (60.79 / 174.12)

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
- total_flops:             108766694400
- breakdown.matmul:        108766694400
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        201326592
- memory_bytes: 402653184
- memory_gb:    0.375

### Params
- count:                  1889111682
- effective_count:        1699581570
- memory_bytes:           2184964614
- memory_gb:              2.034906869754195
- effective_memory_bytes: 1805904390
- effective_memory_gb:    1.6818795260041952
- embedding_count:        189530112
- embedding_memory_bytes: 379060224

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 174.1220
- top_perf_time_ms:         5.7431
- dram_time_ms:             3.8287
- compute_time_ms_lofi:     0.1236
- compute_time_ms_hifi2:    0.2472
- compute_time_ms_hifi3:    0.3708
- compute_time_ms_hifi4:    0.4944

## Files changed
- tests/benchmark/test_llms.py (added test_internlm2_1_8b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added internlm2_1_8b entry)

## tt-forge-models submodule
no change
