loader_path: third_party.tt_forge_models.hermes7b_ita.causal_lm.pytorch.loader
variant_id: Hermes7b_ITA
arch: p150
status: DONE_PASS
test_function: test_hermes7b_ita
samples_per_second: 25.056166529296235
ttft_ms: 341.527285
prefill_pcc: 0.998749
first_decode_pcc: 0.964149
top_perf_samples_per_sec: 43.0916
pct_of_target: 58.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: hermes7b_ita

## Test
tests/benchmark/test_llms.py::test_hermes7b_ita

## Model
- HF name:    raicrits/Hermes7b_ITA
- Loader:     third_party.tt_forge_models.hermes7b_ita.causal_lm.pytorch.loader
- Variant:    ModelVariant.HERMES7B_ITA

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  25.056166529296235
- TTFT (ms):          341.527285
- Prefill PCC:        0.998749
- First decode PCC:   0.964149
- Wall clock:         0:07:37
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_hermes7b_ita_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 58.1% (25.06 / 43.09)

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
- total_flops:             422852952192
- breakdown.matmul:        422852952192
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
- count:                  6738415811
- effective_count:        6607343811
- memory_bytes:           7282696968
- memory_gb:              6.782540090382099
- effective_memory_bytes: 7020552968
- effective_memory_gb:    6.538399465382099
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 43.0916
- top_perf_time_ms:         23.2064
- dram_time_ms:             15.4709
- compute_time_ms_lofi:     0.4805
- compute_time_ms_hifi2:    0.9610
- compute_time_ms_hifi3:    1.4415
- compute_time_ms_hifi4:    1.9221

## Files changed
- tests/benchmark/test_llms.py (added test_hermes7b_ita)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: use hasattr before calling get_weight_dtype_config_path)

## tt-forge-models submodule
no change
