loader_path: third_party.tt_forge_models.ko_yoshida_exp1_averaged.causal_lm.pytorch.loader
variant_id: exp1_averaged
arch: p150
status: DONE_PASS
test_function: test_ko_yoshida_exp1_averaged
samples_per_second: 186.06
ttft_ms: 101.05
prefill_pcc: 0.999520
first_decode_pcc: 0.999324
top_perf_samples_per_sec: 586.4262
pct_of_target: 31.7
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_ko_yoshida_exp1_averaged

## Test
tests/benchmark/test_llms.py::test_ko_yoshida_exp1_averaged

## Model
- HF name:    ko-yoshida/exp1_averaged
- Loader:     third_party.tt_forge_models.ko_yoshida_exp1_averaged.causal_lm.pytorch.loader
- Variant:    ModelVariant.EXP1_AVERAGED

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  186.06
- TTFT (ms):          101.05
- Prefill PCC:        0.999520
- First decode PCC:   0.999324
- Wall clock:         0:03:12
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_ko_yoshida_exp1_averaged_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 31.7% (186.06 / 586.43)

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
- total_flops:             500205061440
- breakdown.matmul:        466892096832
- breakdown.linear:        33312964608
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        693
- memory_bytes: 2772

### KV cache
- count:        25165824
- memory_bytes: 50331648
- memory_gb:    0.046875

### Params
- count:                  386570275
- effective_count:        372234275
- memory_bytes:           424238472
- memory_gb:              0.3951028659939766
- effective_memory_bytes: 395566472
- effective_memory_gb:    0.3683999851346016
- embedding_count:        14336000
- embedding_memory_bytes: 28672000

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 586.4262
- top_perf_time_ms:         1.7052
- dram_time_ms:             0.8004
- compute_time_ms_lofi:     0.5684
- compute_time_ms_hifi2:    1.1368
- compute_time_ms_hifi3:    1.7052
- compute_time_ms_hifi4:    2.2737

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
