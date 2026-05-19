loader_path: third_party.tt_forge_models.rogpt2_base.causal_lm.pytorch.loader
variant_id: Default
arch: p150
status: DONE_PASS
test_function: test_rogpt2_base
samples_per_second: 166.68
ttft_ms: 83.64
prefill_pcc: 0.999330
first_decode_pcc: 0.999472
top_perf_samples_per_sec: 1545.4140
pct_of_target: 10.8
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: rogpt2_base

## Test
tests/benchmark/test_llms.py::test_rogpt2_base

## Model
- HF name:    readerbench/RoGPT2-base
- Loader:     third_party.tt_forge_models.rogpt2_base.causal_lm.pytorch.loader
- Variant:    ModelVariant.BASE ("Default")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  166.68
- TTFT (ms):          83.64
- Prefill PCC:        0.999330
- First decode PCC:   0.999472
- Wall clock:         0:02:07
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_rogpt2_base_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 10.8% (166.68 / 1545.41)

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
- total_flops:             189808902144
- breakdown.matmul:        59285569536
- breakdown.linear:        130523332608
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        792
- memory_bytes: 3168

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
- bound:                    compute
- top_perf_samples_per_sec: 1545.4140
- top_perf_time_ms:         0.6471
- dram_time_ms:             0.4011
- compute_time_ms_lofi:     0.2157
- compute_time_ms_hifi2:    0.4314
- compute_time_ms_hifi3:    0.6471
- compute_time_ms_hifi4:    0.8628

## Files changed
- tests/benchmark/test_llms.py (added test_rogpt2_base)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: lazy tokenizer load, hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added rogpt2_base entry)

## tt-forge-models submodule
no change
