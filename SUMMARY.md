loader_path: third_party.tt_forge_models.gpt2_expansion.causal_lm.pytorch.loader
variant_id: gpt2_expansion
arch: p150
status: DONE_PASS
test_function: test_gpt2_expansion
samples_per_second: 163.011
ttft_ms: 85.547
prefill_pcc: 0.998313
first_decode_pcc: 0.997927
top_perf_samples_per_sec: 1662.0049
pct_of_target: 9.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_gpt2_expansion

## Test
tests/benchmark/test_llms.py::test_gpt2_expansion

## Model
- HF name:    polypo/gpt2-expansion
- Loader:     third_party.tt_forge_models.gpt2_expansion.causal_lm.pytorch.loader
- Variant:    ModelVariant.GPT2_EXPANSION

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  163.011
- TTFT (ms):          85.547
- Prefill PCC:        0.998313
- First decode PCC:   0.997927
- Wall clock:         0:02:08
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gpt2_expansion_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 9.8% (163.011 / 1662.0049)

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
- total_flops:             134447972352
- breakdown.matmul:        41993945088
- breakdown.linear:        92454027264
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

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
- top_perf_samples_per_sec: 1662.0049
- top_perf_time_ms:         0.6017
- dram_time_ms:             0.4011
- compute_time_ms_lofi:     0.1528
- compute_time_ms_hifi2:    0.3056
- compute_time_ms_hifi3:    0.4583
- compute_time_ms_hifi4:    0.6111

## Files changed
- tests/benchmark/test_llms.py (added test_gpt2_expansion)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: use hasattr before calling get_weight_dtype_config_path)

## tt-forge-models submodule
no change
