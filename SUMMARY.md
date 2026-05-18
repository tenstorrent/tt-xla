loader_path: third_party.tt_forge_models.alias_gpt2.causal_lm.pytorch.loader
variant_id: small_x21
arch: p150
status: DONE_PASS
test_function: test_alias_gpt2_small_x21
samples_per_second: 164.449
ttft_ms: 83.13
prefill_pcc: 0.999146
first_decode_pcc: 0.999796
top_perf_samples_per_sec: 1662.0049
pct_of_target: 9.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: alias_gpt2_small_x21

## Test
tests/benchmark/test_llms.py::test_alias_gpt2_small_x21

## Model
- HF name:    stanford-crfm/alias-gpt2-small-x21
- Loader:     third_party.tt_forge_models.alias_gpt2.causal_lm.pytorch.loader
- Variant:    ModelVariant.ALIAS_GPT2_SMALL_X21

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  164.449
- TTFT (ms):          83.13
- Prefill PCC:        0.999146
- First decode PCC:   0.999796
- Wall clock:         0:02:09
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_alias_gpt2_small_x21_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 9.9% (164.45 / 1662.0)

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
- top_perf_samples_per_sec: 1662.0049
- top_perf_time_ms:         0.6017
- dram_time_ms:             0.4011
- compute_time_ms_lofi:     0.0090
- compute_time_ms_hifi2:    0.0180
- compute_time_ms_hifi3:    0.0270
- compute_time_ms_hifi4:    0.0359

## Files changed
- tests/benchmark/test_llms.py (added test_alias_gpt2_small_x21)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
