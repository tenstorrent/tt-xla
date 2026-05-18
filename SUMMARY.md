loader_path: third_party.tt_forge_models.cerebras_gpt_1_3b.causal_lm.pytorch.loader
variant_id: base
arch: p150
status: DONE_PASS
test_function: test_cerebras_gpt_1_3b
samples_per_second: 47.31136078528133
ttft_ms: 268.693419
prefill_pcc: 0.997761
first_decode_pcc: 0.999572
top_perf_samples_per_sec: 193.0891
pct_of_target: 24.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_cerebras_gpt_1_3b

## Test
tests/benchmark/test_llms.py::test_cerebras_gpt_1_3b

## Model
- HF name:    cerebras/Cerebras-GPT-1.3B
- Loader:     third_party.tt_forge_models.cerebras_gpt_1_3b.causal_lm.pytorch.loader
- Variant:    base

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8 (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  47.31136078528133
- TTFT (ms):          268.693419
- Prefill PCC:        0.997761
- First decode PCC:   0.999572
- Wall clock:         0:08:04
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_cerebras_gpt_1_3b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 24.5% (47.31 / 193.09)

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
- total_flops:             83910852608
- breakdown.matmul:        6587285504
- breakdown.linear:        77323567104
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        402653184
- memory_bytes: 805306368
- memory_gb:    0.75

### Params
- count:                  1418649732
- effective_count:        1311529092
- memory_bytes:           1609228940
- memory_gb:              1.4987112395465374
- effective_memory_bytes: 1394987660
- effective_memory_gb:    1.2991834990680218
- embedding_count:        107120640
- embedding_memory_bytes: 214241280

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 193.0891
- top_perf_time_ms:         5.1790
- dram_time_ms:             3.4526
- compute_time_ms_lofi:     0.0954
- compute_time_ms_hifi2:    0.1907
- compute_time_ms_hifi3:    0.2861
- compute_time_ms_hifi4:    0.3814

## Files changed
- tests/benchmark/test_llms.py (added test_cerebras_gpt_1_3b)
- .github/workflows/perf-bench-matrix.json (already contained cerebras_gpt_1_3b entry)

## tt-forge-models submodule
no change
