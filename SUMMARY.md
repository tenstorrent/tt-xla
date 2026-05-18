loader_path: third_party.tt_forge_models.habana_gpt2.causal_lm.pytorch.loader
variant_id: Habana_gpt2
arch: p150
status: DONE_PASS
test_function: test_habana_gpt2
samples_per_second: 166.16095127978434
ttft_ms: 80.588697
prefill_pcc: 0.997231
first_decode_pcc: 0.976714
top_perf_samples_per_sec: 1662.0049
pct_of_target: 10.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_habana_gpt2

## Test
tests/benchmark/test_llms.py::test_habana_gpt2

## Model
- HF name:    gpt2
- Loader:     third_party.tt_forge_models.habana_gpt2.causal_lm.pytorch.loader
- Variant:    ModelVariant.HABANA_GPT2

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  166.16095127978434
- TTFT (ms):          80.588697
- Prefill PCC:        0.997231
- First decode PCC:   0.976714
- Wall clock:         0:02:05
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_habana_gpt2_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 10.0%

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
- tests/benchmark/test_llms.py (added test_habana_gpt2)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
