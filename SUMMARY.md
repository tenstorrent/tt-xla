loader_path: third_party.tt_forge_models.gred_cozmo.causal_lm.pytorch.loader
variant_id: gred_cozmo
arch: p150
status: DONE_PASS
test_function: test_gred_cozmo
samples_per_second: 13.83363364269759
ttft_ms: 381.075502
prefill_pcc: 0.998823
first_decode_pcc: 0.978768
top_perf_samples_per_sec: 596.2671
pct_of_target: 2.3
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_gred_cozmo

## Test
tests/benchmark/test_llms.py::test_gred_cozmo

## Model
- HF name:    bsu-slim/gred-cozmo
- Loader:     third_party.tt_forge_models.gred_cozmo.causal_lm.pytorch.loader
- Variant:    GRED_COZMO

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level >= 1 causes first decode PCC to drop below threshold
(measured 0.929 @ L1, 0.919 @ L2). Using optimization_level=0 (all DRAM)
as the only passing configuration.

## Measured (full model, defaults)
- Sample per second:  13.83363364269759
- TTFT (ms):          381.075502
- Prefill PCC:        0.998823
- First decode PCC:   0.978768
- Wall clock:         0:01:41
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gred_cozmo_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 2.3%

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
- total_flops:             23030857728
- breakdown.matmul:        3696427008
- breakdown.linear:        19334430720
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
- count:                  406290567
- effective_count:        353776775
- memory_bytes:           481659608
- memory_gb:              0.44858046621084213
- effective_memory_bytes: 376632024
- effective_memory_gb:    0.3507659062743187
- embedding_count:        52513792
- embedding_memory_bytes: 105027584

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 596.2671
- top_perf_time_ms:         1.6771
- dram_time_ms:             1.1181
- compute_time_ms_lofi:     0.0262
- compute_time_ms_hifi2:    0.0523
- compute_time_ms_hifi3:    0.0785
- compute_time_ms_hifi4:    0.1047

## Files changed
- tests/benchmark/test_llms.py (added test_gred_cozmo)
- tests/benchmark/benchmarks/llm_benchmark.py (added hasattr check for get_weight_dtype_config_path — general fix for model loaders that don't implement this method)

## tt-forge-models submodule
no change
