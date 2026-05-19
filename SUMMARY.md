loader_path: third_party.tt_forge_models.black_goo_recipe_e.causal_lm.pytorch.loader
variant_id: recipe_e
arch: p150
status: DONE_PASS
test_function: test_black_goo_recipe_e
samples_per_second: 2.950172202732921
ttft_ms: 1960.50465
prefill_pcc: 0.994889
first_decode_pcc: 0.983972
top_perf_samples_per_sec: 82.6308
pct_of_target: 3.6
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_black_goo_recipe_e

## Test
tests/benchmark/test_llms.py::test_black_goo_recipe_e

## Model
- HF name:    KnutJaegersberg/black_goo_recipe_e
- Loader:     third_party.tt_forge_models.black_goo_recipe_e.causal_lm.pytorch.loader
- Variant:    ModelVariant.BLACK_GOO_RECIPE_E (recipe_e)

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 causes TT_THROW (circular buffers clash with L1 buffers);
optimization_level=1 causes low PCC (0.861 vs required 0.94, likely compiler numerical bug).
optimization_level=0 is the most aggressive setting that passes PCC.

## Measured (full model, defaults)
- Sample per second:  2.950172202732921
- TTFT (ms):          1960.50465
- Prefill PCC:        0.994889
- First decode PCC:   0.983972
- Wall clock:         0:05:05
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_black_goo_recipe_e_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 3.6% (2.95 / 82.63; low due to optimization_level=0)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             214093004900
- breakdown.matmul:        214093004900
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        681574400
- memory_bytes: 1363148800
- memory_gb:    1.26953125

### Params
- count:                  3426473784
- effective_count:        3324073784
- memory_bytes:           3736787932
- memory_gb:              3.4801549576222897
- effective_memory_bytes: 3531987932
- effective_memory_gb:    3.2894200943410397
- embedding_count:        102400000
- embedding_memory_bytes: 204800000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 82.6308
- top_perf_time_ms:         12.1020
- dram_time_ms:             8.0680
- compute_time_ms_lofi:     0.2059
- compute_time_ms_hifi2:    0.4117
- compute_time_ms_hifi3:    0.6176
- compute_time_ms_hifi4:    0.8234

## Files changed
- tests/benchmark/test_llms.py (added test_black_goo_recipe_e)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: add hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
