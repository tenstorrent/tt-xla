loader_path: third_party.tt_forge_models.granite_code.causal_lm.pytorch.loader
variant_id: Granite_8B_Code_Base_4K
arch: p150
status: DONE_PASS
test_function: test_granite_code_8b
samples_per_second: 31.057
ttft_ms: 344.19
prefill_pcc: 0.999241
first_decode_pcc: 0.998618
top_perf_samples_per_sec: 39.6068
pct_of_target: 78.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: granite_code_8b

## Test
tests/benchmark/test_llms.py::test_granite_code_8b

## Model
- HF name:    ibm-granite/granite-8b-code-base-4k
- Loader:     third_party.tt_forge_models.granite_code.causal_lm.pytorch.loader
- Variant:    Granite_8B_Code_Base_4K

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  31.057
- TTFT (ms):          344.19
- Prefill PCC:        0.999241
- First decode PCC:   0.998618
- Wall clock:         0:10:17
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_granite_code_8b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 78.4% (31.057 / 39.6068)

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
- total_flops:             515445620864
- breakdown.matmul:        12884902016
- breakdown.linear:        502560718848
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  8256237763
- effective_count:        8054911171
- memory_bytes:           8962728712
- memory_gb:              8.347191579639912
- effective_memory_bytes: 8560075528
- effective_memory_gb:    7.972191579639912
- embedding_count:        201326592
- embedding_memory_bytes: 402653184

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 39.6068
- top_perf_time_ms:         25.2482
- dram_time_ms:             16.8321
- compute_time_ms_lofi:     0.5857
- compute_time_ms_hifi2:    1.1715
- compute_time_ms_hifi3:    1.7572
- compute_time_ms_hifi4:    2.3429

## Files changed
- tests/benchmark/test_llms.py (added test_granite_code_8b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: use hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added granite_code_8b entry)

## tt-forge-models submodule
no change
