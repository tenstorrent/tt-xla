loader_path: third_party.tt_forge_models.olmo_3_7b_instruct_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: n150
status: DONE_PASS
test_function: test_olmo_3_7b_instruct_gguf
samples_per_second: 3.4937
ttft_ms: 1568.021
prefill_pcc: 0.999141
first_decode_pcc: 0.999052
top_perf_samples_per_sec: 41.5763
pct_of_target: 8.4
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_olmo_3_7b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_olmo_3_7b_instruct_gguf

## Model
- HF name:    unsloth/Olmo-3-7B-Instruct-GGUF
- Loader:     third_party.tt_forge_models.olmo_3_7b_instruct_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.OLMO_3_7B_INSTRUCT_Q4_K_M

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level is hard-coded to 0 because optimization_level >= 1 triggers a
compiler error in SDPA lowering: 'ttnn.scaled_dot_product_attention' op Query and result
must have the same element type. This is a compiler-level bug that cannot be fixed from
the test side.

## Measured (full model, defaults)
- Sample per second:  3.4937
- TTFT (ms):          1568.021
- Prefill PCC:        0.999141
- First decode PCC:   0.999052
- Wall clock:         0:05:14
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_olmo_3_7b_instruct_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 8.4% (3.4937 / 41.5763)

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
- total_flops:             442899103872
- breakdown.matmul:        442899103872
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
- count:                  7298011334
- effective_count:        6887272646
- memory_bytes:           8139700500
- memory_gb:              7.58
- effective_memory_bytes: 7318223124
- effective_memory_gb:    6.82
- embedding_count:        410738688
- embedding_memory_bytes: 821477376

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 41.5763
- top_perf_time_ms:         24.0521
- dram_time_ms:             16.0348
- compute_time_ms_lofi:     0.5033
- compute_time_ms_hifi2:    1.0066
- compute_time_ms_hifi3:    1.5099
- compute_time_ms_hifi4:    2.0132

## Files changed
- tests/benchmark/test_llms.py (added test_olmo_3_7b_instruct_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path for loaders without the method)
- .github/workflows/perf-bench-matrix.json (added olmo_3_7b_instruct_gguf entry)

## tt-forge-models submodule
no change
