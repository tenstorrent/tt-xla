loader_path: third_party.tt_forge_models.olmo2.causal_lm.pytorch.loader
variant_id: 1124_7b_sft
arch: p150
status: DONE_PASS
test_function: test_olmo2_1124_7b_sft
samples_per_second: 3.4916
ttft_ms: 1561.95
prefill_pcc: 0.998270
first_decode_pcc: 0.999317
top_perf_samples_per_sec: 41.5748
pct_of_target: 8.4
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_olmo2_1124_7b_sft

## Test
tests/benchmark/test_llms.py::test_olmo2_1124_7b_sft

## Model
- HF name:    allenai/OLMo-2-1124-7B-SFT
- Loader:     third_party.tt_forge_models.olmo2.causal_lm.pytorch.loader
- Variant:    OLMo_2_1124_7B_SFT ("1124_7b_sft")

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=1 and optimization_level=2 both fail with MLIR compiler error:
  'ttnn.scaled_dot_product_attention' op Query and result must have the same element type (Error code: 13)
Only optimization_level=0 produces a working compilation.

Also note: benchmarking infrastructure fix applied — `llm_benchmark.py` now uses
`getattr(model_loader, "get_weight_dtype_config_path", lambda: None)()` so loaders
that do not implement `get_weight_dtype_config_path` (like OLMo-2) no longer crash.

## Measured (full model, defaults)
- Sample per second:  3.4916
- TTFT (ms):          1561.95
- Prefill PCC:        0.998270
- First decode PCC:   0.999317
- Wall clock:         0:04:43
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_olmo2_1124_7b_sft_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 8.4% (3.49 / 41.57 samples/sec)

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
- total_flops:             442918502528
- breakdown.matmul:        442918502528
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
- count:                  7298617542
- effective_count:        6887575750
- memory_bytes:           8140628756
- memory_gb:              7.58
- effective_memory_bytes: 7318545172
- effective_memory_gb:    6.82
- embedding_count:        411041792
- embedding_memory_bytes: 822083584

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 41.5748
- top_perf_time_ms:         24.0531
- dram_time_ms:             16.0354
- compute_time_ms_lofi:     0.5033
- compute_time_ms_hifi2:    1.0066
- compute_time_ms_hifi3:    1.5099
- compute_time_ms_hifi4:    2.0133

## Files changed
- tests/benchmark/test_llms.py (added test_olmo2_1124_7b_sft)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path call with getattr)

## tt-forge-models submodule
no change
