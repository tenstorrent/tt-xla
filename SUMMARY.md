loader_path: third_party.tt_forge_models.olmo.causal_lm.pytorch.loader
variant_id: 1b_0724
arch: p150
status: DONE_PASS
test_function: test_olmo_1b_0724
samples_per_second: 13.706
ttft_ms: 391.231
prefill_pcc: 0.993751
first_decode_pcc: 0.995834
top_perf_samples_per_sec: 229.027
pct_of_target: 5.9
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: olmo_1b_0724

## Test
tests/benchmark/test_llms.py::test_olmo_1b_0724

## Model
- HF name:    allenai/OLMo-1B-0724-hf
- Loader:     third_party.tt_forge_models.olmo.causal_lm.pytorch.loader
- Variant:    ModelVariant.OLMo_1B_0724

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: none
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level forced to 0 because levels 1 and 2 both trigger a
compiler error: `'ttnn.scaled_dot_product_attention' op Query and result must
have the same element type` (Error code: 13). This is a compiler bug.
experimental_weight_dtype disabled (bfp_bf8 also triggers the same compiler
error). Infrastructure fix applied: benchmarks/llm_benchmark.py now uses
hasattr guard before calling get_weight_dtype_config_path (same pattern as
tests/runner/testers/torch/dynamic_torch_model_tester.py).

## Measured (full model, defaults)
- Sample per second:  13.706
- TTFT (ms):          391.231
- Prefill PCC:        0.993751
- First decode PCC:   0.995834
- Wall clock:         0:01:12
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_olmo_1b_0724_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 5.9% (13.706 / 229.027)

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
- total_flops:             75849793664
- breakdown.matmul:        75849793664
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  1279787206
- effective_count:        1176764614
- memory_bytes:           2559574804
- memory_gb:              2.383789796382189
- effective_memory_bytes: 2353529620
- effective_memory_gb:    2.191895265132189
- embedding_count:        103022592
- embedding_memory_bytes: 206045184

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 229.0270
- top_perf_time_ms:         4.3663
- dram_time_ms:             2.9109
- compute_time_ms_lofi:     0.0862
- compute_time_ms_hifi2:    0.1724
- compute_time_ms_hifi3:    0.2586
- compute_time_ms_hifi4:    0.3448

## Files changed
- tests/benchmark/test_llms.py (added test_olmo_1b_0724)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added olmo_1b_0724 entry)

## tt-forge-models submodule
no change
