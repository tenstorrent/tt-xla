loader_path: third_party.tt_forge_models.tiny_random_qwen3.causal_lm.pytorch.loader
variant_id: tiny_random_qwen3
arch: p150
status: DONE_PASS
test_function: test_tiny_random_qwen3
samples_per_second: 553.338
ttft_ms: 26.568
prefill_pcc: 0.999234
first_decode_pcc: 0.998275
top_perf_samples_per_sec: 27517.6522
pct_of_target: 2.0
roofline_bound: compute
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_tiny_random_qwen3

## Test
tests/benchmark/test_llms.py::test_tiny_random_qwen3

## Model
- HF name:    yujiepan/qwen3-tiny-random
- Loader:     third_party.tt_forge_models.tiny_random_qwen3.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINY_RANDOM_QWEN3

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 causes TT_THROW: No circular buffer with id 0 exists in Program (hangs indefinitely); optimization_level=1 is stable.

## Measured (full model, defaults)
- Sample per second:  553.338
- TTFT (ms):          26.568
- Prefill PCC:        0.999234
- First decode PCC:   0.998275
- Wall clock:         0:00:34
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_tiny_random_qwen3_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 2.0% (553.3 / 27517.7)

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
- total_flops:             10659824160
- breakdown.matmul:        10659824160
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        524288
- memory_bytes: 1048576
- memory_gb:    0.0009765625

### Params
- count:                  19522131
- effective_count:        9798227
- memory_bytes:           29859272
- memory_gb:              0.02780861407518387
- effective_memory_bytes: 10411464
- effective_memory_gb:    0.009696431457996368
- embedding_count:        9723904
- embedding_memory_bytes: 19447808

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 27517.6522
- top_perf_time_ms:         0.0363
- dram_time_ms:             0.0208
- compute_time_ms_lofi:     0.0121
- compute_time_ms_hifi2:    0.0242
- compute_time_ms_hifi3:    0.0363
- compute_time_ms_hifi4:    0.0485

## Files changed
- tests/benchmark/test_llms.py (added test_tiny_random_qwen3)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed: guard get_weight_dtype_config_path with hasattr check, mirrors dynamic_torch_model_tester.py)

## tt-forge-models submodule
no change — submodule at fab024ed33
