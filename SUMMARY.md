loader_path: third_party.tt_forge_models.snake7gun_tiny_random_qwen3.causal_lm.pytorch.loader
variant_id: tiny_random_qwen3
arch: p150
status: DONE_PASS
test_function: test_snake7gun_tiny_random_qwen3
samples_per_second: 555.1091636805074
ttft_ms: 26.453981
prefill_pcc: 0.999234
first_decode_pcc: 0.998275
top_perf_samples_per_sec: 32064.8492
pct_of_target: 1.7
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_snake7gun_tiny_random_qwen3

## Test
tests/benchmark/test_llms.py::test_snake7gun_tiny_random_qwen3

## Model
- HF name:    snake7gun/tiny-random-qwen3
- Loader:     third_party.tt_forge_models.snake7gun_tiny_random_qwen3.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINY_RANDOM_QWEN3

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  555.1091636805074
- TTFT (ms):          26.453981
- Prefill PCC:        0.999234
- First decode PCC:   0.998275
- Wall clock:         0:01:04
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_snake7gun_tiny_random_qwen3_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 1.7% (555.1 / 32064.8)

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
- total_flops:             627048480
- breakdown.matmul:        627048480
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

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
- bound:                    dram
- top_perf_samples_per_sec: 32064.8492
- top_perf_time_ms:         0.0312
- dram_time_ms:             0.0208
- compute_time_ms_lofi:     0.0007
- compute_time_ms_hifi2:    0.0014
- compute_time_ms_hifi3:    0.0021
- compute_time_ms_hifi4:    0.0029

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: use hasattr for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
