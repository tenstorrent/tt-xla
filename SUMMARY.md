loader_path: third_party.tt_forge_models.tiny_random_smollm3.causal_lm.pytorch.loader
variant_id: tiny_random_smollm3
arch: p150
status: DONE_PASS
test_function: test_tiny_random_smollm3
samples_per_second: 252.667
ttft_ms: 31.38
prefill_pcc: 0.990835
first_decode_pcc: 0.998454
top_perf_samples_per_sec: 32569.1367
pct_of_target: 0.8
roofline_bound: compute
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_tiny_random_smollm3

## Test
tests/benchmark/test_llms.py::test_tiny_random_smollm3

## Model
- HF name:    optimum-internal-testing/tiny-random-SmolLM3ForCausalLM
- Loader:     third_party.tt_forge_models.tiny_random_smollm3.causal_lm.pytorch.loader
- Variant:    TINY_RANDOM_SMOLLM3

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Notes:
- optimization_level=0: levels 1 (Prefill PCC=0.477 < 0.94) and 2 (compiler circular buffer error) both fail
- Infrastructure fix: added hasattr guard for get_weight_dtype_config_path in llm_benchmark.py

## Measured (full model, defaults)
- Sample per second:  252.667
- TTFT (ms):          31.38
- Prefill PCC:        0.990835
- First decode PCC:   0.998454
- Wall clock:         0:00:29
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_tiny_random_smollm3_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 0.8% (252.667 / 32569.1367)

Note: This is a tiny test model (16.4M params) — the roofline ceiling is extremely high
relative to actual throughput because overhead dominates for such a small model.

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
- total_flops:             9006481680
- breakdown.matmul:        9006481680
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
- count:                  16454094
- effective_count:        8245710
- memory_bytes:           25178548
- memory_gb:              0.023449350148439407
- effective_memory_bytes: 8761780
- effective_memory_gb:    0.008160043507814407
- embedding_count:        8208384
- embedding_memory_bytes: 16416768

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 32569.1367
- top_perf_time_ms:         0.0307
- dram_time_ms:             0.0177
- compute_time_ms_lofi:     0.0102
- compute_time_ms_hifi2:    0.0205
- compute_time_ms_hifi3:    0.0307
- compute_time_ms_hifi4:    0.0409

## Files changed
- tests/benchmark/test_llms.py (added test_tiny_random_smollm3)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added tiny_random_smollm3 entry)

## tt-forge-models submodule
no change
