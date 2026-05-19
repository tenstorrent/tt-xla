loader_path: third_party.tt_forge_models.cohere.causal_lm.pytorch.loader
variant_id: tiny_random
arch: p150
status: DONE_PASS
test_function: test_cohere_tiny_random
samples_per_second: 559.78
ttft_ms: 16.94
prefill_pcc: 0.999909
first_decode_pcc: 0.999938
top_perf_samples_per_sec: 578063.7236
pct_of_target: 0.1
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_cohere_tiny_random

## Test
tests/benchmark/test_llms.py::test_cohere_tiny_random

## Model
- HF name:    optimum-intel-internal-testing/tiny-random-CohereForCausalLM
- Loader:     third_party.tt_forge_models.cohere.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINY_RANDOM

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 causes "No circular buffer with id 0 exists in Program" runtime error on p150.

## Measured (full model, defaults)
- Sample per second:  559.78
- TTFT (ms):          16.94
- Prefill PCC:        0.999909
- First decode PCC:   0.999938
- Wall clock:         0:00:09
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_cohere_tiny_random_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 0.1% (tiny model dominated by fixed overhead, not DRAM throughput)

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
- total_flops:             86130912
- breakdown.matmul:        86130912
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        924
- memory_bytes: 3696

### KV cache
- count:        524288
- memory_bytes: 1048576
- memory_gb:    0.0009765625

### Params
- count:                  81064
- effective_count:        48296
- memory_bytes:           117334
- memory_gb:              0.00010927580296993256
- effective_memory_bytes: 51798
- effective_memory_gb:    4.8240646719932556e-05
- embedding_count:        32768
- embedding_memory_bytes: 65536

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 578063.7236
- top_perf_time_ms:         0.0017
- dram_time_ms:             0.0012
- compute_time_ms_lofi:     0.0001
- compute_time_ms_hifi2:    0.0002
- compute_time_ms_hifi3:    0.0003
- compute_time_ms_hifi4:    0.0004

## Files changed
- tests/benchmark/test_llms.py (added test_cohere_tiny_random)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: use hasattr check before calling get_weight_dtype_config_path)

## tt-forge-models submodule
no change
