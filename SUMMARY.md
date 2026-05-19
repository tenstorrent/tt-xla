loader_path: third_party.tt_forge_models.llama_600m_v4_unigram.causal_lm.pytorch.loader
variant_id: llama_600m_v4_unigram
arch: p150
status: DONE_PASS
test_function: test_llama_600m_v4_unigram
samples_per_second: 136.856
ttft_ms: 72.404
prefill_pcc: 0.997480
first_decode_pcc: 0.998529
top_perf_samples_per_sec: 415.900
pct_of_target: 32.9
roofline_bound: compute
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_llama_600m_v4_unigram

## Test
tests/benchmark/test_llms.py::test_llama_600m_v4_unigram

## Model
- HF name:    deqing/llama-600M-v4-unigram
- Loader:     third_party.tt_forge_models.llama_600m_v4_unigram.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_600M_V4_UNIGRAM

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 fails with ttnn.paged_update_cache sharding error
(expects sharded input but gets interleaved tensor at blackhole/p150).

## Measured (full model, defaults)
- Sample per second:  136.856
- TTFT (ms):          72.404
- Prefill PCC:        0.997480
- First decode PCC:   0.998529
- Wall clock:         0:00:56
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_600m_v4_unigram_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 32.9% (136.856 / 415.900)

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
- total_flops:             705297384576
- breakdown.matmul:        705297384576
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        594
- memory_bytes: 2376

### KV cache
- count:        50331648
- memory_bytes: 100663296
- memory_gb:    0.09375

### Params
- count:                  809277091
- effective_count:        612275875
- memory_bytes:           1044582024
- memory_gb:              0.9728428199887276
- effective_memory_bytes: 650579592
- effective_memory_gb:    0.6058994606137276
- embedding_count:        197001216
- embedding_memory_bytes: 394002432

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 415.900
- top_perf_time_ms:         2.4044
- dram_time_ms:             1.3346
- compute_time_ms_lofi:     0.8015
- compute_time_ms_hifi2:    1.6029
- compute_time_ms_hifi3:    2.4044
- compute_time_ms_hifi4:    3.2059

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
