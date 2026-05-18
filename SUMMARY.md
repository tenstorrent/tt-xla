loader_path: third_party.tt_forge_models.mradermacher_llama_3_1_8b_extended_i1_gguf.causal_lm.pytorch.loader
variant_id: 8B_Extended_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_mradermacher_llama_3_1_8b_extended_i1_gguf
samples_per_second: 31.782527455226997
ttft_ms: 314.709222
prefill_pcc: 0.999399
first_decode_pcc: 0.998805
top_perf_samples_per_sec: 42.0761
pct_of_target: 75.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_mradermacher_llama_3_1_8b_extended_i1_gguf

## Test
tests/benchmark/test_llms.py::test_mradermacher_llama_3_1_8b_extended_i1_gguf

## Model
- HF name:    mradermacher/Llama-3.1-8B-Extended-i1-GGUF
- Loader:     third_party.tt_forge_models.mradermacher_llama_3_1_8b_extended_i1_gguf.causal_lm.pytorch.loader
- Variant:    LLAMA_3_1_8B_EXTENDED_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  31.782527455226997
- TTFT (ms):          314.709222
- Prefill PCC:        0.999399
- First decode PCC:   0.998805
- Wall clock:         0:09:24
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mradermacher_llama_3_1_8b_extended_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 75.5%

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
- total_flops:             486255886464
- breakdown.matmul:        486255886464
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
- count:                  8216441027
- effective_count:        7598014659
- memory_bytes:           9309993480
- memory_gb:              8.670607097446918
- effective_memory_bytes: 8073140744
- effective_memory_gb:    7.5186982229352
- embedding_count:        618426368
- embedding_memory_bytes: 1236852736

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.0761
- top_perf_time_ms:         23.7664
- dram_time_ms:             15.8443
- compute_time_ms_lofi:     0.5526
- compute_time_ms_hifi2:    1.1051
- compute_time_ms_hifi3:    1.6577
- compute_time_ms_hifi4:    2.2103

## Files changed
- tests/benchmark/test_llms.py (added test_mradermacher_llama_3_1_8b_extended_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
