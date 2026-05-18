loader_path: third_party.tt_forge_models.gpt4chan_8b_gguf.causal_lm.pytorch.loader
variant_id: 8B_GGUF
arch: p150
status: DONE_PASS
test_function: test_gpt4chan_8b_gguf
samples_per_second: 33.6491255920473
ttft_ms: 310.708245
prefill_pcc: 0.997646
first_decode_pcc: 0.996752
top_perf_samples_per_sec: 42.5800
pct_of_target: 79.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: gpt4chan_8b_gguf

## Test
tests/benchmark/test_llms.py::test_gpt4chan_8b_gguf

## Model
- HF name:    mradermacher/GPT4chan-8B-GGUF
- Loader:     third_party.tt_forge_models.gpt4chan_8b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GPT4CHAN_8B_GGUF (8B_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.6491255920473
- TTFT (ms):          310.708245
- Prefill PCC:        0.997646
- First decode PCC:   0.996752
- Wall clock:         0:09:30
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gpt4chan_8b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 79.0%

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
- total_flops:             480298664064
- breakdown.matmul:        480298664064
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
- count:                  8030277827
- effective_count:        7504933059
- memory_bytes:           9024931080
- memory_gb:              8.405122049152851
- effective_memory_bytes: 7974241544
- effective_memory_gb:    7.426591165363789
- embedding_count:        525344768
- embedding_memory_bytes: 1050689536

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5800
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: added hasattr check for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
