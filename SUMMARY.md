loader_path: third_party.tt_forge_models.fxmarty_tiny_llama_fast_tokenizer.causal_lm.pytorch.loader
variant_id: tiny_llama_fast_tokenizer
arch: p150
status: DONE_PASS
test_function: test_fxmarty_tiny_llama_fast_tokenizer
samples_per_second: 790.43
ttft_ms: 13.34
prefill_pcc: 0.998017
first_decode_pcc: 0.999463
top_perf_samples_per_sec: 422963.62
pct_of_target: 0.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_fxmarty_tiny_llama_fast_tokenizer

## Test
tests/benchmark/test_llms.py::test_fxmarty_tiny_llama_fast_tokenizer

## Model
- HF name:    fxmarty/tiny-llama-fast-tokenizer
- Loader:     third_party.tt_forge_models.fxmarty_tiny_llama_fast_tokenizer.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINY_LLAMA_FAST_TOKENIZER

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  790.43
- TTFT (ms):          13.34
- Prefill PCC:        0.998017
- First decode PCC:   0.999463
- Wall clock:         0:00:38
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_fxmarty_tiny_llama_fast_tokenizer_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 0.2%

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
- total_flops:             599261256
- breakdown.matmul:        599261256
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        594
- memory_bytes: 2376

### KV cache
- count:        262144
- memory_bytes: 524288
- memory_gb:    0.00048828125

### Params
- count:                  1032405
- effective_count:        520405
- memory_bytes:           1577392
- memory_gb:              0.0014690607786178589
- effective_memory_bytes: 553392
- effective_memory_gb:    0.0005153864622116089
- embedding_count:        512000
- embedding_memory_bytes: 1024000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 422963.6202
- top_perf_time_ms:         0.0024
- dram_time_ms:             0.0016
- compute_time_ms_lofi:     0.0007
- compute_time_ms_hifi2:    0.0014
- compute_time_ms_hifi3:    0.0020
- compute_time_ms_hifi4:    0.0027

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general infrastructure fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
