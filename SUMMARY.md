loader_path: third_party.tt_forge_models.deepcogito_gguf.causal_lm.pytorch.loader
variant_id: Cogito_v1_Preview_Llama_3B_GGUF
arch: p150
status: DONE_PASS
test_function: test_deepcogito_cogito_v1_preview_llama_3b_gguf
samples_per_second: 59.953
ttft_ms: 198.315
prefill_pcc: 0.999183
first_decode_pcc: 0.998635
top_perf_samples_per_sec: 96.005
pct_of_target: 62.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_deepcogito_cogito_v1_preview_llama_3b_gguf

## Test
tests/benchmark/test_llms.py::test_deepcogito_cogito_v1_preview_llama_3b_gguf

## Model
- HF name:    bartowski/deepcogito_cogito-v1-preview-llama-3B-GGUF
- Loader:     third_party.tt_forge_models.deepcogito_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.COGITO_V1_PREVIEW_LLAMA_3B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  59.953
- TTFT (ms):          198.315
- Prefill PCC:        0.999183
- First decode PCC:   0.998635
- Wall clock:         0:06:36
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepcogito_cogito_v1_preview_llama_3b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 62.4% (59.953 / 96.005)

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
- total_flops:             205604782208
- breakdown.matmul:        205604782208
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        234881024
- memory_bytes: 469762048
- memory_gb:    0.4375

### Params
- count:                  3606752451
- effective_count:        3212750019
- memory_bytes:           4201716488
- memory_gb:              3.9131534174084663
- effective_memory_bytes: 3413711624
- effective_memory_gb:    3.1792666986584663
- embedding_count:        394002432
- embedding_memory_bytes: 788004864

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 96.005
- top_perf_time_ms:         10.4161
- dram_time_ms:             6.9441
- compute_time_ms_lofi:     0.2336
- compute_time_ms_hifi2:    0.4673
- compute_time_ms_hifi3:    0.7009
- compute_time_ms_hifi4:    0.9346

## Files changed
- tests/benchmark/test_llms.py (new test function test_deepcogito_cogito_v1_preview_llama_3b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (new entry deepcogito_cogito_v1_preview_llama_3b_gguf)

## tt-forge-models submodule
no change
