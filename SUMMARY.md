loader_path: third_party.tt_forge_models.qwen2_5_coder_3b_q8_0_gguf.causal_lm.pytorch.loader
variant_id: 3B_Q8_0_GGUF
arch: p150
status: DONE_PASS
test_function: test_qwen_2_5_coder_3b_q8_0_gguf
samples_per_second: 41.666736884588026
ttft_ms: 224.537778
prefill_pcc: 0.996282
first_decode_pcc: 0.998207
top_perf_samples_per_sec: 104.6961
pct_of_target: 39.8
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: qwen_2_5_coder_3b_q8_0_gguf

## Test
tests/benchmark/test_llms.py::test_qwen_2_5_coder_3b_q8_0_gguf

## Model
- HF name:    ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF
- Loader:     third_party.tt_forge_models.qwen2_5_coder_3b_q8_0_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_2_5_CODER_3B_Q8_0_GGUF

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 fails with ttnn.paged_update_cache requiring sharded input tensor.

## Measured (full model, defaults)
- Sample per second:  41.666736884588026
- TTFT (ms):          224.537778
- Prefill PCC:        0.996282
- First decode PCC:   0.998207
- Wall clock:         0:03:24
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_qwen_2_5_coder_3b_q8_0_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 39.8% (41.67 / 104.70)

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
- total_flops:             197487558784
- breakdown.matmul:        185405014144
- breakdown.linear:        12082544640
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  3397103811
- effective_count:        3085938883
- memory_bytes:           3901367048
- memory_gb:              3.633431203663349
- effective_memory_bytes: 3279037192
- effective_memory_gb:    3.053841359913349
- embedding_count:        311164928
- embedding_memory_bytes: 622329856

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 104.6961
- top_perf_time_ms:         9.5515
- dram_time_ms:             6.3676
- compute_time_ms_lofi:     0.2244
- compute_time_ms_hifi2:    0.4488
- compute_time_ms_hifi3:    0.6733
- compute_time_ms_hifi4:    0.8977

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: add hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
