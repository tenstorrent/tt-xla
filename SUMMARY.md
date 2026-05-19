loader_path: third_party.tt_forge_models.mindchat_qwen2_4b_gguf.causal_lm.pytorch.loader
variant_id: 4B_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_mindchat_qwen2_4b_gguf
samples_per_second: 37.555946410532385
ttft_ms: 315.74251
prefill_pcc: 0.998323
first_decode_pcc: 0.997433
top_perf_samples_per_sec: 75.2203
pct_of_target: 49.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_mindchat_qwen2_4b_gguf

## Test
tests/benchmark/test_llms.py::test_mindchat_qwen2_4b_gguf

## Model
- HF name:    RichardErkhov/X-D-Lab_-_MindChat-Qwen2-4B-gguf
- Loader:     third_party.tt_forge_models.mindchat_qwen2_4b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.MINDCHAT_QWEN2_4B_Q4_K_M_GGUF ("4B_Q4_K_M_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  37.555946410532385
- TTFT (ms):          315.74251
- Prefill PCC:        0.998323
- First decode PCC:   0.997433
- Wall clock:         0:08:55
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mindchat_qwen2_4b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 49.9% (37.6 / 75.2 samples/sec)

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
- total_flops:             227907338368
- breakdown.matmul:        177565859968
- breakdown.linear:        50341478400
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        838860800
- memory_bytes: 1677721600
- memory_gb:    1.5625

### Params
- count:                  3950369475
- effective_count:        3561413315
- memory_bytes:           4562396936
- memory_gb:              4.249063260853291
- effective_memory_bytes: 3784484616
- effective_memory_gb:    3.5245759561657906
- embedding_count:        388956160
- embedding_memory_bytes: 777912320

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 75.2203
- top_perf_time_ms:         13.2943
- dram_time_ms:             8.8629
- compute_time_ms_lofi:     0.2590
- compute_time_ms_hifi2:    0.5180
- compute_time_ms_hifi3:    0.7770
- compute_time_ms_hifi4:    1.0359

## Files changed
- tests/benchmark/test_llms.py (added test_mindchat_qwen2_4b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
