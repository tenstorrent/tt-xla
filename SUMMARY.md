loader_path: third_party.tt_forge_models.llamantino_2_chat_13b_hf_ultrachat_ita.causal_lm.pytorch.loader
variant_id: llamantino_2_chat_13b_hf_ultrachat_ita
arch: p150
status: DONE_PASS
test_function: test_llamantino_2_chat_13b_hf_ultrachat_ita
samples_per_second: 12.914636861731362
ttft_ms: 650.765363
prefill_pcc: 0.997257
first_decode_pcc: 0.947041
top_perf_samples_per_sec: 22.7801
pct_of_target: 56.7
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_llamantino_2_chat_13b_hf_ultrachat_ita

## Test
tests/benchmark/test_llms.py::test_llamantino_2_chat_13b_hf_ultrachat_ita

## Model
- HF name:    swap-uniba/LLaMAntino-2-chat-13b-hf-UltraChat-ITA
- Loader:     third_party.tt_forge_models.llamantino_2_chat_13b_hf_ultrachat_ita.causal_lm.pytorch.loader
- Variant:    LLAMANTINO_2_CHAT_13B_HF_ULTRACHAT_ITA

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  12.914636861731362
- TTFT (ms):          650.765363
- Prefill PCC:        0.997257
- First decode PCC:   0.947041
- Wall clock:         0:05:36
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llamantino_2_chat_13b_hf_ultrachat_ita_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 56.7% (12.91 / 22.78 samples/sec)

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
- total_flops:             822503342208
- breakdown.matmul:        822503342208
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1677721600
- memory_bytes: 3355443200
- memory_gb:    3.125

### Params
- count:                  13015874755
- effective_count:        12852029635
- memory_bytes:           13983361096
- memory_gb:              13.023
- effective_memory_bytes: 13655670856
- effective_memory_gb:    12.718
- embedding_count:        163845120
- embedding_memory_bytes: 327690240

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.7801
- top_perf_time_ms:         43.8979
- dram_time_ms:             29.2652
- compute_time_ms_lofi:     0.9347
- compute_time_ms_hifi2:    1.8693
- compute_time_ms_hifi3:    2.8040
- compute_time_ms_hifi4:    3.7387

## Files changed
- tests/benchmark/test_llms.py (added test_llamantino_2_chat_13b_hf_ultrachat_ita)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: graceful handling of missing get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added llamantino_2_chat_13b_hf_ultrachat_ita entry)

## tt-forge-models submodule
no change — submodule at 215d1080a28b3d49f8be77cc4243701ca9586b70
