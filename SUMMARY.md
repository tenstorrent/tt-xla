loader_path: third_party.tt_forge_models.deepseek_r1_distill_qwen_1_5b_uncensored_gguf.causal_lm.pytorch.loader
variant_id: deepseek_r1_distill_qwen_1_5b_uncensored_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_deepseek_r1_distill_qwen_1_5b_uncensored_gguf
samples_per_second: 9.365727513359401
ttft_ms: 657.062128
prefill_pcc: 0.993821
first_decode_pcc: 0.990010
top_perf_samples_per_sec: 206.5544
pct_of_target: 4.5
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_deepseek_r1_distill_qwen_1_5b_uncensored_gguf

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_qwen_1_5b_uncensored_gguf

## Model
- HF name:    mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_qwen_1_5b_uncensored_gguf.causal_lm.pytorch.loader
- Variant:    deepseek_r1_distill_qwen_1_5b_uncensored_Q4_K_M_GGUF

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  9.365727513359401
- TTFT (ms):          657.062128
- Prefill PCC:        0.993821
- First decode PCC:   0.990010
- Wall clock:         0:05:16
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_distill_qwen_1_5b_uncensored_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 4.5% (9.37 / 206.55)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             99494920320
- breakdown.matmul:        93855940736
- breakdown.linear:        5638979584
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        58720256
- memory_bytes: 117440512
- memory_gb:    0.109375

### Params
- count:                  1777088198
- effective_count:        1543714502
- memory_bytes:           2107080468
- memory_gb:              1.9623716063797474
- effective_memory_bytes: 1640333076
- effective_memory_gb:    1.5276792235672474
- embedding_count:        233373696
- embedding_memory_bytes: 466747392

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 206.5544
- top_perf_time_ms:         4.8413
- dram_time_ms:             3.2276
- compute_time_ms_lofi:     0.0957
- compute_time_ms_hifi2:    0.1913
- compute_time_ms_hifi3:    0.2870
- compute_time_ms_hifi4:    0.3827

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infra fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
