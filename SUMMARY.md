loader_path: third_party.tt_forge_models.deepseek_r1_distill_qwen_1_5b_uncensored_gguf.causal_lm.pytorch.loader
variant_id: deepseek_r1_distill_qwen_1_5b_uncensored_Q4_K_M_GGUF
arch: n150
status: DONE_FAIL
test_function: test_deepseek_r1_distill_qwen_1_5b_uncensored_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 116.1868
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "First decode PCC 0.902 does not reach required 0.94 with opt2+bfp_bf8; opt1 causes prefill to fail at 0.914; opt2 without bfp_bf8 yields decode PCC 0.890. Q4_K_M GGUF + bfp_bf8 double quantization accumulates error across all 28 layers beyond tolerable threshold. Passes at num_layers=1."

# Benchmark added: test_deepseek_r1_distill_qwen_1_5b_uncensored_gguf

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_qwen_1_5b_uncensored_gguf

## Model
- HF name:    mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_qwen_1_5b_uncensored_gguf.causal_lm.pytorch.loader
- Variant:    deepseek_r1_distill_qwen_1_5b_uncensored_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test FAILED — PCC threshold not met)
- TTFT (ms):          N/A
- Prefill PCC:        0.946487 (best observed, opt2+bfp_bf8, run that failed at decode)
- First decode PCC:   0.901733 (best observed, opt2+bfp_bf8 — fails 0.94 threshold)
- Wall clock:         ~12:14 (opt2 full run)
- Hardware:           n150 (wormhole_b0)

## PCC investigation summary
| Config | Prefill PCC | Decode PCC | Result |
|--------|------------|------------|--------|
| opt2 + bfp_bf8 (default) | 0.946 | 0.902 | FAIL (decode) |
| opt2, no bfp_bf8 | 0.950 | 0.890 | FAIL (decode) |
| opt2 + fp32_dest_acc | 0.946 | 0.902 | FAIL (decode) |
| opt1 + bfp_bf8 | 0.914 | N/A | FAIL (prefill) |
| 1-layer + opt2 + bfp_bf8 | 0.965 | 0.998 | PASS |

The model passes accuracy at 1 layer but fails at full depth (28 layers).
Root cause: Q4_K_M GGUF quantization combined with bfp_bf8 weight conversion
accumulates error beyond the 0.94 threshold across 28 transformer layers.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_distill_qwen_1_5b_uncensored_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (test failed before clean benchmark)

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             98790277248
- breakdown.matmul:        93151297664
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
- count:                  1777088195
- effective_count:        1543714499
- memory_bytes:           2107080456
- memory_gb:              1.9623715952038765
- effective_memory_bytes: 1640333064
- effective_memory_gb:    1.5276792123913765
- embedding_count:        233373696
- embedding_memory_bytes: 466747392

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 116.1868
- top_perf_time_ms:         8.6068
- dram_time_ms:             5.7379
- compute_time_ms_lofi:     0.3859
- compute_time_ms_hifi2:    0.7718
- compute_time_ms_hifi3:    1.1577
- compute_time_ms_hifi4:    1.5436

## Files changed
- tests/benchmark/test_llms.py (added test_deepseek_r1_distill_qwen_1_5b_uncensored_gguf with FAILED comment)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added test entry)

## tt-forge-models submodule
no change
