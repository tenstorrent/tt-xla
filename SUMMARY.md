loader_path: third_party.tt_forge_models.qwen2_5_coder_1_5b_instruct_gguf.causal_lm.pytorch.loader
variant_id: ggml_org_Qwen_2_5_Coder_1_5B_Instruct_Q8_0
arch: p150
status: DONE_PASS
test_function: test_qwen_2_5_coder_1_5b_instruct_q8_0
samples_per_second: 65.16
ttft_ms: 153.92
prefill_pcc: 0.997307
first_decode_pcc: 0.997913
top_perf_samples_per_sec: 206.5544
pct_of_target: 31.5
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: qwen_2_5_coder_1_5b_instruct_q8_0

## Test
tests/benchmark/test_llms.py::test_qwen_2_5_coder_1_5b_instruct_q8_0

## Model
- HF name:    ggml-org/Qwen2.5-Coder-1.5B-Instruct-Q8_0-GGUF
- Loader:     third_party.tt_forge_models.qwen2_5_coder_1_5b_instruct_gguf.causal_lm.pytorch.loader
- Variant:    GGML_ORG_QWEN_2_5_CODER_1_5B_INSTRUCT_Q8_0

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  65.16
- TTFT (ms):          153.92
- Prefill PCC:        0.997307
- First decode PCC:   0.997913
- Wall clock:         0:03:09
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_qwen_2_5_coder_1_5b_instruct_q8_0_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 31.5%

Note: optimization_level=1 required because optimization_level=2 fails with
"ttnn.paged_update_cache requires sharded input" (paged_update_cache op
constraint). This accounts for the gap between measured and roofline throughput.

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
- top_perf_samples_per_sec: 206.5544
- top_perf_time_ms:         4.8413
- dram_time_ms:             3.2276
- compute_time_ms_lofi:     0.1123
- compute_time_ms_hifi2:    0.2245
- compute_time_ms_hifi3:    0.3368
- compute_time_ms_hifi4:    0.4490

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
93218a34fc9fc6a671e0e41101da470c80891b2a → 85aa82a4f0
Added GGML_ORG_QWEN_2_5_CODER_1_5B_INSTRUCT_Q8_0 variant and GGUF compat patches
for qwen2_5_coder_1_5b_instruct_gguf loader (hf-bringup-25 branch)
