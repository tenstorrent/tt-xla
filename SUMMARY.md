loader_path: third_party.tt_forge_models.deepseek_r1_distill_qwen_1_5b_4bit.causal_lm.pytorch.loader
variant_id: DeepSeek_R1_Distill_Qwen_1.5B_4bit
arch: n150
status: DONE_PASS
test_function: test_deepseek_r1_distill_qwen_1_5b_4bit
samples_per_second: 34.014231518597825
ttft_ms: 368.5062
prefill_pcc: 0.974534
first_decode_pcc: 0.967126
top_perf_samples_per_sec: 116.1868
pct_of_target: 29.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: test_deepseek_r1_distill_qwen_1_5b_4bit

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_qwen_1_5b_4bit

## Model
- HF name:    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_qwen_1_5b_4bit.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPSEEK_R1_DISTILL_QWEN_1_5B_4BIT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: none (disabled — bfp_bf8 caused first decode PCC to drop below 0.94; full model got 0.938 with bfp_bf8 vs 0.967 without)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  34.01
- TTFT (ms):          368.5
- Prefill PCC:        0.974534
- First decode PCC:   0.967126
- Wall clock:         0:08:54
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_distill_qwen_1_5b_4bit_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 29.3% (34.01 / 116.19)

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
- memory_bytes:           3554176776
- memory_gb:              3.310085065662861
- effective_memory_bytes: 3087429384
- effective_memory_gb:    2.875392682850361
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
- tests/benchmark/test_llms.py (added test_deepseek_r1_distill_qwen_1_5b_4bit)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path to use hasattr guard)

## tt-forge-models submodule
no change
