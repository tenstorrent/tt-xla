loader_path: third_party.tt_forge_models.llm_jp.causal_lm.pytorch.loader
variant_id: 3.1_13B
arch: p150
status: DONE_FAIL
test_function: test_llm_jp_3_1_13b
samples_per_second: 14.35
ttft_ms: 590.08
prefill_pcc: 0.9978
first_decode_pcc: 0.897
top_perf_samples_per_sec: 22.2503
pct_of_target: 64.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "First decode PCC fails with bfp_bf8 weights (0.897 < 0.94 threshold); quantization error accumulates across 40 layers; without bfp_bf8, model OOMs (p150 DRAM insufficient for bf16 13B weights + KV cache at batch_size=32)"

# Benchmark added: test_llm_jp_3_1_13b

## Test
tests/benchmark/test_llms.py::test_llm_jp_3_1_13b

## Model
- HF name:    llm-jp/llm-jp-3.1-13b
- Loader:     third_party.tt_forge_models.llm_jp.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLMJP_3_1_13B (= "3.1_13B")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8 (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  14.35
- TTFT (ms):          590.08
- Prefill PCC:        0.9978  (PASS)
- First decode PCC:   0.8970  (FAIL — below 0.94 threshold)
- Wall clock:         0:12:14
- Hardware:           p150 (Blackhole)

## Failure analysis
The model requires bfp_bf8 weight quantization to fit in DRAM on a single p150
chip (batch_size=32, ISL=128). Without bfp_bf8, the full model (13B bf16 weights
+ KV cache) OOMs during the PCC benchmark phase.

With bfp_bf8, bfloat8 quantization error accumulates across all 40 transformer
layers, degrading the decode PCC from ~0.96 at num_layers=1 to 0.897 at the
full model. All optimization levels tested fail PCC:
- opt_level=2 + bfp_bf8 (full): decode PCC = 0.897 (FAIL)
- opt_level=1 + bfp_bf8 (full): decode PCC = 0.813 (FAIL)
- opt_level=0 + bfp_bf8 (1 layer): PCC denominator=0 / garbage tokens (FAIL)
- no bfp_bf8 (full): OOM during PCC benchmark phase

Per skill instructions, the PCC threshold (0.94) must not be lowered to mask
the numerical issue. This row is recorded as DONE_FAIL.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llm_jp_3_1_13b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 64.5% (14.35 / 22.25)

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
- total_flops:             844648939648
- breakdown.matmul:        844648939648
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
- count:                  13707924675
- effective_count:        13198054595
- memory_bytes:           15043062536
- memory_gb:              14.00994373112917
- effective_memory_bytes: 14023322376
- effective_memory_gb:    13.06023669987917
- embedding_count:        509870080
- embedding_memory_bytes: 1019740160

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.2503
- top_perf_time_ms:         44.9433
- dram_time_ms:             29.9622
- compute_time_ms_lofi:     0.9598
- compute_time_ms_hifi2:    1.9197
- compute_time_ms_hifi3:    2.8795
- compute_time_ms_hifi4:    3.8393

## Files changed
- tests/benchmark/test_llms.py (added test_llm_jp_3_1_13b)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path to use hasattr guard, general infrastructure fix)

## tt-forge-models submodule
no change
