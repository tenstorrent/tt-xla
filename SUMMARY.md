loader_path: third_party.tt_forge_models.deepseek_r1_distill_qwen_1_5b_mlx.causal_lm.pytorch.loader
variant_id: DeepSeek_R1_Distill_Qwen_1.5B_MLX
arch: p150
status: DONE_PASS
test_function: test_deepseek_r1_distill_qwen_1_5b_mlx
samples_per_second: 67.55
ttft_ms: 151.12
prefill_pcc: 0.977839
first_decode_pcc: 0.980288
top_perf_samples_per_sec: 206.5544
pct_of_target: 32.7
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_deepseek_r1_distill_qwen_1_5b_mlx

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_qwen_1_5b_mlx

## Model
- HF name:    mlx-community/DeepSeek-R1-Distill-Qwen-1.5B
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_qwen_1_5b_mlx.causal_lm.pytorch.loader
- Variant:    DeepSeek_R1_Distill_Qwen_1.5B_MLX

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 fails with a compiler error:
  ttnn.paged_update_cache requires sharded input but receives non-sharded tensor.
  This is a known compiler limitation on p150/blackhole at the current tt-mlir version.

## Measured (full model, defaults)
- Sample per second:  67.55
- TTFT (ms):          151.12
- Prefill PCC:        0.977839
- First decode PCC:   0.980288
- Wall clock:         0:02:00
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_distill_qwen_1_5b_mlx_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 32.7% (67.55 / 206.55)

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
- tests/benchmark/test_llms.py (new test_deepseek_r1_distill_qwen_1_5b_mlx function)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change (submodule HEAD: 05a2a92b7e)
