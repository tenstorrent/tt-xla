loader_path: third_party.tt_forge_models.dolphin3_0_llama3_2_3b_mlx_4bit.causal_lm.pytorch.loader
variant_id: 3B_MLX_4bit
arch: p150
status: DONE_PASS
test_function: test_dolphin3_0_llama3_2_3b_mlx_4bit
samples_per_second: 33.83
ttft_ms: 397.748
prefill_pcc: 0.998637
first_decode_pcc: 0.998581
top_perf_samples_per_sec: 96.0049
pct_of_target: 35.2
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_dolphin3_0_llama3_2_3b_mlx_4bit

## Test
tests/benchmark/test_llms.py::test_dolphin3_0_llama3_2_3b_mlx_4bit

## Model
- HF name:    mlx-community/dolphin3.0-llama3.2-3B-4Bit
- Loader:     third_party.tt_forge_models.dolphin3_0_llama3_2_3b_mlx_4bit.causal_lm.pytorch.loader
- Variant:    ModelVariant.DOLPHIN3_0_LLAMA3_2_3B_MLX_4BIT

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 causes RESOURCE_EXHAUSTED (OOM) on the full 28-layer model
on p150. The SRAM optimization in opt_level=2 saturates SRAM when all 28 layers are
present. opt_level=1 (all tensors in DRAM) resolves this.

## Measured (full model, optimization_level=1)
- Sample per second:  33.83
- TTFT (ms):          397.748
- Prefill PCC:        0.998637
- First decode PCC:   0.998581
- Wall clock:         0:03:19
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_dolphin3_0_llama3_2_3b_mlx_4bit_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 35.2% (33.83 / 96.0049)
Note: lower than ideal because optimization_level=1 (not 2) is required to avoid OOM.

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
- total_flops:             205605175424
- breakdown.matmul:        205605175424
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
- count:                  3606764739
- effective_count:        3212756163
- memory_bytes:           4201735304
- memory_gb:              3.9131709411740303
- effective_memory_bytes: 3413718152
- effective_memory_gb:    3.1792727783322334
- embedding_count:        394008576
- embedding_memory_bytes: 788017152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 96.0049
- top_perf_time_ms:         10.4161
- dram_time_ms:             6.9441
- compute_time_ms_lofi:     0.2336
- compute_time_ms_hifi2:    0.4673
- compute_time_ms_hifi3:    0.7009
- compute_time_ms_hifi4:    0.9346

## Files changed
- tests/benchmark/test_llms.py (added test_dolphin3_0_llama3_2_3b_mlx_4bit)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added dolphin3_0_llama3_2_3b_mlx_4bit entry)
- third_party/tt_forge_models (updated submodule pointer)
- SUMMARY.md

## tt-forge-models submodule
93218a34fc9fc6a671e0e41101da470c80891b2a → c9b45c4dfe71bf9beed21e9db576f2728db20aeb
Updated to include dolphin3_0_llama3_2_3b_mlx_4bit loader from bringup branch
arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-29
