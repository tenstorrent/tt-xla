loader_path: third_party.tt_forge_models.bitnet_b1_58_large_tq2_0_gguf.causal_lm.pytorch.loader
variant_id: bitnet_b1_58_large_TQ2_0
arch: p150
status: DONE_PASS
test_function: test_bitnet_b1_58_large_tq2_0_gguf
samples_per_second: 56.16
ttft_ms: 258.40
prefill_pcc: 0.999031
first_decode_pcc: 0.998979
top_perf_samples_per_sec: 321.089
pct_of_target: 17.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_bitnet_b1_58_large_tq2_0_gguf

## Test
tests/benchmark/test_llms.py::test_bitnet_b1_58_large_tq2_0_gguf

## Model
- HF name:    gianni-cor/bitnet_b1_58-large-TQ2_0
- Loader:     third_party.tt_forge_models.bitnet_b1_58_large_tq2_0_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BITNET_B1_58_LARGE_TQ2_0

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  56.16
- TTFT (ms):          258.40
- Prefill PCC:        0.999031
- First decode PCC:   0.998979
- Wall clock:         0:05:14
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bitnet_b1_58_large_tq2_0_gguf_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 17.5% (56.16 / 321.09)

Note: Gap from roofline is large (17.5%) but all performance knobs
(optimization_level=2, trace_enabled=True, experimental_weight_dtype=bfp_bf8)
are already at their most aggressive settings. The model is DRAM-bound.

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
- total_flops:             792751965792
- breakdown.matmul:        792751965792
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  777998003
- effective_count:        728842931
- memory_bytes:           872903560
- memory_gb:              0.8129547908902168
- effective_memory_bytes: 774593416
- effective_memory_gb:    0.7213963344693184
- embedding_count:        49155072
- embedding_memory_bytes: 98310144

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 321.08979532151375
- top_perf_time_ms:         3.1144
- dram_time_ms:             2.0763
- compute_time_ms_lofi:     0.9009
- compute_time_ms_hifi2:    1.8017
- compute_time_ms_hifi3:    2.7026
- compute_time_ms_hifi4:    3.6034

## Files changed
- tests/benchmark/test_llms.py (added test_bitnet_b1_58_large_tq2_0_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: handle loaders without get_weight_dtype_config_path)

## tt-forge-models submodule
no change
