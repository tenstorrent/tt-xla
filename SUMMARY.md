loader_path: third_party.tt_forge_models.amkkk_qwen3_5_2b_abiliterate_all_layers_baked_gguf_quantized.causal_lm.pytorch.loader
variant_id: 2B_GGUF_Quantized
arch: p150
status: DONE_PASS
test_function: test_amkkk_qwen3_5_2b_abiliterate_all_layers_baked_gguf_quantized
samples_per_second: 81.17
ttft_ms: 188.88
prefill_pcc: 0.998798
first_decode_pcc: 0.998524
top_perf_samples_per_sec: 208.07
pct_of_target: 39.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: amkkk_qwen3_5_2b_abiliterate_all_layers_baked_gguf_quantized

## Test
tests/benchmark/test_llms.py::test_amkkk_qwen3_5_2b_abiliterate_all_layers_baked_gguf_quantized

## Model
- HF name:    amkkk/Qwen3.5_2B_Abiliterate_All_Layers_Baked_GGUF_quantized
- Loader:     third_party.tt_forge_models.amkkk_qwen3_5_2b_abiliterate_all_layers_baked_gguf_quantized.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_2B_ABILITERATE_ALL_LAYERS_BAKED_GGUF_QUANTIZED (2B_GGUF_Quantized)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  81.17
- TTFT (ms):          188.88
- Prefill PCC:        0.998798
- First decode PCC:   0.998524
- Wall clock:         0:06:00
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_amkkk_qwen3_5_2b_abiliterate_all_layers_baked_gguf_quantized_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 39.0% (81.17 / 208.07)

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
- total_flops:             98582921344
- breakdown.matmul:        98582921344
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        50331648
- memory_bytes: 100663296
- memory_gb:    0.09375

### Params
- count:                  2049024195
- effective_count:        1540464835
- memory_bytes:           2653963016
- memory_gb:              2.4716956689953804
- effective_memory_bytes: 1636844296
- effective_memory_gb:    1.5244300439953804
- embedding_count:        508559360
- embedding_memory_bytes: 1017118720

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 208.07
- top_perf_time_ms:         4.8062
- dram_time_ms:             3.2041
- compute_time_ms_lofi:     0.1120
- compute_time_ms_hifi2:    0.2241
- compute_time_ms_hifi3:    0.3361
- compute_time_ms_hifi4:    0.4481

## Files changed
- tests/benchmark/test_llms.py (new test function added)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (new entry added)

## tt-forge-models submodule
no change
