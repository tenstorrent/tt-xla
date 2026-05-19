loader_path: third_party.tt_forge_models.ziadky_velora_ai_supported_models_gguf.causal_lm.pytorch.loader
variant_id: CRYSTAL_THINK_V2_Q4
arch: p150
status: DONE_PASS
test_function: test_ziadky_velora_crystal_think_v2_q4
samples_per_second: 35.082150669862635
ttft_ms: 320.455495
prefill_pcc: 0.997637
first_decode_pcc: 0.997533
top_perf_samples_per_sec: 389.7574
pct_of_target: 9.0
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_ziadky_velora_crystal_think_v2_q4

## Test
tests/benchmark/test_llms.py::test_ziadky_velora_crystal_think_v2_q4

## Model
- HF name:    ZiADKY/VeloraAI_SupportedModels
- Loader:     third_party.tt_forge_models.ziadky_velora_ai_supported_models_gguf.causal_lm.pytorch.loader
- Variant:    CRYSTAL_THINK_V2_Q4 (GGUF file: Crystal_Think_V2_Q4.gguf)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  35.082150669862635
- TTFT (ms):          320.455495
- Prefill PCC:        0.997637
- First decode PCC:   0.997533
- Wall clock:         0:08:37
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_ziadky_velora_crystal_think_v2_q4_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 9.0% (35.08 / 389.76 samples/sec)

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
- total_flops:             752604940416
- breakdown.matmul:        752604940416
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        25165824
- memory_bytes: 50331648
- memory_gb:    0.046875

### Params
- count:                  1080707523
- effective_count:        691751363
- memory_bytes:           1512916232
- memory_gb:              1.4090130403637886
- effective_memory_bytes: 735003912
- effective_memory_gb:    0.6845257356762886
- embedding_count:        388956160
- embedding_memory_bytes: 777912320

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 389.7574
- top_perf_time_ms:         2.5657
- dram_time_ms:             1.4440
- compute_time_ms_lofi:     0.8552
- compute_time_ms_hifi2:    1.7105
- compute_time_ms_hifi3:    2.5657
- compute_time_ms_hifi4:    3.4209

## Infrastructure fix
Fixed a general benchmarking infrastructure bug in `tests/benchmark/benchmarks/llm_benchmark.py`:
`model_loader.get_weight_dtype_config_path()` was called unconditionally but is not part of the
`ForgeModel` base class. Fixed with `getattr` fallback so loaders without this method gracefully skip.

## Files changed
- tests/benchmark/test_llms.py (added test_ziadky_velora_crystal_think_v2_q4)
- tests/benchmark/benchmarks/llm_benchmark.py (general infra fix: safe getattr for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added ziadky_velora_crystal_think_v2_q4 entry)

## tt-forge-models submodule
no change
