loader_path: third_party.tt_forge_models.poe_8b_glm5_gguf.causal_lm.pytorch.loader
variant_id: dripghad1223_8B_GLM5_GGUF
arch: p150
status: DONE_PASS
test_function: test_poe_8b_glm5_gguf
samples_per_second: 20.15
ttft_ms: 502.05
prefill_pcc: 0.994386
first_decode_pcc: 0.999003
top_perf_samples_per_sec: 42.0551
pct_of_target: 47.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_poe_8b_glm5_gguf

## Test
tests/benchmark/test_llms.py::test_poe_8b_glm5_gguf

## Model
- HF name:    Crownelius/Poe-8B-GLM5-Opus4.6-Sonnet4.5-Kimi-Grok-Gemini-3-pro-preview-HERETIC
- Loader:     third_party.tt_forge_models.poe_8b_glm5_gguf.causal_lm.pytorch.loader
- Variant:    dripghad1223_8B_GLM5_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  20.15
- TTFT (ms):          502.05
- Prefill PCC:        0.994386
- First decode PCC:   0.999003
- Wall clock:         0:12:50
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_poe_8b_glm5_gguf_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 47.9%

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
- total_flops:             484358238208
- breakdown.matmul:        484358238208
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        34
- memory_bytes: 134

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  8190742087
- effective_count:        7568406084
- memory_bytes:           9286405784
- memory_gb:              8.648639343678951
- effective_memory_bytes: 8041721484
- effective_memory_gb:    7.489436756819487
- embedding_count:        622336003
- embedding_memory_bytes: 1244684300

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.0551
- top_perf_time_ms:         23.7784
- dram_time_ms:             15.8522
- compute_time_ms_lofi:     0.5504
- compute_time_ms_hifi2:    1.1008
- compute_time_ms_hifi3:    1.6512
- compute_time_ms_hifi4:    2.2016

## Files changed
- tests/benchmark/test_llms.py (added test_poe_8b_glm5_gguf)
- tests/benchmark/llm_utils/decode_utils.py (general fix: handle VL model configs with nested text_config in init_static_cache)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added poe_8b_glm5_gguf entry)

## tt-forge-models submodule
no change
