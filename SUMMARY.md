loader_path: third_party.tt_forge_models.gpt2_spanish.causal_lm.pytorch.loader
variant_id: gpt2_spanish
arch: p150
status: DONE_PASS
test_function: test_gpt2_spanish
samples_per_second: 165.63
ttft_ms: 83.01
prefill_pcc: 0.999554
first_decode_pcc: 0.999328
top_perf_samples_per_sec: 1612.6059
pct_of_target: 10.3
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_gpt2_spanish

## Test
tests/benchmark/test_llms.py::test_gpt2_spanish

## Model
- HF name:    DeepESP/gpt2-spanish
- Loader:     third_party.tt_forge_models.gpt2_spanish.causal_lm.pytorch.loader
- Variant:    ModelVariant.GPT2_SPANISH

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  165.63
- TTFT (ms):          83.01
- Prefill PCC:        0.999554
- First decode PCC:   0.999328
- Wall clock:         0:02:07
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gpt2_spanish_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 10.3% (165.63 / 1612.61)

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
- total_flops:             181900197888
- breakdown.matmul:        56815337472
- breakdown.linear:        125084860416
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        759
- memory_bytes: 3036

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  163037316
- effective_count:        123653508
- memory_bytes:           210429500
- memory_gb:              0.19597774371504784
- effective_memory_bytes: 131661884
- effective_memory_gb:    0.12261968478560448
- embedding_count:        39383808
- embedding_memory_bytes: 78767616

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 1612.6059
- top_perf_time_ms:         0.6201
- dram_time_ms:             0.4011
- compute_time_ms_lofi:     0.2067
- compute_time_ms_hifi2:    0.4134
- compute_time_ms_hifi3:    0.6201
- compute_time_ms_hifi4:    0.8268

## Files changed
- tests/benchmark/test_llms.py (added test_gpt2_spanish)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: fallback tokenizer loading for loaders that don't call _load_tokenizer in load_model; guard get_weight_dtype_config_path call for older-style loaders)

## tt-forge-models submodule
no change
