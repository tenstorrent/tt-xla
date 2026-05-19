loader_path: third_party.tt_forge_models.nomic_ai_gpt4all_falcon_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_nomic_ai_gpt4all_falcon_gguf
samples_per_second: 9.793959523110955
ttft_ms: 741.918929
prefill_pcc: 0.994883
first_decode_pcc: 0.998781
top_perf_samples_per_sec: 47.7034
pct_of_target: 20.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_nomic_ai_gpt4all_falcon_gguf

## Test
tests/benchmark/test_llms.py::test_nomic_ai_gpt4all_falcon_gguf

## Model
- HF name:    maddes8cht/nomic-ai-gpt4all-falcon-gguf
- Loader:     third_party.tt_forge_models.nomic_ai_gpt4all_falcon_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.NOMIC_AI_GPT4ALL_FALCON_Q4_K_M (value: "Q4_K_M")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  9.793959523110955
- TTFT (ms):          741.918929
- Prefill PCC:        0.994883
- First decode PCC:   0.998781
- Wall clock:         0:10:15
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_nomic_ai_gpt4all_falcon_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 20.5% (9.79 / 47.70)

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
- total_flops:             445353295936
- breakdown.matmul:        445353295936
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        16777216
- memory_bytes: 33554432
- memory_gb:    0.03125

### Params
- count:                  7217189992
- effective_count:        6921720936
- memory_bytes:           7945548442
- memory_gb:              7.399868631735444
- effective_memory_bytes: 7354610330
- effective_memory_gb:    6.849514627829194
- embedding_count:        295469056
- embedding_memory_bytes: 590938112

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 47.7034
- top_perf_time_ms:         20.9629
- dram_time_ms:             13.9752
- compute_time_ms_lofi:     0.5061
- compute_time_ms_hifi2:    1.0122
- compute_time_ms_hifi3:    1.5182
- compute_time_ms_hifi4:    2.0243

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr check, matching existing pattern in tests/runner/testers/torch/dynamic_torch_model_tester.py)

## tt-forge-models submodule
no change
