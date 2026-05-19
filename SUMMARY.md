loader_path: third_party.tt_forge_models.llama_3_nectar_dpo_8b.causal_lm.pytorch.loader
variant_id: Llama_3_Nectar_DPO_8B
arch: p150
status: DONE_PASS
test_function: test_llama_3_nectar_dpo_8b
samples_per_second: 33.809
ttft_ms: 310.228
prefill_pcc: 0.998587
first_decode_pcc: 0.998576
top_perf_samples_per_sec: 342.5017
pct_of_target: 9.9
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_llama_3_nectar_dpo_8b

## Test
tests/benchmark/test_llms.py::test_llama_3_nectar_dpo_8b

## Model
- HF name:    ibivibiv/llama-3-nectar-dpo-8B
- Loader:     third_party.tt_forge_models.llama_3_nectar_dpo_8b.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_3_NECTAR_DPO_8B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.809
- TTFT (ms):          310.228
- Prefill PCC:        0.998587
- First decode PCC:   0.998576
- Wall clock:         0:08:36
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_3_nectar_dpo_8b_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 9.9% (33.809 / 342.5017)

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
- total_flops:             856443324672
- breakdown.matmul:        856443324672
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        594
- memory_bytes: 2376

### KV cache
- count:        8388608
- memory_bytes: 16777216
- memory_gb:    0.015625

### Params
- count:                  1268789443
- effective_count:        743452867
- memory_bytes:           1840603912
- memory_gb:              1.7141959741711617
- effective_memory_bytes: 789930760
- effective_memory_gb:    0.7356803491711617
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 342.5017
- top_perf_time_ms:         2.9197
- dram_time_ms:             1.5143
- compute_time_ms_lofi:     0.9732
- compute_time_ms_hifi2:    1.9465
- compute_time_ms_hifi3:    2.9197
- compute_time_ms_hifi4:    3.8929

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
