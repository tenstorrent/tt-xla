loader_path: third_party.tt_forge_models.pygmalion_2_7b.causal_lm.pytorch.loader
variant_id: 2.7B
arch: p150
status: DONE_PASS
test_function: test_pygmalion_2_7b
samples_per_second: 17.075689320158364
ttft_ms: 383.052119
prefill_pcc: 0.987550
first_decode_pcc: 0.983376
top_perf_samples_per_sec: 99.7811816615593
pct_of_target: 17.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_pygmalion_2_7b

## Test
tests/benchmark/test_llms.py::test_pygmalion_2_7b

## Model
- HF name:    PygmalionAI/pygmalion-2.7b
- Loader:     third_party.tt_forge_models.pygmalion_2_7b.causal_lm.pytorch.loader
- Variant:    ModelVariant.PYGMALION_2_7B (2.7B)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  17.075689320158364
- TTFT (ms):          383.052119
- Prefill PCC:        0.987550
- First decode PCC:   0.983376
- Wall clock:         0:06:42
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_pygmalion_2_7b_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 17.1% (17.08 / 99.78)

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
- total_flops:             170653286400
- breakdown.matmul:        49841602560
- breakdown.linear:        120811683840
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        671088640
- memory_bytes: 1342177280
- memory_gb:    1.25

### Params
- count:                  2779969699
- effective_count:        2646068899
- memory_bytes:           3080035112
- memory_gb:              2.8685062304139137
- effective_memory_bytes: 2812233512
- effective_memory_gb:    2.619096554815769
- embedding_count:        133900800
- embedding_memory_bytes: 267801600

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 99.7811816615593
- top_perf_time_ms:         10.0219298203125
- dram_time_ms:             6.681286546875
- compute_time_ms_lofi:     0.1939241890909091
- compute_time_ms_hifi2:    0.3878483781818182
- compute_time_ms_hifi3:    0.581772567272728
- compute_time_ms_hifi4:    0.7756967563636364

## Files changed
- tests/benchmark/test_llms.py (added test_pygmalion_2_7b)
- .github/workflows/perf-bench-matrix.json (added pygmalionai_pygmalion-2.7b entry)
- tests/benchmark/benchmarks/llm_benchmark.py (general infrastructure fix: hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
