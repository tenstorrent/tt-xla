loader_path: third_party.tt_forge_models.ko_gpt_trinity.causal_lm.pytorch.loader
variant_id: ko_gpt_trinity_1_2B_v0_5
arch: p150
status: DONE_PASS
test_function: test_ko_gpt_trinity_1_2b
samples_per_second: 44.424
ttft_ms: 349.362
prefill_pcc: 0.999223
first_decode_pcc: 0.999445
top_perf_samples_per_sec: 215.1971
pct_of_target: 20.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_ko_gpt_trinity_1_2b

## Test
tests/benchmark/test_llms.py::test_ko_gpt_trinity_1_2b

## Model
- HF name:    skt/ko-gpt-trinity-1.2B-v0.5
- Loader:     third_party.tt_forge_models.ko_gpt_trinity.causal_lm.pytorch.loader
- Variant:    ModelVariant.KO_GPT_TRINITY_1_2B_V0_5

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  44.424
- TTFT (ms):          349.362
- Prefill PCC:        0.999223
- First decode PCC:   0.999445
- Wall clock:         0:15:44
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_ko_gpt_trinity_1_2b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 44.424 / 215.1971 = 20.6%

Note: Gap from roofline is large (~79%). Model runs at optimization_level=2,
trace=True, bfp_bf8 (best settings). Ko-GPT-Trinity is a GPT-2 style
architecture (24 layers, 1.2B params) — small models tend to have higher
host/runtime overhead relative to the roofline.

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
- total_flops:             74252451840
- breakdown.matmul:        6291456000
- breakdown.linear:        67960995840
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        377487360
- memory_bytes: 754974720
- memory_gb:    0.703125

### Params
- count:                  1260860292
- effective_count:        1160590212
- memory_bytes:           1435062284
- memory_gb:              1.336505901068449
- effective_memory_bytes: 1234522124
- effective_memory_gb:    1.149738322943449
- embedding_count:        100270080
- embedding_memory_bytes: 200540160

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 215.1971
- top_perf_time_ms:         4.6469
- dram_time_ms:             3.0979
- compute_time_ms_lofi:     0.0844
- compute_time_ms_hifi2:    0.1688
- compute_time_ms_hifi3:    0.2531
- compute_time_ms_hifi4:    0.3375

## Files changed
- tests/benchmark/test_llms.py (added test_ko_gpt_trinity_1_2b)
- .github/workflows/perf-bench-matrix.json (added ko_gpt_trinity_1_2b entry)
- tests/benchmark/benchmarks/llm_benchmark.py (defensive hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
