loader_path: third_party.tt_forge_models.kraken_karcher_12b_v1_i1_gguf.causal_lm.pytorch.loader
variant_id: KRAKEN_KARCHER_12B_V1_I1_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_kraken_karcher_12b
samples_per_second: 21.054296834549405
ttft_ms: 469.655477
prefill_pcc: 0.994485
first_decode_pcc: 0.975726
top_perf_samples_per_sec: 27.7856
pct_of_target: 75.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_kraken_karcher_12b

## Test
tests/benchmark/test_llms.py::test_kraken_karcher_12b

## Model
- HF name:    mradermacher/Kraken-Karcher-12B-v1-i1-GGUF
- Loader:     third_party.tt_forge_models.kraken_karcher_12b_v1_i1_gguf.causal_lm.pytorch.loader
- Variant:    KRAKEN_KARCHER_12B_V1_I1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  21.054296834549405
- TTFT (ms):          469.655477
- Prefill PCC:        0.994485
- First decode PCC:   0.975726
- Wall clock:         0:12:40
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_kraken_karcher_12b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 75.8%

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
- total_flops:             740883497088
- breakdown.matmul:        740883497088
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        335544320
- memory_bytes: 671088640
- memory_gb:    0.625

### Params
- count:                  12247833795
- effective_count:        11576719555
- memory_bytes:           13642882376
- memory_gb:              12.705924339592457
- effective_memory_bytes: 12300653896
- effective_memory_gb:    11.455876655876637
- embedding_count:        671114240
- embedding_memory_bytes: 1342228480

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 27.7856
- top_perf_time_ms:         35.9898
- dram_time_ms:             23.9932
- compute_time_ms_lofi:     0.8419
- compute_time_ms_hifi2:    1.6838
- compute_time_ms_hifi3:    2.5257
- compute_time_ms_hifi4:    3.3677

## Files changed
- tests/benchmark/test_llms.py (new test_kraken_karcher_12b function)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
