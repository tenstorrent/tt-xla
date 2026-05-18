loader_path: third_party.tt_forge_models.huggyllama_13b.causal_lm.pytorch.loader
variant_id: huggyllama_llama_13b
arch: p150
status: DONE_PASS
test_function: test_huggyllama_llama_13b
samples_per_second: 9.962265767834479
ttft_ms: 698.956138
prefill_pcc: 0.998694
first_decode_pcc: 0.990187
top_perf_samples_per_sec: 22.7802
pct_of_target: 43.7
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: test_huggyllama_llama_13b

## Test
tests/benchmark/test_llms.py::test_huggyllama_llama_13b

## Model
- HF name:    huggyllama/llama-13b
- Loader:     third_party.tt_forge_models.huggyllama_13b.causal_lm.pytorch.loader
- Variant:    ModelVariant.HUGGYLLAMA_LLAMA_13B

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: none (bfp_bf8 causes decode PCC regression below 0.94)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  9.962265767834479
- TTFT (ms):          698.956138
- Prefill PCC:        0.998694
- First decode PCC:   0.990187
- Wall clock:         0:04:01
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_huggyllama_llama_13b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 43.7% (9.96 / 22.78)

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
- total_flops:             822503014528
- breakdown.matmul:        822503014528
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1677721600
- memory_bytes: 3355443200
- memory_gb:    3.125

### Params
- count:                  13015864515
- effective_count:        12852024515
- memory_bytes:           26031729416
- memory_gb:              24.243937261402607
- effective_memory_bytes: 25704049416
- effective_memory_gb:    23.938761480152607
- embedding_count:        163840000
- embedding_memory_bytes: 327680000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.7802
- top_perf_time_ms:         43.8979
- dram_time_ms:             29.2652
- compute_time_ms_lofi:     0.9347
- compute_time_ms_hifi2:    1.8693
- compute_time_ms_hifi3:    2.8040
- compute_time_ms_hifi4:    3.7387

## Files changed
- tests/benchmark/test_llms.py (added test_huggyllama_llama_13b)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
