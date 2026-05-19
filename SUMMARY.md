loader_path: third_party.tt_forge_models.exaone4.causal_lm.pytorch.loader
variant_id: 4.0_1.2B
arch: p150
status: DONE_PASS
test_function: test_exaone4_1_2b
samples_per_second: 73.93
ttft_ms: 172.40
prefill_pcc: 0.987985
first_decode_pcc: 0.992771
top_perf_samples_per_sec: 235.5430
pct_of_target: 31.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_exaone4_1_2b

## Test
tests/benchmark/test_llms.py::test_exaone4_1_2b

## Model
- HF name:    LGAI-EXAONE/EXAONE-4.0-1.2B
- Loader:     third_party.tt_forge_models.exaone4.causal_lm.pytorch.loader
- Variant:    ModelVariant.EXAONE_4_0_1_2B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  73.93
- TTFT (ms):          172.40
- Prefill PCC:        0.987985
- First decode PCC:   0.992771
- Wall clock:         0:05:13
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_exaone4_1_2b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 31.4% (73.93 / 235.54)

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
- total_flops:             81872814144
- breakdown.matmul:        81872814144
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        125829120
- memory_bytes: 251658240
- memory_gb:    0.234375

### Params
- count:                  1489106851
- effective_count:        1279391651
- memory_bytes:           1778905224
- memory_gb:              1.6567345932126045
- effective_memory_bytes: 1359474824
- effective_memory_gb:    1.2661095932126045
- embedding_count:        209715200
- embedding_memory_bytes: 419430400

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 235.5430
- top_perf_time_ms:         4.2455
- dram_time_ms:             2.8303
- compute_time_ms_lofi:     0.0930
- compute_time_ms_hifi2:    0.1861
- compute_time_ms_hifi3:    0.2791
- compute_time_ms_hifi4:    0.3721

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
