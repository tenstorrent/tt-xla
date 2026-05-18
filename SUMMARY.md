loader_path: third_party.tt_forge_models.baguettotron.causal_lm.pytorch.loader
variant_id: baguettotron
arch: p150
status: DONE_PASS
test_function: test_baguettotron
samples_per_second: 68.18344914682676
ttft_ms: 372.260282
prefill_pcc: 0.994629
first_decode_pcc: 0.989972
top_perf_samples_per_sec: 740.8238
pct_of_target: 9.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_baguettotron

## Test
tests/benchmark/test_llms.py::test_baguettotron

## Model
- HF name:    PleIAs/Baguettotron
- Loader:     third_party.tt_forge_models.baguettotron.causal_lm.pytorch.loader
- Variant:    ModelVariant.BAGUETTOTRON

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  68.18344914682676
- TTFT (ms):          372.260282
- Prefill PCC:        0.994629
- First decode PCC:   0.989972
- Wall clock:         0:12:56
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_baguettotron_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 9.2% (68.18 / 740.82)

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
- total_flops:             20535312448
- breakdown.matmul:        20535312448
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
- count:                  358705891
- effective_count:        320957155
- memory_bytes:           416601864
- memory_gb:              0.38799072057008743
- effective_memory_bytes: 341104392
- effective_memory_gb:    0.31767822057008743
- embedding_count:        37748736
- embedding_memory_bytes: 75497472

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 740.8238
- top_perf_time_ms:         1.3498
- dram_time_ms:             0.8999
- compute_time_ms_lofi:     0.0233
- compute_time_ms_hifi2:    0.0467
- compute_time_ms_hifi3:    0.0700
- compute_time_ms_hifi4:    0.0933

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
