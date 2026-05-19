loader_path: third_party.tt_forge_models.pygmalion_6b.causal_lm.pytorch.loader
variant_id: 6B
arch: p150
status: DONE_PASS
test_function: test_pygmalion_6b
samples_per_second: 7.414752910520223
ttft_ms: 1248.454124
prefill_pcc: 0.991849
first_decode_pcc: 0.999293
top_perf_samples_per_sec: 48.7636
pct_of_target: 15.2
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_pygmalion_6b

## Test
tests/benchmark/test_llms.py::test_pygmalion_6b

## Model
- HF name:    PygmalionAI/pygmalion-6b
- Loader:     third_party.tt_forge_models.pygmalion_6b.causal_lm.pytorch.loader
- Variant:    ModelVariant.PYGMALION_6B (6B)

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 fails with compiler error: Physical shard shape (17, 32) must be tile {32, 32} sized. This is a GPTJ rotary attention shard alignment limitation in the TT-MLIR compiler.

## Measured (full model, defaults)
- Sample per second:  7.414752910520223
- TTFT (ms):          1248.454124
- Prefill PCC:        0.991849
- First decode PCC:   0.999293
- Wall clock:         0:03:52
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_pygmalion_6b_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 15.2% (7.41 / 48.76)

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
- total_flops:             374009273344
- breakdown.matmul:        120259084288
- breakdown.linear:        253750189056
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        939524096
- memory_bytes: 1879048192
- memory_gb:    1.75

### Params
- count:                  6054552931
- effective_count:        5848114531
- memory_bytes:           6630747080
- memory_gb:              6.175364442169666
- effective_memory_bytes: 6217870280
- effective_memory_gb:    5.790842957794666
- embedding_count:        206438400
- embedding_memory_bytes: 412876800

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 48.7636
- top_perf_time_ms:         20.5071
- dram_time_ms:             13.6714
- compute_time_ms_lofi:     0.4250
- compute_time_ms_hifi2:    0.8500
- compute_time_ms_hifi3:    1.2750
- compute_time_ms_hifi4:    1.7000

## Files changed
- tests/benchmark/test_llms.py (added test_pygmalion_6b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: use hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
