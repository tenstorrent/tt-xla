loader_path: third_party.tt_forge_models.starcoderbase_1b.causal_lm.pytorch.loader
variant_id: 1B
arch: p150
status: DONE_PASS
test_function: test_starcoderbase_1b
samples_per_second: 65.583
ttft_ms: 146.751
prefill_pcc: 0.997768
first_decode_pcc: 0.998185
top_perf_samples_per_sec: 288.9237
pct_of_target: 22.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: starcoderbase_1b

## Test
tests/benchmark/test_llms.py::test_starcoderbase_1b

## Model
- HF name:    bigcode/starcoderbase-1b
- Loader:     third_party.tt_forge_models.starcoderbase_1b.causal_lm.pytorch.loader
- Variant:    ModelVariant.STARCODERBASE_1B (1B)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  65.583
- TTFT (ms):          146.751
- Prefill PCC:        0.997768
- First decode PCC:   0.998185
- Wall clock:         0:04:28
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_starcoderbase_1b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 22.7% (65.58 / 288.92)

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
- total_flops:             71683473408
- breakdown.matmul:        6442450944
- breakdown.linear:        65241022464
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        25165824
- memory_bytes: 50331648
- memory_gb:    0.046875

### Params
- count:                  1237870724
- effective_count:        1120430212
- memory_bytes:           1425854988
- memory_gb:              1.3279309384524822
- effective_memory_bytes: 1190973964
- effective_memory_gb:    1.1091809384524822
- embedding_count:        117440512
- embedding_memory_bytes: 234881024

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 288.9237
- top_perf_time_ms:         3.4611
- dram_time_ms:             2.3074
- compute_time_ms_lofi:     0.0815
- compute_time_ms_hifi2:    0.1629
- compute_time_ms_hifi3:    0.2444
- compute_time_ms_hifi4:    0.3258

## Files changed
- tests/benchmark/test_llms.py (added test_starcoderbase_1b)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added bigcode_starcoderbase-1b entry)

## tt-forge-models submodule
no change
