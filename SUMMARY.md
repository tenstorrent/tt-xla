loader_path: third_party.tt_forge_models.tulu_13b_fp16.causal_lm.pytorch.loader
variant_id: 13B_FP16
arch: p150
status: DONE_PASS
test_function: test_tulu_13b_fp16
samples_per_second: 12.786775
ttft_ms: 624.170423
prefill_pcc: 0.997941
first_decode_pcc: 0.992774
top_perf_samples_per_sec: 22.7801
pct_of_target: 56.1
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_tulu_13b_fp16

## Test
tests/benchmark/test_llms.py::test_tulu_13b_fp16

## Model
- HF name:    TheBloke/tulu-13B-fp16
- Loader:     third_party.tt_forge_models.tulu_13b_fp16.causal_lm.pytorch.loader
- Variant:    ModelVariant.TULU_13B_FP16 = "13B_FP16"

## Test config landed
- optimization_level:        1 (optimization_level=2 fails PCC on p150 for this model)
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  12.786775
- TTFT (ms):          624.170423
- Prefill PCC:        0.997941
- First decode PCC:   0.992774
- Wall clock:         0:11:41
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_tulu_13b_fp16_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 56.1% (12.787 / 22.780)

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
- total_flops:             822503342208
- breakdown.matmul:        822503342208
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
- count:                  13015874755
- effective_count:        12852029635
- memory_bytes:           13983361096
- memory_gb:              13.02301985770464
- effective_memory_bytes: 13655670856
- effective_memory_gb:    12.717834539711475
- embedding_count:        163845120
- embedding_memory_bytes: 327690240

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.7801
- top_perf_time_ms:         43.8979
- dram_time_ms:             29.2652
- compute_time_ms_lofi:     0.9347
- compute_time_ms_hifi2:    1.8693
- compute_time_ms_hifi3:    2.8040
- compute_time_ms_hifi4:    3.7387

## Files changed
- tests/benchmark/test_llms.py (added test_tulu_13b_fp16)
- .github/workflows/perf-bench-matrix.json (added tulu_13b_fp16 entry)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path to use hasattr)

## tt-forge-models submodule
Checked out to c9b45c4dfe71 (arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-29) — adds tulu_13b_fp16/causal_lm/pytorch/loader.py
