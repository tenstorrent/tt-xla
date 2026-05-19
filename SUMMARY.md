loader_path: third_party.tt_forge_models.sarashina2_2.causal_lm.pytorch.loader
variant_id: sarashina2.2_3b_instruct_v0.1
arch: p150
status: DONE_PASS
test_function: test_sarashina2_2_3b_instruct
samples_per_second: 31.35
ttft_ms: 366.71
prefill_pcc: 0.996647
first_decode_pcc: 0.992714
top_perf_samples_per_sec: 96.5264
pct_of_target: 32.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: sarashina2_2_3b_instruct

## Test
tests/benchmark/test_llms.py::test_sarashina2_2_3b_instruct

## Model
- HF name:    sbintuitions/sarashina2.2-3b-instruct-v0.1
- Loader:     third_party.tt_forge_models.sarashina2_2.causal_lm.pytorch.loader
- Variant:    ModelVariant.SARASHINA2_2_3B_INSTRUCT_V0_1

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  31.35
- TTFT (ms):          366.71
- Prefill PCC:        0.996647
- First decode PCC:   0.992714
- Wall clock:         0:08:55
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_sarashina2_2_3b_instruct_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 32.5% (31.35 / 96.5264)

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
- total_flops:             197971148960
- breakdown.matmul:        197971148960
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
- count:                  3355609811
- effective_count:        3093465811
- memory_bytes:           3811252040
- memory_gb:              3.5495050624012947
- effective_memory_bytes: 3286964040
- effective_memory_gb:    3.0612238124012947
- embedding_count:        262144000
- embedding_memory_bytes: 524288000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 96.5264
- top_perf_time_ms:         10.3599
- dram_time_ms:             6.9066
- compute_time_ms_lofi:     0.2250
- compute_time_ms_hifi2:    0.4499
- compute_time_ms_hifi3:    0.6749
- compute_time_ms_hifi4:    0.8999

## Files changed
- tests/benchmark/test_llms.py (added test_sarashina2_2_3b_instruct)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path to use hasattr guard)
- .github/workflows/perf-bench-matrix.json (added sarashina2_2_3b_instruct entry)

## tt-forge-models submodule
no change
