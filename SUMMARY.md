loader_path: third_party.tt_forge_models.granite_4_0_micro_gguf.causal_lm.pytorch.loader
variant_id: Granite_4.0_Micro_Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_granite_4_0_micro_gguf
samples_per_second: 51.592367660014354
ttft_ms: 265.599877
prefill_pcc: 0.997621
first_decode_pcc: 0.998156
top_perf_samples_per_sec: 92.6984
pct_of_target: 55.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: granite_4_0_micro_gguf

## Test
tests/benchmark/test_llms.py::test_granite_4_0_micro_gguf

## Model
- HF name:    lmstudio-community/granite-4.0-micro-GGUF
- Loader:     third_party.tt_forge_models.granite_4_0_micro_gguf.causal_lm.pytorch.loader
- Variant:    Granite_4.0_Micro_Q4_K_M

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  51.592367660014354
- TTFT (ms):          265.599877
- Prefill PCC:        0.997621
- First decode PCC:   0.998156
- Wall clock:         0:08:33
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_granite_4_0_micro_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 55.7%

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
- total_flops:             217768263744
- breakdown.matmul:        217768263744
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        167772160
- memory_bytes: 335544320
- memory_gb:    0.3125

### Params
- count:                  3659737766
- effective_count:        3402836646
- memory_bytes:           4129511054
- memory_gb:              3.8459068667143583
- effective_memory_bytes: 3615708814
- effective_memory_gb:    3.3673912417143583
- embedding_count:        256901120
- embedding_memory_bytes: 513802240

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 92.6984
- top_perf_time_ms:         10.7877
- dram_time_ms:             7.1918
- compute_time_ms_lofi:     0.2475
- compute_time_ms_hifi2:    0.4949
- compute_time_ms_hifi3:    0.7424
- compute_time_ms_hifi4:    0.9899

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
