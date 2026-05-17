loader_path: third_party.tt_forge_models.corianas_256_5epoch.causal_lm.pytorch.loader
variant_id: corianas_256_5epoch
arch: n150
status: DONE_PASS
test_function: test_corianas_256_5epoch
samples_per_second: 54.759
ttft_ms: 242.188
prefill_pcc: 0.997986
first_decode_pcc: 0.999005
top_perf_samples_per_sec: 309.2511
pct_of_target: 17.7
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_corianas_256_5epoch

## Test
tests/benchmark/test_llms.py::test_corianas_256_5epoch

## Model
- HF name:    Corianas/256_5epoch
- Loader:     third_party.tt_forge_models.corianas_256_5epoch.causal_lm.pytorch.loader
- Variant:    CORIANAS_256_5EPOCH

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  54.759
- TTFT (ms):          242.188
- Prefill PCC:        0.997986
- First decode PCC:   0.999005
- Wall clock:         0:04:59
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_corianas_256_5epoch_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 17.7%

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             275935461376
- breakdown.matmul:        59491422208
- breakdown.linear:        216444039168
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        124780544
- memory_bytes: 249561088
- memory_gb:    0.232421875

### Params
- count:                  310656772
- effective_count:        253748932
- memory_bytes:           383886160
- memory_gb:              0.3575218468904495
- effective_memory_bytes: 270070480
- effective_memory_gb:    0.2515227347612381
- embedding_count:        56907840
- embedding_memory_bytes: 113815680

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 309.2511
- top_perf_time_ms:         3.2336
- dram_time_ms:             1.3554
- compute_time_ms_lofi:     1.0779
- compute_time_ms_hifi2:    2.1557
- compute_time_ms_hifi3:    3.2336
- compute_time_ms_hifi4:    4.3115

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
