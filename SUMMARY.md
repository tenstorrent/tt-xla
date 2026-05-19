loader_path: third_party.tt_forge_models.lexora_lite_3b_v2.causal_lm.pytorch.loader
variant_id: 3B_v2
arch: p150
status: DONE_PASS
test_function: test_lexora_lite_3b_v2
samples_per_second: 41.66
ttft_ms: 223.78
prefill_pcc: 0.993413
first_decode_pcc: 0.999504
top_perf_samples_per_sec: 693.5057
pct_of_target: 6.0
roofline_bound: compute
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_lexora_lite_3b_v2

## Test
tests/benchmark/test_llms.py::test_lexora_lite_3b_v2

## Model
- HF name:    DeepMount00/Lexora-Lite-3B_v2
- Loader:     third_party.tt_forge_models.lexora_lite_3b_v2.causal_lm.pytorch.loader
- Variant:    LEXORA_LITE_3B_V2 (3B_v2)

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 (default) fails with ttnn.paged_update_cache validation error
("Expect input_tensor to be sharded"). optimization_level=1 passes cleanly.

## Measured (full model, defaults)
- Sample per second:  41.66
- TTFT (ms):          223.78
- Prefill PCC:        0.993413
- First decode PCC:   0.999504
- Wall clock:         0:02:51
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_lexora_lite_3b_v2_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 6.0% (41.66 / 693.51)

Note: Low utilisation is expected at optimization_level=1 (tensors remain in DRAM; SRAM
placement via optimization_level=2 is blocked by a compiler bug in ttnn.paged_update_cache).

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
- total_flops:             422971787392
- breakdown.matmul:        417266141312
- breakdown.linear:        5705646080
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        2097152
- memory_bytes: 4194304
- memory_gb:    0.00390625

### Params
- count:                  699409094
- effective_count:        388244166
- memory_bytes:           1034848020
- memory_gb:              0.9637773223221302
- effective_memory_bytes: 412518164
- effective_memory_gb:    0.3841874785721302
- embedding_count:        311164928
- embedding_memory_bytes: 622329856

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 693.5057
- top_perf_time_ms:         1.4419
- dram_time_ms:             0.7862
- compute_time_ms_lofi:     0.4806
- compute_time_ms_hifi2:    0.9613
- compute_time_ms_hifi3:    1.4419
- compute_time_ms_hifi4:    1.9226

## Files changed
- tests/benchmark/test_llms.py (added test_lexora_lite_3b_v2)
- tests/benchmark/benchmarks/llm_benchmark.py (harness fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
