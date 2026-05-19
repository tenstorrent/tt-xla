loader_path: third_party.tt_forge_models.ministral_3b_reasoning_gguf.causal_lm.pytorch.loader
variant_id: 3B_Reasoning_GGUF
arch: p150
status: DONE_PASS
test_function: test_ministral_3b_reasoning_gguf
samples_per_second: 41.1
ttft_ms: 239.58
prefill_pcc: 0.998470
first_decode_pcc: 0.999042
top_perf_samples_per_sec: 90.7540
pct_of_target: 45.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: ministral_3b_reasoning_gguf

## Test
tests/benchmark/test_llms.py::test_ministral_3b_reasoning_gguf

## Model
- HF name:    unsloth/Ministral-3-3B-Reasoning-2512
- Loader:     third_party.tt_forge_models.ministral_3b_reasoning_gguf.causal_lm.pytorch.loader
- Variant:    3B_Reasoning_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  41.10
- TTFT (ms):          239.58
- Prefill PCC:        0.998470
- First decode PCC:   0.999042
- Wall clock:         0:11:55
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_ministral_3b_reasoning_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 45.3%

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
- total_flops:             219445985408
- breakdown.matmul:        219445985408
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        218103808
- memory_bytes: 436207616
- memory_gb:    0.40625

### Params
- count:                  3831659718
- effective_count:        3429006534
- memory_bytes:           4448779028
- memory_gb:              4.143248338252306
- effective_memory_bytes: 3643472660
- effective_memory_gb:    3.393248338252306
- embedding_count:        402653184
- embedding_memory_bytes: 805306368

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 90.7540
- top_perf_time_ms:         11.0188
- dram_time_ms:             7.3459
- compute_time_ms_lofi:     0.2494
- compute_time_ms_hifi2:    0.4987
- compute_time_ms_hifi3:    0.7481
- compute_time_ms_hifi4:    0.9975

## Files changed
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
