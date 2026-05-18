loader_path: third_party.tt_forge_models.babylm_interaction_baseline_simpo.causal_lm.pytorch.loader
variant_id: base
arch: p150
status: DONE_PASS
test_function: test_babylm_interaction_baseline_simpo
samples_per_second: 171.785
ttft_ms: 82.004
prefill_pcc: 0.999631
first_decode_pcc: 0.999738
top_perf_samples_per_sec: 1911.7278
pct_of_target: 9.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_babylm_interaction_baseline_simpo

## Test
tests/benchmark/test_llms.py::test_babylm_interaction_baseline_simpo

## Model
- HF name:    BabyLM-community/babylm-interaction-baseline-simpo
- Loader:     third_party.tt_forge_models.babylm_interaction_baseline_simpo.causal_lm.pytorch.loader
- Variant:    base

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  171.785
- TTFT (ms):          82.004
- Prefill PCC:        0.999631
- First decode PCC:   0.999738
- Wall clock:         0:02:04
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_babylm_interaction_baseline_simpo_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 9.0%

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
- total_flops:             118631792640
- breakdown.matmul:        15300820992
- breakdown.linear:        103330971648
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        627
- memory_bytes: 2508

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  111008388
- effective_count:        97639044
- memory_bytes:           130760204
- memory_gb:              0.12177992984652519
- effective_memory_bytes: 104021516
- effective_memory_gb:    0.09687758609652519
- embedding_count:        13369344
- embedding_memory_bytes: 26738688

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1911.7278
- top_perf_time_ms:         0.5231
- dram_time_ms:             0.3487
- compute_time_ms_lofi:     0.1348
- compute_time_ms_hifi2:    0.2696
- compute_time_ms_hifi3:    0.4044
- compute_time_ms_hifi4:    0.5392

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
