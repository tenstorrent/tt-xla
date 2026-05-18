loader_path: third_party.tt_forge_models.gpt2_fa.causal_lm.pytorch.loader
variant_id: Default
arch: p150
status: DONE_PASS
test_function: test_gpt2_fa
samples_per_second: 167.25
ttft_ms: 83.38
prefill_pcc: 0.999362
first_decode_pcc: 0.998879
top_perf_samples_per_sec: 1716.6602
pct_of_target: 9.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_gpt2_fa

## Test
tests/benchmark/test_llms.py::test_gpt2_fa

## Model
- HF name:    HooshvareLab/gpt2-fa
- Loader:     third_party.tt_forge_models.gpt2_fa.causal_lm.pytorch.loader
- Variant:    ModelVariant.BASE (value="Default")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  167.25
- TTFT (ms):          83.38
- Prefill PCC:        0.999362
- First decode PCC:   0.998879
- Wall clock:         0:02:04
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gpt2_fa_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 9.7% (167.25 / 1716.66)

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
- total_flops:             7502905344
- breakdown.matmul:        2064433152
- breakdown.linear:        5438472192
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  150356100
- effective_count:        117312900
- memory_bytes:           191011388
- memory_gb:              0.17789321765303612
- effective_memory_bytes: 124924988
- effective_memory_gb:    0.11634546145796776
- embedding_count:        33043200
- embedding_memory_bytes: 66086400

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1716.6602
- top_perf_time_ms:         0.5825
- dram_time_ms:             0.3884
- compute_time_ms_lofi:     0.0085
- compute_time_ms_hifi2:    0.0171
- compute_time_ms_hifi3:    0.0256
- compute_time_ms_hifi4:    0.0341

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
