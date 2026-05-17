loader_path: third_party.tt_forge_models.communicative_baby_dpo.causal_lm.pytorch.loader
variant_id: communicative_baby_dpo
arch: n150
status: DONE_PASS
test_function: test_communicative_baby_dpo
samples_per_second: 104.99
ttft_ms: 152.536
prefill_pcc: 0.998960
first_decode_pcc: 0.991579
top_perf_samples_per_sec: 715.0872
pct_of_target: 14.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_communicative_baby_dpo

## Test
tests/benchmark/test_llms.py::test_communicative_baby_dpo

## Model
- HF name:    CLAUSE-Bielefeld/communicative-baby-dpo
- Loader:     third_party.tt_forge_models.communicative_baby_dpo.causal_lm.pytorch.loader
- Variant:    communicative_baby_dpo

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  104.99
- TTFT (ms):          152.536
- Prefill PCC:        0.998960
- First decode PCC:   0.991579
- Wall clock:         0:03:56
- Hardware:           n300 (wormhole_b0, single_chip_assumption=True)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_communicative_baby_dpo_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 14.7% (104.99 / 715.09)

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
- total_flops:             8071086144
- breakdown.matmul:        8071086144
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        134217728
- memory_bytes: 268435456
- memory_gb:    0.25

### Params
- count:                  134814883
- effective_count:        126144675
- memory_bytes:           151401288
- memory_gb:              0.14100343734025955
- effective_memory_bytes: 134060872
- effective_memory_gb:    0.12485391646623611
- embedding_count:        8670208
- embedding_memory_bytes: 17340416

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 715.0872
- top_perf_time_ms:         1.3984
- dram_time_ms:             0.9323
- compute_time_ms_lofi:     0.0315
- compute_time_ms_hifi2:    0.0631
- compute_time_ms_hifi3:    0.0946
- compute_time_ms_hifi4:    0.1261

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general harness fix: guard get_weight_dtype_config_path with hasattr check)

## tt-forge-models submodule
no change
