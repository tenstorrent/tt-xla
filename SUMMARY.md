loader_path: third_party.tt_forge_models.communicative_baby.causal_lm.pytorch.loader
variant_id: rfolmo_score
arch: n150
status: DONE_PASS
test_function: test_communicative_baby_rfolmo_score
samples_per_second: 106.47
ttft_ms: 148.06
prefill_pcc: 0.999247
first_decode_pcc: 0.996789
top_perf_samples_per_sec: 587.3733
pct_of_target: 18.1
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_communicative_baby_rfolmo_score

## Test
tests/benchmark/test_llms.py::test_communicative_baby_rfolmo_score

## Model
- HF name:    CLAUSE-Bielefeld/communicative-baby-rfolmo_score
- Loader:     third_party.tt_forge_models.communicative_baby.causal_lm.pytorch.loader
- Variant:    ModelVariant.RFOLMO_SCORE

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  106.47
- TTFT (ms):          148.06
- Prefill PCC:        0.999247
- First decode PCC:   0.996789
- Wall clock:         0:04:04
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_communicative_baby_rfolmo_score_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 18.1% (106.47 / 587.37)

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
- total_flops:             145279550592
- breakdown.matmul:        145279550592
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        594
- memory_bytes: 2376

### KV cache
- count:        134217728
- memory_bytes: 268435456
- memory_gb:    0.25

### Params
- count:                  134814883
- effective_count:        126144675
- memory_bytes:           151401288
- memory_gb:              0.141
- effective_memory_bytes: 134060872
- effective_memory_gb:    0.125
- embedding_count:        8670208
- embedding_memory_bytes: 17340416

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 587.3733
- top_perf_time_ms:         1.7025
- dram_time_ms:             0.9323
- compute_time_ms_lofi:     0.5675
- compute_time_ms_hifi2:    1.1350
- compute_time_ms_hifi3:    1.7025
- compute_time_ms_hifi4:    2.2700

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
