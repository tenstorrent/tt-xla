loader_path: third_party.tt_forge_models.communicative_baby_rfolmo_score.causal_lm.pytorch.loader
variant_id: communicative_baby_rfolmo_score
arch: n150
status: DONE_PASS
test_function: test_communicative_baby_rfolmo_score
samples_per_second: 211.25
ttft_ms: 86.32
prefill_pcc: 0.998900
first_decode_pcc: 0.997092
top_perf_samples_per_sec: 1271.2661
pct_of_target: 16.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_communicative_baby_rfolmo_score

## Test
tests/benchmark/test_llms.py::test_communicative_baby_rfolmo_score

## Model
- HF name:    CLAUSE-Bielefeld/communicative-baby-rfolmo_score
- Loader:     third_party.tt_forge_models.communicative_baby_rfolmo_score.causal_lm.pytorch.loader
- Variant:    ModelVariant.COMMUNICATIVE_BABY_RFOLMO_SCORE

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  211.25
- TTFT (ms):          86.32
- Prefill PCC:        0.998900
- First decode PCC:   0.997092
- Wall clock:         0:01:55
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_communicative_baby_rfolmo_score_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 16.6% (211.25 / 1271.27)

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
- bound:                    dram
- top_perf_samples_per_sec: 1271.2661
- top_perf_time_ms:         0.7866
- dram_time_ms:             0.5244
- compute_time_ms_lofi:     0.1651
- compute_time_ms_hifi2:    0.3302
- compute_time_ms_hifi3:    0.4953
- compute_time_ms_hifi4:    0.6604

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path call with hasattr)

## tt-forge-models submodule
no change
