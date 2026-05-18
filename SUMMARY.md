loader_path: third_party.tt_forge_models.communicative_baby_dpo_synthetic.causal_lm.pytorch.loader
variant_id: Communicative_Baby_DPO_Synthetic
arch: p150
status: DONE_PASS
test_function: test_communicative_baby_dpo_synthetic
samples_per_second: 203.89496625790838
ttft_ms: 86.70444
prefill_pcc: 0.997760
first_decode_pcc: 0.989937
top_perf_samples_per_sec: 1271.2661
pct_of_target: 16.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_communicative_baby_dpo_synthetic

## Test
tests/benchmark/test_llms.py::test_communicative_baby_dpo_synthetic

## Model
- HF name:    CLAUSE-Bielefeld/communicative-baby-dpo-synthetic
- Loader:     third_party.tt_forge_models.communicative_baby_dpo_synthetic.causal_lm.pytorch.loader
- Variant:    COMMUNICATIVE_BABY_DPO_SYNTHETIC

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  203.89496625790838
- TTFT (ms):          86.70444
- Prefill PCC:        0.997760
- First decode PCC:   0.989937
- Wall clock:         0:01:55
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_communicative_baby_dpo_synthetic_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 16.0%

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
- memory_gb:              0.14100343734025955
- effective_memory_bytes: 134060872
- effective_memory_gb:    0.12485391646623611
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
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: add hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
