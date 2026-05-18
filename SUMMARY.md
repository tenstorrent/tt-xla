loader_path: third_party.tt_forge_models.abacusai.causal_lm.pytorch.loader
variant_id: bigstral_12B_32K
arch: p150
status: DONE_PASS
test_function: test_abacusai_bigstral_12b_32k
samples_per_second: 20.19
ttft_ms: 567.01
prefill_pcc: 0.999542
first_decode_pcc: 0.998431
top_perf_samples_per_sec: 25.8281
pct_of_target: 78.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_abacusai_bigstral_12b_32k

## Test
tests/benchmark/test_llms.py::test_abacusai_bigstral_12b_32k

## Model
- HF name:    abacusai/bigstral-12b-32k
- Loader:     third_party.tt_forge_models.abacusai.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIGSTRAL_12B_32K

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  20.19
- TTFT (ms):          567.01
- Prefill PCC:        0.999542
- First decode PCC:   0.998431
- Wall clock:         0:16:37
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_abacusai_bigstral_12b_32k_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 78.2% (20.19 / 25.83)

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
- total_flops:             790072656000
- breakdown.matmul:        790072656000
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        469762048
- memory_bytes: 939524096
- memory_gb:    0.875

### Params
- count:                  12476420291
- effective_count:        12345348291
- memory_bytes:           13379511048
- memory_gb:              12.4606406763196
- effective_memory_bytes: 13117367048
- effective_memory_gb:    12.2165000513196
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 25.8281
- top_perf_time_ms:         38.7175
- dram_time_ms:             25.8117
- compute_time_ms_lofi:     0.8978
- compute_time_ms_hifi2:    1.7956
- compute_time_ms_hifi3:    2.6934
- compute_time_ms_hifi4:    3.5912

## Files changed
- tests/benchmark/test_llms.py (new test function test_abacusai_bigstral_12b_32k)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
