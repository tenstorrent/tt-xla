loader_path: third_party.tt_forge_models.llasa.causal_lm.pytorch.loader
variant_id: 1B
arch: p150
status: DONE_FAIL
test_function: test_llasa_1b
samples_per_second: 108.88
ttft_ms: 106.16
prefill_pcc: 0.951264
first_decode_pcc: 0.932954
top_perf_samples_per_sec: 230.3087
pct_of_target: 47.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "first decode PCC 0.932954 below required 0.94 with bfp_bf8+opt=2+trace=True; tried opt=1 (prefill PCC 0.859), no bfp (prefill PCC 0.844), fp32_dest_acc_en=True (decode PCC 0.915), trace=False (decode PCC 0.934) - all configurations fail decode PCC threshold"

# Benchmark added: test_llasa_1b

## Test
tests/benchmark/test_llms.py::test_llasa_1b

## Model
- HF name:    HKUSTAudio/Llasa-1B
- Loader:     third_party.tt_forge_models.llasa.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLASA_1B (value: "1B")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  108.88
- TTFT (ms):          106.16
- Prefill PCC:        0.951264 (PASS)
- First decode PCC:   0.932954 (FAIL — required 0.94)
- Wall clock:         0:03:05
- Hardware:           p150

## PCC exploration summary
All configurations fail the 0.94 first decode PCC threshold:
| Config                                    | Prefill PCC | Decode PCC |
|-------------------------------------------|-------------|------------|
| opt=2, bfp_bf8, trace=True  (baseline)    | 0.951264 ✓  | 0.932954 ✗ |
| opt=2, bfp_bf8, fp32_dest_acc_en=True     | 0.953375 ✓  | 0.915364 ✗ |
| opt=2, bfp_bf8, trace=False               | 0.951264 ✓  | 0.933710 ✗ |
| opt=1, bfp_bf8, trace=True                | 0.859398 ✗  | N/A        |
| opt=2, no bfp (empty string), trace=True  | 0.843797 ✗  | N/A        |

Best decode PCC achieved: 0.933710 (trace=False). Required: 0.94. Gap: ~0.007.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llasa_1b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 47.3% (108.88 / 230.31)

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
- total_flops:             87678779456
- breakdown.matmul:        87678779456
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        67108864
- memory_bytes: 134217728
- memory_gb:    0.125

### Params
- count:                  1766950944
- effective_count:        1370048544
- memory_bytes:           3533901888
- memory_gb:              3.2912026047706604
- effective_memory_bytes: 2740097088
- effective_memory_gb:    2.5519142746925354
- embedding_count:        396902400
- embedding_memory_bytes: 793804800

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 230.3087
- top_perf_time_ms:         4.3420
- dram_time_ms:             2.8947
- compute_time_ms_lofi:     0.0996
- compute_time_ms_hifi2:    0.1993
- compute_time_ms_hifi3:    0.2989
- compute_time_ms_hifi4:    0.3985

## Files changed
- tests/benchmark/test_llms.py (added test_llasa_1b)
- tests/benchmark/benchmarks/llm_benchmark.py (guarded get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added llasa_1b entry)

## tt-forge-models submodule
no change
