loader_path: third_party.tt_forge_models.acestep.causal_lm.pytorch.loader
variant_id: acestep_5hz_lm_4b
arch: n150
status: DONE_PASS
test_function: test_acestep_5hz_lm_4b
samples_per_second: 16.871
ttft_ms: 709.55
prefill_pcc: 0.959160
first_decode_pcc: 0.991157
top_perf_samples_per_sec: 41.4516
pct_of_target: 40.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_acestep_5hz_lm_4b

## Test
tests/benchmark/test_llms.py::test_acestep_5hz_lm_4b

## Model
- HF name:    ACE-Step/acestep-5Hz-lm-4B
- Loader:     third_party.tt_forge_models.acestep.causal_lm.pytorch.loader
- Variant:    ModelVariant.ACESTEP_5HZ_LM_4B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  16.871
- TTFT (ms):          709.55
- Prefill PCC:        0.959160
- First decode PCC:   0.991157
- Wall clock:         ~0:15:00
- Hardware:           n150 (n300 single-chip mode)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_acestep_5hz_lm_4b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 40.7% (16.871 / 41.4516)

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
- total_flops:             268118917248
- breakdown.matmul:        268118917248
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  4745596611
- effective_count:        4189554371
- memory_bytes:           5563670408
- memory_gb:              5.182
- effective_memory_bytes: 4451585928
- effective_memory_gb:    4.146
- embedding_count:        556042240
- embedding_memory_bytes: 1112084480

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 41.4516
- top_perf_time_ms:         24.1245
- dram_time_ms:             16.0830
- compute_time_ms_lofi:     1.0473
- compute_time_ms_hifi2:    2.0947
- compute_time_ms_hifi3:    3.1420
- compute_time_ms_hifi4:    4.1894

## Files changed
- tests/benchmark/test_llms.py (added test_acestep_5hz_lm_4b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added acestep_5hz_lm_4b entry)

## tt-forge-models submodule
no change
