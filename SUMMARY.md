loader_path: third_party.tt_forge_models.decs_1_5b_i1_gguf.causal_lm.pytorch.loader
variant_id: i1_Q4_K_M
arch: n150
status: DONE_PASS
test_function: test_decs_1_5b_i1_q4_k_m
samples_per_second: 34.91
ttft_ms: 370.95
prefill_pcc: 0.960293
first_decode_pcc: 0.955600
top_perf_samples_per_sec: 116.1868
pct_of_target: 30.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: test_decs_1_5b_i1_q4_k_m

## Test
tests/benchmark/test_llms.py::test_decs_1_5b_i1_q4_k_m

## Model
- HF name:    mradermacher/DECS_1.5B-i1-GGUF
- Loader:     third_party.tt_forge_models.decs_1_5b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DECS_1_5B_I1_Q4_K_M (value: "i1_Q4_K_M")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: none (bfp_bf8 caused Prefill PCC=0.900 < 0.94 threshold)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  34.91
- TTFT (ms):          370.95
- Prefill PCC:        0.960293
- First decode PCC:   0.955600
- Wall clock:         0:09:49
- Hardware:           n300 (single-chip wormhole_b0 / n150 roofline assumption)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_decs_1_5b_i1_q4_k_m_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 30.0% (34.91 / 116.19)
Note: gap partly explained by experimental_weight_dtype=None (bfp_bf8 disabled due to PCC failure)

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
- total_flops:             98790277248
- breakdown.matmul:        93151297664
- breakdown.linear:        5638979584
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        58720256
- memory_bytes: 117440512
- memory_gb:    0.109375

### Params
- count:                  1777088195
- effective_count:        1543714499
- memory_bytes:           3554176776
- memory_gb:              3.310085065662861
- effective_memory_bytes: 3087429384
- effective_memory_gb:    2.875392682850361
- embedding_count:        233373696
- embedding_memory_bytes: 466747392

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 116.1868
- top_perf_time_ms:         8.6068
- dram_time_ms:             5.7379
- compute_time_ms_lofi:     0.3859
- compute_time_ms_hifi2:    0.7718
- compute_time_ms_hifi3:    1.1577
- compute_time_ms_hifi4:    1.5436

## Files changed
- tests/benchmark/test_llms.py (new test function test_decs_1_5b_i1_q4_k_m)
- tests/benchmark/benchmarks/llm_benchmark.py (infra fix: hasattr guard for get_weight_dtype_config_path; None->empty string for experimental_weight_dtype)
- .github/workflows/perf-bench-matrix.json (added decs_1_5b_i1_q4_k_m entry)

## tt-forge-models submodule
no change
