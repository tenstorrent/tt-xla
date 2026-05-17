loader_path: third_party.tt_forge_models.asplos2026.causal_lm.pytorch.loader
variant_id: Qwen1.5_0.5B_Q4_0
arch: n150
status: DONE_PASS
test_function: test_asplos2026_qwen1_5_0_5b_q4_0
samples_per_second: 60.236
ttft_ms: 314.863
prefill_pcc: 0.9969
first_decode_pcc: 0.9951
top_perf_samples_per_sec: 169.0693
pct_of_target: 35.6
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_asplos2026_qwen1_5_0_5b_q4_0

## Test
tests/benchmark/test_llms.py::test_asplos2026_qwen1_5_0_5b_q4_0

## Model
- HF name:    chanwoocho/asplos2026
- Loader:     third_party.tt_forge_models.asplos2026.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN1_5_0_5B_Q4_0

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  60.236
- TTFT (ms):          314.863
- Prefill PCC:        0.9969
- First decode PCC:   0.9951
- Wall clock:         0:07:29
- Hardware:           n150 (wormhole_b0, n300 single-chip assumption)

## Decode roofline (first decode graph, single-chip)
Source JSON: tests/benchmark/tt_xla_asplos2026_qwen1_5_0_5b_q4_0_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 35.6% (60.236 / 169.069)

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
- total_flops:             504723932224
- breakdown.matmul:        422542574656
- breakdown.linear:        82181357568
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        201326592
- memory_bytes: 402653184
- memory_gb:    0.375

### Params
- count:                  619570339
- effective_count:        463987875
- memory_bytes:           804268680
- memory_gb:              0.7490335777401924
- effective_memory_bytes: 493103752
- effective_memory_gb:    0.4592386558651924
- embedding_count:        155582464
- embedding_memory_bytes: 311164928

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 169.0693
- top_perf_time_ms:         5.9147
- dram_time_ms:             2.3823
- compute_time_ms_lofi:     1.9716
- compute_time_ms_hifi2:    3.9432
- compute_time_ms_hifi3:    5.9147
- compute_time_ms_hifi4:    7.8863

## Files changed
- tests/benchmark/test_llms.py (added test_asplos2026_qwen1_5_0_5b_q4_0)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
