loader_path: third_party.tt_forge_models.anakin87_llama_3_8b_ita_slerp.causal_lm.pytorch.loader
variant_id: Llama_3_8B_ita_slerp
arch: n150
status: DONE_PASS
test_function: test_anakin87_llama_3_8b_ita_slerp
samples_per_second: 17.696658893211367
ttft_ms: 661.910893
prefill_pcc: 0.998147
first_decode_pcc: 0.998518
top_perf_samples_per_sec: 23.9513
pct_of_target: 73.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_anakin87_llama_3_8b_ita_slerp

## Test
tests/benchmark/test_llms.py::test_anakin87_llama_3_8b_ita_slerp

## Model
- HF name:    anakin87/Llama-3-8b-ita-slerp
- Loader:     third_party.tt_forge_models.anakin87_llama_3_8b_ita_slerp.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_3_8B_ITA_SLERP

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  17.696658893211367
- TTFT (ms):          661.910893
- Prefill PCC:        0.998147
- First decode PCC:   0.998518
- Wall clock:         0:17:26
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_anakin87_llama_3_8b_ita_slerp_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 73.9% (17.70 / 23.95)

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
- total_flops:             480298139776
- breakdown.matmul:        480298139776
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  8030261443
- effective_count:        7504924867
- memory_bytes:           9024905992
- memory_gb:              8.4050986841321
- effective_memory_bytes: 7974232840
- effective_memory_gb:    7.426583059132099
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 23.9513
- top_perf_time_ms:         41.7514
- dram_time_ms:             27.8343
- compute_time_ms_lofi:     1.8762
- compute_time_ms_hifi2:    3.7523
- compute_time_ms_hifi3:    5.6285
- compute_time_ms_hifi4:    7.5047

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
