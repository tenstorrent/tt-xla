loader_path: third_party.tt_forge_models.characterbuilderai_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: n150
status: DONE_PASS
test_function: test_characterbuilderai_gguf_q4_k_m
samples_per_second: 18.873602552148792
ttft_ms: 652.702459
prefill_pcc: 0.998570
first_decode_pcc: 0.997978
top_perf_samples_per_sec: 25.2202
pct_of_target: 74.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_characterbuilderai_gguf_q4_k_m

## Test
tests/benchmark/test_llms.py::test_characterbuilderai_gguf_q4_k_m

## Model
- HF name:    Idrinth/characterbuilderai-gguf
- Loader:     third_party.tt_forge_models.characterbuilderai_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.CHARACTERBUILDERAI_Q4_K_M (= "Q4_K_M")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  18.873602552148792
- TTFT (ms):          652.702459
- Prefill PCC:        0.998570
- First decode PCC:   0.997978
- Wall clock:         0:17:39
- Hardware:           n150 (wormhole_b0, single chip from n300)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_characterbuilderai_gguf_q4_k_m_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 74.8% (18.87 / 25.22)

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
- total_flops:             455266533504
- breakdown.matmul:        455266533504
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
- count:                  7248023747
- effective_count:        7113806019
- memory_bytes:           7827104520
- memory_gb:              7.289559133350849
- effective_memory_bytes: 7558669064
- effective_memory_gb:    7.039559133350849
- embedding_count:        134217728
- embedding_memory_bytes: 268435456

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 25.2202
- top_perf_time_ms:         39.6507
- dram_time_ms:             26.4338
- compute_time_ms_lofi:     1.7784
- compute_time_ms_hifi2:    3.5568
- compute_time_ms_hifi3:    5.3352
- compute_time_ms_hifi4:    7.1135

## Files changed
- tests/benchmark/test_llms.py (added test_characterbuilderai_gguf_q4_k_m)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed: add hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added characterbuilderai_gguf_q4_k_m entry)

## tt-forge-models submodule
no change
