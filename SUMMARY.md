loader_path: third_party.tt_forge_models.bartowski_thedrummer_anubis_mini_8b_v1_gguf.causal_lm.pytorch.loader
variant_id: anubis_mini_8b_v1_Q4_K_M_GGUF
arch: n150
status: DONE_PASS
test_function: test_bartowski_thedrummer_anubis_mini_8b_v1_gguf
samples_per_second: 17.848596727384365
ttft_ms: 683.063086
prefill_pcc: 0.998021
first_decode_pcc: 0.998124
top_perf_samples_per_sec: 23.9513
pct_of_target: 74.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_bartowski_thedrummer_anubis_mini_8b_v1_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_thedrummer_anubis_mini_8b_v1_gguf

## Model
- HF name:    bartowski/TheDrummer_Anubis-Mini-8B-v1-GGUF
- Loader:     third_party.tt_forge_models.bartowski_thedrummer_anubis_mini_8b_v1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ANUBIS_MINI_8B_V1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  17.848596727384365
- TTFT (ms):          683.063086
- Prefill PCC:        0.998021
- First decode PCC:   0.998124
- Wall clock:         0:18:03
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_thedrummer_anubis_mini_8b_v1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 74.5%

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
- total_flops:             480298401920
- breakdown.matmul:        480298401920
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
- count:                  8030269635
- effective_count:        7504928963
- memory_bytes:           9024918536
- memory_gb:              8.405110366642475
- effective_memory_bytes: 7974237192
- effective_memory_gb:    7.426587112247944
- embedding_count:        525340672
- embedding_memory_bytes: 1050681344

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 23.9513
- top_perf_time_ms:         41.7515
- dram_time_ms:             27.8343
- compute_time_ms_lofi:     1.8762
- compute_time_ms_hifi2:    3.7523
- compute_time_ms_hifi3:    5.6285
- compute_time_ms_hifi4:    7.5047

## Files changed
- tests/benchmark/test_llms.py (added test_bartowski_thedrummer_anubis_mini_8b_v1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path call with hasattr)

## tt-forge-models submodule
no change
