loader_path: third_party.tt_forge_models.bartowski_phi_3_mini_4k_instruct_gguf.causal_lm.pytorch.loader
variant_id: Phi_3_Mini_4K_Instruct_GGUF
arch: n150
status: DONE_PASS
test_function: test_bartowski_phi_3_mini_4k_instruct_gguf
samples_per_second: 8.682243956346337
ttft_ms: 1153.51142
prefill_pcc: 0.997856
first_decode_pcc: 0.997175
top_perf_samples_per_sec: 41.118932564812035
pct_of_target: 21.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_bartowski_phi_3_mini_4k_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_phi_3_mini_4k_instruct_gguf

## Model
- HF name:    bartowski/Phi-3-mini-4k-instruct-GGUF
- Loader:     third_party.tt_forge_models.bartowski_phi_3_mini_4k_instruct_gguf.causal_lm.pytorch.loader
- Variant:    Phi_3_Mini_4K_Instruct_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  8.682243956346337
- TTFT (ms):          1153.51142
- Prefill PCC:        0.997856
- First decode PCC:   0.997175
- Wall clock:         0:21:21
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_phi_3_mini_4k_instruct_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 21.1% (8.68 / 41.12)

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
- total_flops:             238232272992
- breakdown.matmul:        238232272992
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        805306368
- memory_bytes: 1610612736
- memory_gb:    1.5

### Params
- count:                  3821079731
- effective_count:        3722579123
- memory_bytes:           4152429256
- memory_gb:              3.867251105606556
- effective_memory_bytes: 3955428040
- effective_memory_gb:    3.683779425919056
- embedding_count:        98500608
- embedding_memory_bytes: 197001216

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 41.118932564812035
- top_perf_time_ms:         24.319697463541665
- dram_time_ms:             16.21313164236111
- compute_time_ms_lofi:     0.930594816375
- compute_time_ms_hifi2:    1.86118963275
- compute_time_ms_hifi3:    2.791784449125011
- compute_time_ms_hifi4:    3.7223792655

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
