loader_path: third_party.tt_forge_models.bartowski_phi_3_mini_4k_instruct_gguf.causal_lm.pytorch.loader
variant_id: Phi_3_Mini_4K_Instruct_GGUF
arch: p150
status: DONE_PASS
test_function: test_bartowski_phi_3_mini_4k_instruct_gguf
samples_per_second: 6.368722190347544
ttft_ms: 1071.871506
prefill_pcc: 0.997841
first_decode_pcc: 0.996243
top_perf_samples_per_sec: 73.1003
pct_of_target: 8.7
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
- Sample per second:  6.368722190347544
- TTFT (ms):          1071.871506
- Prefill PCC:        0.997841
- First decode PCC:   0.996243
- Wall clock:         0:42:00
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_phi_3_mini_4k_instruct_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 8.7% (6.37 / 73.10)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             239842885728
- breakdown.matmul:        239842885728
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
- count:                  3821079734
- effective_count:        3722579126
- memory_bytes:           4152429268
- memory_gb:              3.867251116782427
- effective_memory_bytes: 3955428052
- effective_memory_gb:    3.683779437094927
- embedding_count:        98500608
- embedding_memory_bytes: 197001216

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 73.1003
- top_perf_time_ms:         13.6798
- dram_time_ms:             9.1199
- compute_time_ms_lofi:     0.2306
- compute_time_ms_hifi2:    0.4612
- compute_time_ms_hifi3:    0.6919
- compute_time_ms_hifi4:    0.9225

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
