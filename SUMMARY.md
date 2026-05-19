loader_path: third_party.tt_forge_models.dolphin3_llama3_2_1b_bartowski_gguf.causal_lm.pytorch.loader
variant_id: 1B_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_dolphin3_llama3_2_1b_bartowski_gguf
samples_per_second: 19.39
ttft_ms: 251.33
prefill_pcc: 0.998804
first_decode_pcc: 0.998234
top_perf_samples_per_sec: 254.04
pct_of_target: 7.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_dolphin3_llama3_2_1b_bartowski_gguf

## Test
tests/benchmark/test_llms.py::test_dolphin3_llama3_2_1b_bartowski_gguf

## Model
- HF name:    bartowski/Dolphin3.0-Llama3.2-1B-GGUF
- Loader:     third_party.tt_forge_models.dolphin3_llama3_2_1b_bartowski_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DOLPHIN3_0_LLAMA3_2_1B_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  19.39
- TTFT (ms):          251.33
- Prefill PCC:        0.998804
- First decode PCC:   0.998234
- Wall clock:         0:18:31
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_dolphin3_llama3_2_1b_bartowski_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 7.6% (19.39 / 254.04)

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
- total_flops:             79624929344
- breakdown.matmul:        79624929344
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
- count:                  1498491046
- effective_count:        1235818662
- memory_bytes:           1838465940
- memory_gb:              1.7122048325836658
- effective_memory_bytes: 1313121172
- effective_memory_gb:    1.2229393906891346
- embedding_count:        262672384
- embedding_memory_bytes: 525344768

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 254.04
- top_perf_time_ms:         3.9365
- dram_time_ms:             2.6243
- compute_time_ms_lofi:     0.0766
- compute_time_ms_hifi2:    0.1531
- compute_time_ms_hifi3:    0.2297
- compute_time_ms_hifi4:    0.3062

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
