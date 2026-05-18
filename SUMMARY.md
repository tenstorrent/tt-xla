loader_path: third_party.tt_forge_models.fallen_amoral_gemma3_gguf.causal_lm.pytorch.loader
variant_id: 12B_GGUF
arch: p150
status: DONE_PASS
test_function: test_fallen_amoral_gemma3_12b_gguf
samples_per_second: 13.713324233942746
ttft_ms: 956.028582
prefill_pcc: 0.995692
first_decode_pcc: 0.995765
top_perf_samples_per_sec: 26.3289
pct_of_target: 52.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_fallen_amoral_gemma3_12b_gguf

## Test
tests/benchmark/test_llms.py::test_fallen_amoral_gemma3_12b_gguf

## Model
- HF name:    mradermacher/Fallen_Amoral_Gemma3-12B-i1-GGUF
- Loader:     third_party.tt_forge_models.fallen_amoral_gemma3_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.FALLEN_AMORAL_GEMMA3_12B_GGUF (12B_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8" (default from test_llm)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  13.713324233942746
- TTFT (ms):          956.028582
- Prefill PCC:        0.995692
- First decode PCC:   0.995765
- Wall clock:         ~0:30:00
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_fallen_amoral_gemma3_12b_gguf_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 52.1% (13.71 / 26.33)

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
- total_flops:             752977182976
- breakdown.matmul:        752977182976
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
- count:                  12772913157
- effective_count:        11766034437
- memory_bytes:           14517419022
- memory_gb:              13.520400060340762
- effective_memory_bytes: 12503661582
- effective_memory_gb:    11.644942296668887
- embedding_count:        1006878720
- embedding_memory_bytes: 2013757440

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 26.3289
- top_perf_time_ms:         37.9810
- dram_time_ms:             25.3207
- compute_time_ms_lofi:     0.8557
- compute_time_ms_hifi2:    1.7113
- compute_time_ms_hifi3:    2.5670
- compute_time_ms_hifi4:    3.4226

## Files changed
- tests/benchmark/test_llms.py (added test_fallen_amoral_gemma3_12b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: added hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
