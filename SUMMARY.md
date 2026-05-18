loader_path: third_party.tt_forge_models.dolphin_2_9_llama3_8b_gguf.causal_lm.pytorch.loader
variant_id: Dolphin_2.9_Llama3_8B_GGUF
arch: p150
status: DONE_PASS
test_function: test_dolphin_2_9_llama3_8b_gguf
samples_per_second: 32.80108832927738
ttft_ms: 311.675332
prefill_pcc: 0.998654
first_decode_pcc: 0.996835
top_perf_samples_per_sec: 42.58
pct_of_target: 77.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_dolphin_2_9_llama3_8b_gguf

## Test
tests/benchmark/test_llms.py::test_dolphin_2_9_llama3_8b_gguf

## Model
- HF name:    QuantFactory/dolphin-2.9-llama3-8b-GGUF
- Loader:     third_party.tt_forge_models.dolphin_2_9_llama3_8b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DOLPHIN_2_9_LLAMA3_8B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  32.80108832927738
- TTFT (ms):          311.675332
- Prefill PCC:        0.998654
- First decode PCC:   0.996835
- Wall clock:         0:09:35
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_dolphin_2_9_llama3_8b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 77.0% (32.80 / 42.58)

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
- total_flops:             480298664064
- breakdown.matmul:        480298664064
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
- count:                  8030277827
- effective_count:        7504933059
- memory_bytes:           9024931080
- memory_gb:              8.405122049152851
- effective_memory_bytes: 7974241544
- effective_memory_gb:    7.426591165363789
- embedding_count:        525344768
- embedding_memory_bytes: 1050689536

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.58
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py (added test_dolphin_2_9_llama3_8b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (defensive getattr for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added dolphin_2_9_llama3_8b_gguf entry)

## tt-forge-models submodule
no change
