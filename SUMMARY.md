loader_path: third_party.tt_forge_models.bartowski_llama_3_1_nemotron_nano_8b_v1_gguf.causal_lm.pytorch.loader
variant_id: 3.1_Nemotron_Nano_8B_v1_GGUF_Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_bartowski_llama_3_1_nemotron_nano_8b_v1_gguf
samples_per_second: 25.35
ttft_ms: 353.19
prefill_pcc: 0.978079
first_decode_pcc: 0.966399
top_perf_samples_per_sec: 42.58
pct_of_target: 59.5
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_bartowski_llama_3_1_nemotron_nano_8b_v1_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_llama_3_1_nemotron_nano_8b_v1_gguf

## Model
- HF name:    bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF
- Loader:     third_party.tt_forge_models.bartowski_llama_3_1_nemotron_nano_8b_v1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BARTOWSKI_LLAMA_3_1_NEMOTRON_NANO_8B_V1_GGUF_Q4_K_M

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Notes
- optimization_level=2 causes Prefill PCC=0.924 (below 0.94 threshold) due to bfp_bf8
  quantization error accumulating across 32 dequantized-GGUF layers. optimization_level=1
  gives Prefill PCC=0.978.
- experimental_weight_dtype=None (bf16) causes OOM (ENOMEM/error code 12) on p150 even
  with num_layers=1, confirming bfp_bf8 is required for this model to fit on device.
- Infrastructure fix applied: llm_benchmark.py now uses getattr() for get_weight_dtype_config_path
  so GGUF loaders without this method do not raise AttributeError.

## Measured (full model, defaults)
- Sample per second:  25.35
- TTFT (ms):          353.19
- Prefill PCC:        0.978079
- First decode PCC:   0.966399
- Wall clock:         0:04:44
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_llama_3_1_nemotron_nano_8b_v1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 59.5% (25.35 / 42.58)

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
- top_perf_samples_per_sec: 42.5800
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py (new test function)
- tests/benchmark/benchmarks/llm_benchmark.py (getattr fix for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (new entry)

## tt-forge-models submodule
no change
