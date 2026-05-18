loader_path: third_party.tt_forge_models.ai_girlfriend_v2_gguf.causal_lm.pytorch.loader
variant_id: 8B_GGUF
arch: p150
status: DONE_PASS
test_function: test_ai_girlfriend_v2_gguf_8b
samples_per_second: 32.36168198643852
ttft_ms: 312.104625
prefill_pcc: 0.998901
first_decode_pcc: 0.998529
top_perf_samples_per_sec: 42.5800
pct_of_target: 76.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_ai_girlfriend_v2_gguf_8b

## Test
tests/benchmark/test_llms.py::test_ai_girlfriend_v2_gguf_8b

## Model
- HF name:    mradermacher/ai-girlfriend-v2-i1-GGUF
- Loader:     third_party.tt_forge_models.ai_girlfriend_v2_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.AI_GIRLFRIEND_V2_8B_GGUF (8B_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  32.36168198643852
- TTFT (ms):          312.104625
- Prefill PCC:        0.998901
- First decode PCC:   0.998529
- Wall clock:         0:09:33
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_ai_girlfriend_v2_gguf_8b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 76.0% (32.36 / 42.58)

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
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (restored hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
