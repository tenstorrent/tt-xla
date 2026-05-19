loader_path: third_party.tt_forge_models.josie_4b_thinking_i1_gguf.causal_lm.pytorch.loader
variant_id: 4B_THINKING_I1_GGUF
arch: p150
status: DONE_PASS
test_function: test_josie_4b_thinking_i1_gguf
samples_per_second: 34.998727770791916
ttft_ms: 322.345353
prefill_pcc: 0.997150
first_decode_pcc: 0.997660
top_perf_samples_per_sec: 76.5390
pct_of_target: 45.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_josie_4b_thinking_i1_gguf

## Test
tests/benchmark/test_llms.py::test_josie_4b_thinking_i1_gguf

## Model
- HF name:    mradermacher/JOSIE-4B-Thinking-i1-GGUF
- Loader:     third_party.tt_forge_models.josie_4b_thinking_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.JOSIE_4B_THINKING_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  34.998727770791916
- TTFT (ms):          322.345353
- Prefill PCC:        0.997150
- First decode PCC:   0.997660
- Wall clock:         0:08:39
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_josie_4b_thinking_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 45.7% (34.99 / 76.54 samples/sec)

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
- total_flops:             257425408128
- breakdown.matmul:        257425408128
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  4411424451
- effective_count:        4022468291
- memory_bytes:           5051969288
- memory_gb:              4.705013044178486
- effective_memory_bytes: 4274056968
- effective_memory_gb:    3.980525739490986
- embedding_count:        388956160
- embedding_memory_bytes: 777912320

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 76.5390
- top_perf_time_ms:         13.0652
- dram_time_ms:             8.7102
- compute_time_ms_lofi:     0.2925
- compute_time_ms_hifi2:    0.5851
- compute_time_ms_hifi3:    0.8776
- compute_time_ms_hifi4:    1.1701

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: added hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
