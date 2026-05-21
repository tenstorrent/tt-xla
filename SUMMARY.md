loader_path: third_party.tt_forge_models.deepseek_llama3_1_hari_8b_i1_gguf.causal_lm.pytorch.loader
variant_id: 8B_HARI_I1_GGUF
arch: p150
status: DONE_PASS
test_function: test_deepseek_llama3_1_hari_8b_i1_gguf
samples_per_second: 33.75019001163455
ttft_ms: 305.925133
prefill_pcc: 0.997987
first_decode_pcc: 0.991528
top_perf_samples_per_sec: 42.5800
pct_of_target: 79.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_deepseek_llama3_1_hari_8b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_deepseek_llama3_1_hari_8b_i1_gguf

## Model
- HF name:    mradermacher/DeepSeek-llama3.1-HARI-8B-i1-GGUF
- Loader:     third_party.tt_forge_models.deepseek_llama3_1_hari_8b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPSEEK_LLAMA3_1_HARI_8B_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.75019001163455
- TTFT (ms):          305.925133
- Prefill PCC:        0.997987
- First decode PCC:   0.991528
- Wall clock:         0:09:11
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_llama3_1_hari_8b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 79.3%

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
- top_perf_samples_per_sec: 42.5800
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py (added test_deepseek_llama3_1_hari_8b_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing get_weight_dtype_config_path guard)

## tt-forge-models submodule
no change
