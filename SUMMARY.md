loader_path: third_party.tt_forge_models.sure_llama3_1_gguf.causal_lm.pytorch.loader
variant_id: 8B_GGUF
arch: p150
status: DONE_PASS
test_function: test_sure_llama3_1_8b_gguf
samples_per_second: 31.21
ttft_ms: 317.30
prefill_pcc: 0.999255
first_decode_pcc: 0.998442
top_perf_samples_per_sec: 42.5786
pct_of_target: 73.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: sure_llama3_1_8b_gguf

## Test
tests/benchmark/test_llms.py::test_sure_llama3_1_8b_gguf

## Model
- HF name:    mradermacher/SURE-LLaMA3.1-GGUF
- Loader:     third_party.tt_forge_models.sure_llama3_1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.SURE_LLAMA3_1_8B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  31.21
- TTFT (ms):          317.30
- Prefill PCC:        0.999255
- First decode PCC:   0.998442
- Wall clock:         0:09:58
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_sure_llama3_1_8b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 73.3% (31.21 / 42.58)

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
- total_flops:             480314916992
- breakdown.matmul:        480314916992
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
- count:                  8030785731
- effective_count:        7505187011
- memory_bytes:           9025708808
- memory_gb:              8.406
- effective_memory_bytes: 7974511368
- effective_memory_gb:    7.427
- embedding_count:        525598720
- embedding_memory_bytes: 1051197440

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5786
- top_perf_time_ms:         23.4860
- dram_time_ms:             15.6573
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py (added test_sure_llama3_1_8b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
