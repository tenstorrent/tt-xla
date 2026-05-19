loader_path: third_party.tt_forge_models.qwen_2_5_coder_14b_gguf.causal_lm.pytorch.loader
variant_id: unsloth_14B_Instruct_GGUF
arch: p150
status: DONE_PASS
test_function: test_qwen_2_5_coder_14b_instruct_gguf
samples_per_second: 16.594365336212608
ttft_ms: 530.257389
prefill_pcc: 0.999216
first_decode_pcc: 0.999030
top_perf_samples_per_sec: 22.9948
pct_of_target: 72.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: qwen_2_5_coder_14b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_qwen_2_5_coder_14b_instruct_gguf

## Model
- HF name:    unsloth/Qwen2.5-Coder-14B-Instruct-GGUF
- Loader:     third_party.tt_forge_models.qwen_2_5_coder_14b_gguf.causal_lm.pytorch.loader
- Variant:    UNSLOTH_QWEN_2_5_CODER_14B_INSTRUCT_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  16.594365336212608
- TTFT (ms):          530.257389
- Prefill PCC:        0.999216
- First decode PCC:   0.999030
- Wall clock:         0:15:46
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_qwen_2_5_coder_14b_instruct_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 72.2%

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
- total_flops:             895411028096
- breakdown.matmul:        782657126528
- breakdown.linear:        112753901568
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        402653184
- memory_bytes: 805306368
- memory_gb:    0.75

### Params
- count:                  14770033859
- effective_count:        13991466179
- memory_bytes:           16423856904
- memory_gb:              15.295908696949482
- effective_memory_bytes: 14866721544
- effective_memory_gb:    13.845713384449482
- embedding_count:        778567680
- embedding_memory_bytes: 1557135360

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.9948
- top_perf_time_ms:         43.4881
- dram_time_ms:             28.9921
- compute_time_ms_lofi:     1.0175
- compute_time_ms_hifi2:    2.0350
- compute_time_ms_hifi3:    3.0525
- compute_time_ms_hifi4:    4.0701

## Files changed
- tests/benchmark/test_llms.py (added test_qwen_2_5_coder_14b_instruct_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added qwen_2_5_coder_14b_instruct_gguf entry with runs-on: p150)

## tt-forge-models submodule
no change
