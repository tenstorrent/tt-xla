loader_path: third_party.tt_forge_models.dnd_gguf.causal_lm.pytorch.loader
variant_id: Qwen2.5_0.5B_GGUF
arch: p150
status: DONE_PASS
test_function: test_dnd_gguf_qwen2_5_0_5b
samples_per_second: 146.96
ttft_ms: 129.14
prefill_pcc: 0.982441
first_decode_pcc: 0.987124
top_perf_samples_per_sec: 637.5013
pct_of_target: 23.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_dnd_gguf_qwen2_5_0_5b

## Test
tests/benchmark/test_llms.py::test_dnd_gguf_qwen2_5_0_5b

## Model
- HF name:    elusinchi/dnd-gguf-models (qwen2.5-0.5b/base_q4_k_m.gguf)
- Loader:     third_party.tt_forge_models.dnd_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_2_5_0_5B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  146.96
- TTFT (ms):          129.14
- Prefill PCC:        0.982441
- First decode PCC:   0.987124
- Wall clock:         0:04:14
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_dnd_gguf_qwen2_5_0_5b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 23.1% (146.96 / 637.50)

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
- total_flops:             31614402624
- breakdown.matmul:        30028070976
- breakdown.linear:        1586331648
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        25165824
- memory_bytes: 50331648
- memory_gb:    0.046875

### Params
- count:                  630167587
- effective_count:        494032931
- memory_bytes:           797246856
- memory_gb:              0.7424939945340157
- effective_memory_bytes: 524977544
- effective_memory_gb:    0.48892343789339066
- embedding_count:        136134656
- embedding_memory_bytes: 272269312

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 637.5013
- top_perf_time_ms:         1.5686
- dram_time_ms:             1.0457
- compute_time_ms_lofi:     0.0359
- compute_time_ms_hifi2:    0.0719
- compute_time_ms_hifi3:    0.1078
- compute_time_ms_hifi4:    0.1437

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: defensive hasattr check for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
