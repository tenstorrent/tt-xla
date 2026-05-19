loader_path: third_party.tt_forge_models.duckllm_1_0_0_6b_gguf.causal_lm.pytorch.loader
variant_id: 1.0_0.6B_GGUF
arch: p150
status: DONE_PASS
test_function: test_duckllm_1_0_0_6b_gguf
samples_per_second: 122.63
ttft_ms: 239.975609
prefill_pcc: 0.971219
first_decode_pcc: 0.973061
top_perf_samples_per_sec: 637.5013
pct_of_target: 19.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_duckllm_1_0_0_6b_gguf

## Test
tests/benchmark/test_llms.py::test_duckllm_1_0_0_6b_gguf

## Model
- HF name:    DuckLLM/DuckLLM-1.0-0.6B-GGUF
- Loader:     third_party.tt_forge_models.duckllm_1_0_0_6b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DUCKLLM_1_0_0_6B_GGUF (1.0_0.6B_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  122.63
- TTFT (ms):          239.975609
- Prefill PCC:        0.971219
- First decode PCC:   0.973061
- Wall clock:         0:09:29
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_duckllm_1_0_0_6b_gguf_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 19.2% (122.63 / 637.50)

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
- total_flops:             537444844608
- breakdown.matmul:        510477206592
- breakdown.linear:        26967638016
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

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
- compute_time_ms_lofi:     0.5168
- compute_time_ms_hifi2:    1.0335
- compute_time_ms_hifi3:    1.5503
- compute_time_ms_hifi4:    2.0671

## Files changed
- tests/benchmark/test_llms.py (added test_duckllm_1_0_0_6b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (add hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added duckllm_1_0_0_6b_gguf entry)

## tt-forge-models submodule
no change
