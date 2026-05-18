loader_path: third_party.tt_forge_models.dmind_3_mini_i1_gguf.causal_lm.pytorch.loader
variant_id: 3_mini_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_dmind_3_mini_i1_gguf
samples_per_second: 46.317
ttft_ms: 254.594
prefill_pcc: 0.998204
first_decode_pcc: 0.998051
top_perf_samples_per_sec: 95.8156
pct_of_target: 48.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_dmind_3_mini_i1_gguf

## Test
tests/benchmark/test_llms.py::test_dmind_3_mini_i1_gguf

## Model
- HF name:    mradermacher/DMind-3-mini-i1-GGUF
- Loader:     third_party.tt_forge_models.dmind_3_mini_i1_gguf.causal_lm.pytorch.loader
- Variant:    DMIND_3_MINI_I1_GGUF (3_mini_i1_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  46.317
- TTFT (ms):          254.594
- Prefill PCC:        0.998204
- First decode PCC:   0.998051
- Wall clock:         0:08:31
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_dmind_3_mini_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 48.3% (46.317 / 95.8156)

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
- total_flops:             212483440768
- breakdown.matmul:        212483440768
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        134217728
- memory_bytes: 268435456
- memory_gb:    0.25

### Params
- count:                  3955927747
- effective_count:        3320228547
- memory_bytes:           4799305480
- memory_gb:              4.469701535999775
- effective_memory_bytes: 3527907080
- effective_memory_gb:    3.285619504749775
- embedding_count:        635699200
- embedding_memory_bytes: 1271398400

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 95.8156
- top_perf_time_ms:         10.4367
- dram_time_ms:             6.9578
- compute_time_ms_lofi:     0.2415
- compute_time_ms_hifi2:    0.4829
- compute_time_ms_hifi3:    0.7244
- compute_time_ms_hifi4:    0.9658

## Files changed
- tests/benchmark/test_llms.py (new test function test_dmind_3_mini_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change (submodule at 82b75428c1b0c9489bc2df43de7b5f6b2761bfbb)
