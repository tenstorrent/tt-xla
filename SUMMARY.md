loader_path: third_party.tt_forge_models.aiqarus_agent_4b_i1_gguf.causal_lm.pytorch.loader
variant_id: 4B_i1_GGUF
arch: n150
status: DONE_PASS
test_function: test_aiqarus_agent_4b_i1_gguf
samples_per_second: 17.33
ttft_ms: 687.06
prefill_pcc: 0.998216
first_decode_pcc: 0.997056
top_perf_samples_per_sec: 43.0532
pct_of_target: 40.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_aiqarus_agent_4b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_aiqarus_agent_4b_i1_gguf

## Model
- HF name:    mradermacher/aiqarus-agent-4b-i1-GGUF
- Loader:     third_party.tt_forge_models.aiqarus_agent_4b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.AIQARUS_AGENT_4B_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  17.33
- TTFT (ms):          687.06
- Prefill PCC:        0.998216
- First decode PCC:   0.997056
- Wall clock:         0:16:17
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_aiqarus_agent_4b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 40.3%

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

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
- top_perf_samples_per_sec: 43.0532
- top_perf_time_ms:         23.2271
- dram_time_ms:             15.4847
- compute_time_ms_lofi:     1.0056
- compute_time_ms_hifi2:    2.0111
- compute_time_ms_hifi3:    3.0167
- compute_time_ms_hifi4:    4.0223

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: add hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
