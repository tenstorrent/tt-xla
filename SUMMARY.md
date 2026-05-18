loader_path: third_party.tt_forge_models.aisingapore_llama_sea_lion_v3_5_8b_r_gguf.causal_lm.pytorch.loader
variant_id: Llama_SEA_LION_v3_5_8B_R_GGUF
arch: p150
status: DONE_PASS
test_function: test_sea_lion_v3_5_8b_r_gguf
samples_per_second: 33.825375901704966
ttft_ms: 308.101075
prefill_pcc: 0.998302
first_decode_pcc: 0.998486
top_perf_samples_per_sec: 42.58
pct_of_target: 79.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_sea_lion_v3_5_8b_r_gguf

## Test
tests/benchmark/test_llms.py::test_sea_lion_v3_5_8b_r_gguf

## Model
- HF name:    aisingapore/Llama-SEA-LION-v3.5-8B-R-GGUF
- Loader:     third_party.tt_forge_models.aisingapore_llama_sea_lion_v3_5_8b_r_gguf.causal_lm.pytorch.loader
- Variant:    LLAMA_SEA_LION_V3_5_8B_R_GGUF ("Llama_SEA_LION_v3_5_8B_R_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.83
- TTFT (ms):          308.1
- Prefill PCC:        0.998302
- First decode PCC:   0.998486
- Wall clock:         0:09:34
- Hardware:           p150 (blackhole P300/1-chip, qb2-21-forge-benchmark-perf-0)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_sea_lion_v3_5_8b_r_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 33.83 / 42.58 = 79.4%

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
- memory_gb:              8.41
- effective_memory_bytes: 7974232840
- effective_memory_gb:    7.43
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.58
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py (added test_sea_lion_v3_5_8b_r_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: getattr for get_weight_dtype_config_path — handles loaders missing this optional method)
- .github/workflows/perf-bench-matrix.json (added sea_lion_v3_5_8b_r_gguf entry)

## tt-forge-models submodule
no change

## Notes
- Requires TT_MESH_GRAPH_DESC_PATH to point to a valid fabric mesh graph descriptor (e.g.
  third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto)
  on this machine (qb2-21-forge-benchmark-perf-0) which has a P300 BlackHole board detected as
  CUSTOM cluster type. The p150 mesh descriptor (1x1 BlackHole mesh) works correctly.
- Requires gguf>=0.10.0 Python package for loading the GGUF checkpoint format.
