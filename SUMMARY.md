loader_path: third_party.tt_forge_models.clean_take99_v1_i1_gguf.causal_lm.pytorch.loader
variant_id: v1_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_clean_take99_v1_i1_gguf
samples_per_second: 4.067688428008911
ttft_ms: 911.020181
prefill_pcc: 0.997649
first_decode_pcc: 0.998359
top_perf_samples_per_sec: 76.5390
pct_of_target: 5.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_clean_take99_v1_i1_gguf

## Test
tests/benchmark/test_llms.py::test_clean_take99_v1_i1_gguf

## Model
- HF name:    mradermacher/CleanTake99_v1-i1-GGUF
- Loader:     third_party.tt_forge_models.clean_take99_v1_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.CLEAN_TAKE99_V1_I1_GGUF (v1_i1_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  4.067688428008911
- TTFT (ms):          911.020181
- Prefill PCC:        0.997649
- First decode PCC:   0.998359
- Wall clock:         ~0:23:00
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_clean_take99_v1_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 5.3% (4.07 / 76.54)

Note: Performance is 5.3% of roofline with all aggressive settings enabled
(optimization_level=2, trace=True, bfp_bf8). This is DRAM-bound; the gap
may reflect suboptimal kernel selection or pipeline stalls on this model
architecture. No further test-level knobs are available.

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
- total_flops:             259841327232
- breakdown.matmul:        259841327232
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
- count:                  4411424454
- effective_count:        4022468294
- memory_bytes:           5051969300
- memory_gb:              4.705013055354357
- effective_memory_bytes: 4274056980
- effective_memory_gb:    3.9805257506668568
- embedding_count:        388956160
- embedding_memory_bytes: 777912320

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 76.5390
- top_perf_time_ms:         13.0652
- dram_time_ms:             8.7102
- compute_time_ms_lofi:     0.2498
- compute_time_ms_hifi2:    0.4997
- compute_time_ms_hifi3:    0.7495
- compute_time_ms_hifi4:    0.9994

## Files changed
- tests/benchmark/test_llms.py (added test_clean_take99_v1_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
