loader_path: third_party.tt_forge_models.cat_translate_7b_i1_gguf.causal_lm.pytorch.loader
variant_id: CAT_Translate_7b_i1_GGUF
arch: n150
status: DONE_PASS
test_function: test_cat_translate_7b_i1_gguf
samples_per_second: 18.414728419429
ttft_ms: 653.310322
prefill_pcc: 0.999117
first_decode_pcc: 0.998974
top_perf_samples_per_sec: 24.8209
pct_of_target: 74.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_cat_translate_7b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_cat_translate_7b_i1_gguf

## Model
- HF name:    mradermacher/CAT-Translate-7b-i1-GGUF
- Loader:     third_party.tt_forge_models.cat_translate_7b_i1_gguf.causal_lm.pytorch.loader
- Variant:    CAT_TRANSLATE_7B_I1_GGUF = "CAT_Translate_7b_i1_GGUF"

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  18.414728419429
- TTFT (ms):          653.310322
- Prefill PCC:        0.999117
- First decode PCC:   0.998974
- Wall clock:         0:18:39
- Hardware:           n150 (wormhole_b0, n300 single-chip)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_cat_translate_7b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 74.2% (18.41 / 24.82)

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
- total_flops:             462867923072
- breakdown.matmul:        462867923072
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
- count:                  7485567171
- effective_count:        7232577731
- memory_bytes:           8190842888
- memory_gb:              7.6283168867230415
- effective_memory_bytes: 7684864008
- effective_memory_gb:    7.157087333500385
- embedding_count:        252989440
- embedding_memory_bytes: 505978880

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 24.8209
- top_perf_time_ms:         40.2886
- dram_time_ms:             26.8591
- compute_time_ms_lofi:     1.8081
- compute_time_ms_hifi2:    3.6162
- compute_time_ms_hifi3:    5.4242
- compute_time_ms_hifi4:    7.2323

## Files changed
- tests/benchmark/test_llms.py (added test_cat_translate_7b_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added cat_translate_7b_i1_gguf entry)

## tt-forge-models submodule
no change
