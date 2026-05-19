loader_path: third_party.tt_forge_models.mradermacher_cactus_dream_horror_12b_i1_gguf.causal_lm.pytorch.loader
variant_id: 12B_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_mradermacher_cactus_dream_horror_12b_i1_gguf
samples_per_second: 21.02
ttft_ms: 469.19
prefill_pcc: 0.997315
first_decode_pcc: 0.995429
top_perf_samples_per_sec: 27.7857
pct_of_target: 75.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_mradermacher_cactus_dream_horror_12b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_mradermacher_cactus_dream_horror_12b_i1_gguf

## Model
- HF name:    mradermacher/Cactus-Dream-Horror-12B-i1-GGUF
- Loader:     third_party.tt_forge_models.mradermacher_cactus_dream_horror_12b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.MRADERMACHER_CACTUS_DREAM_HORROR_12B_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  21.02
- TTFT (ms):          469.19
- Prefill PCC:        0.997315
- First decode PCC:   0.995429
- Wall clock:         0:12:54
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mradermacher_cactus_dream_horror_12b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 75.6% (21.02 / 27.79)

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
- total_flops:             740883169408
- breakdown.matmul:        740883169408
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        335544320
- memory_bytes: 671088640
- memory_gb:    0.625

### Params
- count:                  12247823555
- effective_count:        11576714435
- memory_bytes:           13642866696
- memory_gb:              12.705909736454487
- effective_memory_bytes: 12300648456
- effective_memory_gb:    11.45587158948183
- embedding_count:        671109120
- embedding_memory_bytes: 1342218240

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 27.7857
- top_perf_time_ms:         35.9898
- dram_time_ms:             23.9932
- compute_time_ms_lofi:     0.8419
- compute_time_ms_hifi2:    1.6838
- compute_time_ms_hifi3:    2.5257
- compute_time_ms_hifi4:    3.3677

## Files changed
- tests/benchmark/test_llms.py (added test_mradermacher_cactus_dream_horror_12b_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr check)
- .github/workflows/perf-bench-matrix.json (added mradermacher_cactus_dream_horror_12b_i1_gguf entry)

## tt-forge-models submodule
no change
