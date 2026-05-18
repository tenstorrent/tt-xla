loader_path: third_party.tt_forge_models.cat_translate_7b_i1_gguf.causal_lm.pytorch.loader
variant_id: CAT_Translate_7b_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_cat_translate_7b_i1_gguf
samples_per_second: 35.075951042860396
ttft_ms: 306.943216
prefill_pcc: 0.998857
first_decode_pcc: 0.999289
top_perf_samples_per_sec: 44.1260
pct_of_target: 79.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_cat_translate_7b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_cat_translate_7b_i1_gguf

## Model
- HF name:    mradermacher/CAT-Translate-7b-i1-GGUF
- Loader:     third_party.tt_forge_models.cat_translate_7b_i1_gguf.causal_lm.pytorch.loader
- Variant:    CAT_TRANSLATE_7B_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  35.076
- TTFT (ms):          306.943
- Prefill PCC:        0.998857
- First decode PCC:   0.999289
- Wall clock:         0:11:24
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_cat_translate_7b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 79.5%

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
- top_perf_samples_per_sec: 44.1260
- top_perf_time_ms:         22.6624
- dram_time_ms:             15.1082
- compute_time_ms_lofi:     0.5260
- compute_time_ms_hifi2:    1.0520
- compute_time_ms_hifi3:    1.5780
- compute_time_ms_hifi4:    2.1039

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json
- third_party/tt_forge_models (93218a34f → e0e38a88d: Add gguf>=0.10.0 requirement for CAT-Translate-7b i1 GGUF model)

## tt-forge-models submodule
93218a34fc9fc6a671e0e41101da470c80891b2a → e0e38a88d2d53c0d8d595fe8b53dd336e0f55d13
Add gguf>=0.10.0 requirement for CAT-Translate-7b i1 GGUF model
