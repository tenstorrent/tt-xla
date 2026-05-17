loader_path: third_party.tt_forge_models.atom_olmo3_7b_i1_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: n150
status: DONE_FAIL
test_function: test_atom_olmo3_7b_i1_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 23.3867
pct_of_target: null
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: null
failure_reason: "OOM during full model execution: TT_FATAL bank_manager.cpp:462 at batch_size=32 with optimization_level=0; optimization_level=1,2 fail with 'ttnn.scaled_dot_product_attention op Query and result must have the same element type' compiler error"

# Benchmark added: test_atom_olmo3_7b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_atom_olmo3_7b_i1_gguf

## Model
- HF name:    mradermacher/atom-olmo3-7b-i1-GGUF
- Loader:     third_party.tt_forge_models.atom_olmo3_7b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ATOM_OLMO3_7B_Q4_K_M (= "Q4_K_M")

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: none
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure analysis
- optimization_level=1,2: compiler error — `ttnn.scaled_dot_product_attention` op:
  "Query and result must have the same element type" during compilation of attention layer.
- optimization_level=0 (only viable level): TT_FATAL in bank_manager.cpp:462 during
  execution of full 7B model at batch_size=32 — DRAM OOM at runtime.
- 1-layer test at optimization_level=0 passes cleanly:
  Prefill PCC=0.999997, First decode PCC=0.999925, ~42.6 samples/sec.

## Measured (full model, defaults)
- Sample per second:  null (full model OOM)
- TTFT (ms):          null
- Prefill PCC:        null
- First decode PCC:   null
- Wall clock:         ~9:07 (failed)
- Hardware:           n150 (wormhole_b0, n300 board single chip)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_atom_olmo3_7b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (full model did not complete)

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
- total_flops:             442899103872
- breakdown.matmul:        442899103872
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1073741824
- memory_bytes: 2147483648
- memory_gb:    2

### Params
- count:                  7298011334
- effective_count:        6887272646
- memory_bytes:           8139700500
- memory_gb:              7.58
- effective_memory_bytes: 7318223124
- effective_memory_gb:    6.82
- embedding_count:        410738688
- embedding_memory_bytes: 821477376

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 23.3867
- top_perf_time_ms:         42.7594
- dram_time_ms:             28.5062
- compute_time_ms_lofi:     1.7301
- compute_time_ms_hifi2:    3.4601
- compute_time_ms_hifi3:    5.1902
- compute_time_ms_hifi4:    6.9203

## Files changed
- tests/benchmark/test_llms.py (added test_atom_olmo3_7b_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path missing method)
- .github/workflows/perf-bench-matrix.json (added atom_olmo3_7b_i1_gguf entry)

## tt-forge-models submodule
no change
