loader_path: third_party.tt_forge_models.aya_x_yeni_i1_gguf.causal_lm.pytorch.loader
variant_id: I1_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_aya_x_yeni_i1_q4_k_m_gguf
samples_per_second: 7.753180443709736
ttft_ms: 2060.925335
prefill_pcc: 0.999259
first_decode_pcc: 0.998648
top_perf_samples_per_sec: 39.8953
pct_of_target: 19.4
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: aya_x_yeni_i1_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_aya_x_yeni_i1_q4_k_m_gguf

## Model
- HF name:    mradermacher/Aya-X-Yeni-i1-GGUF
- Loader:     third_party.tt_forge_models.aya_x_yeni_i1_gguf.causal_lm.pytorch.loader
- Variant:    AYA_X_YENI_I1_Q4_K_M_GGUF (value: "I1_Q4_K_M_GGUF")

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 causes "Statically allocated circular buffers clash with L1 buffers"
(TT_THROW in tt_metal/impl/program/program.cpp during trace capture). Using optimization_level=1
as the most aggressive stable setting.

## Measured (full model, defaults)
- Sample per second:  7.753180443709736
- TTFT (ms):          2060.925335
- Prefill PCC:        0.999259
- First decode PCC:   0.998648
- Wall clock:         0:06:19
- Hardware:           p150 (blackhole p300c)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_aya_x_yeni_i1_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 19.4% (7.75 / 39.90)

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
- total_flops:             513785462912
- breakdown.matmul:        513785462912
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
- count:                  9076609220
- effective_count:        8028033220
- memory_bytes:           10627064586
- memory_gb:              9.897225150838494
- effective_memory_bytes: 8529912586
- effective_memory_gb:    7.944100150838494
- embedding_count:        1048576000
- embedding_memory_bytes: 2097152000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 39.8953
- top_perf_time_ms:         25.0656
- dram_time_ms:             16.7104
- compute_time_ms_lofi:     0.5838
- compute_time_ms_hifi2:    1.1677
- compute_time_ms_hifi3:    1.7515
- compute_time_ms_hifi4:    2.3354

## Files changed
- tests/benchmark/test_llms.py (added test_aya_x_yeni_i1_q4_k_m_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added aya_x_yeni_i1_q4_k_m_gguf entry)

## tt-forge-models submodule
93218a34fc9fc6a671e0e41101da470c80891b2a → e52ad04838f185f07ebba5ffb9e692644148e57a
(aya_x_yeni_i1_gguf loader not present in original commit; updated to version containing the loader)
