loader_path: third_party.tt_forge_models.azure_starlight_12b_heretic_i1_gguf.causal_lm.pytorch.loader
variant_id: AZURE_STARLIGHT_12B_HERETIC_I1_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_azure_starlight_12b_heretic_i1_q4_k_m_gguf
samples_per_second: 20.077
ttft_ms: 498.922
prefill_pcc: 0.990339
first_decode_pcc: 0.994816
top_perf_samples_per_sec: 27.7857
pct_of_target: 72.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_azure_starlight_12b_heretic_i1_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_azure_starlight_12b_heretic_i1_q4_k_m_gguf

## Model
- HF name:    mradermacher/Azure-Starlight-12B-Heretic-i1-GGUF
- Loader:     third_party.tt_forge_models.azure_starlight_12b_heretic_i1_gguf.causal_lm.pytorch.loader
- Variant:    AZURE_STARLIGHT_12B_HERETIC_I1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  20.077
- TTFT (ms):          498.922
- Prefill PCC:        0.990339
- First decode PCC:   0.994816
- Wall clock:         0:13:20
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_azure_starlight_12b_heretic_i1_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 72.3% (20.077 / 27.7857)

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
- total_flops:             740882841728
- breakdown.matmul:        740882841728
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
- count:                  12247813315
- effective_count:        11576709315
- memory_bytes:           13642851016
- memory_gb:              12.705895133316517
- effective_memory_bytes: 12300643016
- effective_memory_gb:    11.455866523087025
- embedding_count:        671104000
- embedding_memory_bytes: 1342208000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 27.7857
- top_perf_time_ms:         35.9898
- dram_time_ms:             23.9932
- compute_time_ms_lofi:     0.8419
- compute_time_ms_hifi2:    1.6838
- compute_time_ms_hifi3:    2.5257
- compute_time_ms_hifi4:    3.3676

## Files changed
- tests/benchmark/test_llms.py (new test function added)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
