loader_path: third_party.tt_forge_models.llama_300m_v5_fivegram.causal_lm.pytorch.loader
variant_id: llama_300M_v5_fivegram
arch: p150
status: DONE_PASS
test_function: test_llama_300m_v5_fivegram
samples_per_second: 230.9445408190609
ttft_ms: 70.430871
prefill_pcc: 0.985694
first_decode_pcc: 0.993820
top_perf_samples_per_sec: 893.5162
pct_of_target: 25.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_llama_300m_v5_fivegram

## Test
tests/benchmark/test_llms.py::test_llama_300m_v5_fivegram

## Model
- HF name:    deqing/llama-300M-v5-fivegram
- Loader:     third_party.tt_forge_models.llama_300m_v5_fivegram.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_300M_V5_FIVEGRAM

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  230.9445408190609
- TTFT (ms):          70.430871
- Prefill PCC:        0.985694
- First decode PCC:   0.993820
- Wall clock:         0:02:03
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_300m_v5_fivegram_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 25.8%

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
- total_flops:             20484980800
- breakdown.matmul:        20484980800
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        50331648
- memory_bytes: 100663296
- memory_gb:    0.09375

### Params
- count:                  451437731
- effective_count:        320103587
- memory_bytes:           602802824
- memory_gb:              0.5614038780331612
- effective_memory_bytes: 340134536
- effective_memory_gb:    0.31677497178316116
- embedding_count:        131334144
- embedding_memory_bytes: 262668288

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 893.5162
- top_perf_time_ms:         1.1192
- dram_time_ms:             0.7461
- compute_time_ms_lofi:     0.0233
- compute_time_ms_hifi2:    0.0466
- compute_time_ms_hifi3:    0.0698
- compute_time_ms_hifi4:    0.0931

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: added hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
