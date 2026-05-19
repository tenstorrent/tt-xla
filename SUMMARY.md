loader_path: third_party.tt_forge_models.smollm3_gguf.causal_lm.pytorch.loader
variant_id: 3B_Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_smollm3_gguf_3b_q4_k_m
samples_per_second: 54.89176031013845
ttft_ms: 245.871853
prefill_pcc: 0.997755
first_decode_pcc: 0.995693
top_perf_samples_per_sec: 102.5977
pct_of_target: 53.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: smollm3_gguf_3b_q4_k_m

## Test
tests/benchmark/test_llms.py::test_smollm3_gguf_3b_q4_k_m

## Model
- HF name:    bartowski/HuggingFaceTB_SmolLM3-3B-GGUF
- Loader:     third_party.tt_forge_models.smollm3_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.SMOLLM3_3B_Q4_K_M

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  54.89176031013845
- TTFT (ms):          245.871853
- Prefill PCC:        0.997755
- First decode PCC:   0.995693
- Wall clock:         0:08:32
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_smollm3_gguf_3b_q4_k_m_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 53.5%

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
- total_flops:             196796743808
- breakdown.matmul:        196796743808
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        150994944
- memory_bytes: 301989888
- memory_gb:    0.28125

### Params
- count:                  3337767108
- effective_count:        3075098820
- memory_bytes:           3792769804
- memory_gb:              3.532292138785124
- effective_memory_bytes: 3267433228
- effective_memory_gb:    3.043034326285124
- embedding_count:        262668288
- embedding_memory_bytes: 525336576

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 102.5977
- top_perf_time_ms:         9.7468
- dram_time_ms:             6.4979
- compute_time_ms_lofi:     0.2236
- compute_time_ms_hifi2:    0.4473
- compute_time_ms_hifi3:    0.6709
- compute_time_ms_hifi4:    0.8945

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
