loader_path: third_party.tt_forge_models.qwen_3_swallow.causal_lm.pytorch.loader
variant_id: 8B_CPT_v0_2
arch: p150
status: DONE_PASS
test_function: test_qwen_3_swallow_8b_cpt_v0_2
samples_per_second: 25.6512
ttft_ms: 376.011319
prefill_pcc: 0.997103
first_decode_pcc: 0.999128
top_perf_samples_per_sec: 42.0551
pct_of_target: 61.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_qwen_3_swallow_8b_cpt_v0_2

## Test
tests/benchmark/test_llms.py::test_qwen_3_swallow_8b_cpt_v0_2

## Model
- HF name:    tokyotech-llm/Qwen3-Swallow-8B-CPT-v0.2
- Loader:     third_party.tt_forge_models.qwen_3_swallow.causal_lm.pytorch.loader
- Variant:    QWEN3_SWALLOW_8B_CPT_V0_2

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  25.6512
- TTFT (ms):          376.011319
- Prefill PCC:        0.997103
- First decode PCC:   0.999128
- Wall clock:         0:09:11
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_qwen_3_swallow_8b_cpt_v0_2_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 61.0%

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
- total_flops:             484358226048
- breakdown.matmul:        484358226048
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
- count:                  8190735555
- effective_count:        7568405699
- memory_bytes:           9286380296
- memory_gb:              8.64861560612917
- effective_memory_bytes: 8041720584
- effective_memory_gb:    7.4894359186291695
- embedding_count:        622329856
- embedding_memory_bytes: 1244659712

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.0551
- top_perf_time_ms:         23.7784
- dram_time_ms:             15.8522
- compute_time_ms_lofi:     0.5504
- compute_time_ms_hifi2:    1.1008
- compute_time_ms_hifi3:    1.6512
- compute_time_ms_hifi4:    2.2016

## Files changed
- tests/benchmark/test_llms.py (added test_qwen_3_swallow_8b_cpt_v0_2)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr check)

## tt-forge-models submodule
no change
