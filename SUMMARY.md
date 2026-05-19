loader_path: third_party.tt_forge_models.gams3_12b_instruct.causal_lm.pytorch.loader
variant_id: GAMS3_12B_INSTRUCT
arch: p150
status: DONE_PASS
test_function: test_gams3_12b_instruct
samples_per_second: 9.703643
ttft_ms: 849.321078
prefill_pcc: 0.998792
first_decode_pcc: 0.999143
top_perf_samples_per_sec: 26.3289
pct_of_target: 36.9
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_gams3_12b_instruct

## Test
tests/benchmark/test_llms.py::test_gams3_12b_instruct

## Model
- HF name:    cjvt/GaMS3-12B-Instruct
- Loader:     third_party.tt_forge_models.gams3_12b_instruct.causal_lm.pytorch.loader
- Variant:    ModelVariant.GAMS3_12B_INSTRUCT

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  9.703643
- TTFT (ms):          849.321078
- Prefill PCC:        0.998792
- First decode PCC:   0.999143
- Wall clock:         0:11:20
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gams3_12b_instruct_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 36.9% (9.70 / 26.33)

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
- total_flops:             752977182976
- breakdown.matmul:        752977182976
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        805306368
- memory_bytes: 1610612736
- memory_gb:    1.5

### Params
- count:                  12772913157
- effective_count:        11766034437
- memory_bytes:           14517419022
- memory_gb:              13.52
- effective_memory_bytes: 12503661582
- effective_memory_gb:    11.64
- embedding_count:        1006878720
- embedding_memory_bytes: 2013757440

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 26.3289
- top_perf_time_ms:         37.9810
- dram_time_ms:             25.3207
- compute_time_ms_lofi:     0.8557
- compute_time_ms_hifi2:    1.7113
- compute_time_ms_hifi3:    2.5670
- compute_time_ms_hifi4:    3.4226

## Files changed
- tests/benchmark/test_llms.py (added test_gams3_12b_instruct)
- tests/benchmark/benchmarks/llm_benchmark.py (2 bug fixes: layer_types override now syncs per-layer attention_type; get_weight_dtype_config_path now guarded with hasattr)

## tt-forge-models submodule
no change
