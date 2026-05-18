loader_path: third_party.tt_forge_models.kimina_prover_distill.causal_lm.pytorch.loader
variant_id: 8B
arch: p150
status: DONE_PASS
test_function: test_kimina_prover_distill_8b
samples_per_second: 24.152815831880943
ttft_ms: 388.186734
prefill_pcc: 0.997904
first_decode_pcc: 0.997897
top_perf_samples_per_sec: 42.0551
pct_of_target: 57.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_kimina_prover_distill_8b

## Test
tests/benchmark/test_llms.py::test_kimina_prover_distill_8b

## Model
- HF name:    AI-MO/Kimina-Prover-Distill-8B
- Loader:     third_party.tt_forge_models.kimina_prover_distill.causal_lm.pytorch.loader
- Variant:    ModelVariant.KIMINA_PROVER_DISTILL_8B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  24.152815831880943
- TTFT (ms):          388.186734
- Prefill PCC:        0.997904
- First decode PCC:   0.997897
- Wall clock:         0:09:55
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_kimina_prover_distill_8b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 57.4%

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
- tests/benchmark/test_llms.py (added test_kimina_prover_distill_8b)
- tests/benchmark/benchmarks/llm_benchmark.py (added hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
