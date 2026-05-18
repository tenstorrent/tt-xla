loader_path: third_party.tt_forge_models.corianas_quokka_256m.causal_lm.pytorch.loader
variant_id: Quokka_256m
arch: p150
status: DONE_PASS
test_function: test_corianas_quokka_256m
samples_per_second: 111.48
ttft_ms: 114.24
prefill_pcc: 0.999165
first_decode_pcc: 0.999174
top_perf_samples_per_sec: 874.4023
pct_of_target: 12.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_corianas_quokka_256m

## Test
tests/benchmark/test_llms.py::test_corianas_quokka_256m

## Model
- HF name:    Corianas/Quokka_256m
- Loader:     third_party.tt_forge_models.corianas_quokka_256m.causal_lm.pytorch.loader
- Variant:    ModelVariant.QUOKKA_256M

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  111.48
- TTFT (ms):          114.24
- Prefill PCC:        0.999165
- First decode PCC:   0.999174
- Wall clock:         0:02:45
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_corianas_quokka_256m_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 12.7% (111.48 / 874.40)

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
- total_flops:             275939012608
- breakdown.matmul:        59494973440
- breakdown.linear:        216444039168
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        124780544
- memory_bytes: 249561088
- memory_gb:    0.232421875

### Params
- count:                  310663300
- effective_count:        253752196
- memory_bytes:           383896156
- memory_gb:              0.3575311563909054
- effective_memory_bytes: 270073948
- effective_memory_gb:    0.25152596458792686
- embedding_count:        56911104
- embedding_memory_bytes: 113822208

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 874.4023
- top_perf_time_ms:         1.1436
- dram_time_ms:             0.7624
- compute_time_ms_lofi:     0.3136
- compute_time_ms_hifi2:    0.6271
- compute_time_ms_hifi3:    0.9407
- compute_time_ms_hifi4:    1.2543

## Files changed
- tests/benchmark/test_llms.py (added test_corianas_quokka_256m)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
