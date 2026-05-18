loader_path: third_party.tt_forge_models.acestep.causal_lm.pytorch.loader
variant_id: acestep_5hz_lm_4b
arch: p150
status: DONE_PASS
test_function: test_acestep_5hz_lm_4b
samples_per_second: 34.121070523995705
ttft_ms: 332.299118
prefill_pcc: 0.940592
first_decode_pcc: 0.984814
top_perf_samples_per_sec: 73.6918
pct_of_target: 46.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_acestep_5hz_lm_4b

## Test
tests/benchmark/test_llms.py::test_acestep_5hz_lm_4b

## Model
- HF name:    ACE-Step/acestep-5Hz-lm-4B
- Loader:     third_party.tt_forge_models.acestep.causal_lm.pytorch.loader
- Variant:    ModelVariant.ACESTEP_5HZ_LM_4B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  34.121070523995705
- TTFT (ms):          332.299118
- Prefill PCC:        0.940592
- First decode PCC:   0.984814
- Wall clock:         0:08:29
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_acestep_5hz_lm_4b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 46.3% (34.12 / 73.69)

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
- total_flops:             268118917248
- breakdown.matmul:        268118917248
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
- count:                  4745596611
- effective_count:        4189554371
- memory_bytes:           5563670408
- memory_gb:              5.18157184869051
- effective_memory_bytes: 4451585928
- effective_memory_gb:    4.145862467586994
- embedding_count:        556042240
- embedding_memory_bytes: 1112084480

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 73.6918
- top_perf_time_ms:         13.5700
- dram_time_ms:             9.0467
- compute_time_ms_lofi:     0.3047
- compute_time_ms_hifi2:    0.6094
- compute_time_ms_hifi3:    0.9140
- compute_time_ms_hifi4:    1.2187

## Files changed
- tests/benchmark/test_llms.py (added test_acestep_5hz_lm_4b)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: guard get_weight_dtype_config_path with hasattr, matching dynamic_torch_model_tester.py pattern)

## tt-forge-models submodule
no change
