loader_path: third_party.tt_forge_models.gpt_oss_dflash.causal_lm.pytorch.loader
variant_id: 20B_DFlash
arch: p150
status: DONE_PASS
test_function: test_gpt_oss_dflash_20b
samples_per_second: 104.02
ttft_ms: 90.65
prefill_pcc: 0.9979
first_decode_pcc: 0.9978
top_perf_samples_per_sec: 244.10
pct_of_target: 42.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: gpt_oss_dflash_20b

## Test
tests/benchmark/test_llms.py::test_gpt_oss_dflash_20b

## Model
- HF name:    z-lab/gpt-oss-20b-DFlash
- Loader:     third_party.tt_forge_models.gpt_oss_dflash.causal_lm.pytorch.loader
- Variant:    ModelVariant.GPT_OSS_20B_DFLASH

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  104.02
- TTFT (ms):          90.65
- Prefill PCC:        0.9979
- First decode PCC:   0.9978
- Wall clock:         0:02:17
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gpt_oss_dflash_20b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 42.6%

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
- total_flops:             84629995584
- breakdown.matmul:        71038402624
- breakdown.linear:        13591592960
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        33554432
- memory_bytes: 67108864
- memory_gb:    0.0625

### Params
- count:                  1901559268
- effective_count:        1322425828
- memory_bytes:           2563451660
- memory_gb:              2.39
- effective_memory_bytes: 1405184780
- effective_memory_gb:    1.31
- embedding_count:        579133440
- embedding_memory_bytes: 1158266880

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 244.10
- top_perf_time_ms:         4.0967
- dram_time_ms:             2.7312
- compute_time_ms_lofi:     0.0962
- compute_time_ms_hifi2:    0.1923
- compute_time_ms_hifi3:    0.2885
- compute_time_ms_hifi4:    0.3847

## Files changed
- tests/benchmark/test_llms.py (added test_gpt_oss_dflash_20b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added gpt_oss_dflash_20b entry)

## tt-forge-models submodule
no change
