loader_path: third_party.tt_forge_models.mitkox_gemma_2b_dpo_uncensored_4bit.causal_lm.pytorch.loader
variant_id: gemma_2b_dpo_uncensored_4bit
arch: p150
status: DONE_PASS
test_function: test_mitkox_gemma_2b_dpo_uncensored_4bit
samples_per_second: 79.94929412259451
ttft_ms: 157.466235
prefill_pcc: 0.994689
first_decode_pcc: 0.999829
top_perf_samples_per_sec: 130.1101
pct_of_target: 61.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_mitkox_gemma_2b_dpo_uncensored_4bit

## Test
tests/benchmark/test_llms.py::test_mitkox_gemma_2b_dpo_uncensored_4bit

## Model
- HF name:    mitkox/gemma-2b-dpo-uncensored-4bit
- Loader:     third_party.tt_forge_models.mitkox_gemma_2b_dpo_uncensored_4bit.causal_lm.pytorch.loader
- Variant:    GEMMA_2B_DPO_UNCENSORED_4BIT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  79.94929412259451
- TTFT (ms):          157.466235
- Prefill PCC:        0.994689
- First decode PCC:   0.999829
- Wall clock:         0:05:38
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mitkox_gemma_2b_dpo_uncensored_4bit_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 61.4%

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
- total_flops:             160390185216
- breakdown.matmul:        160390185216
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        37748736
- memory_bytes: 75497472
- memory_gb:    0.0703125

### Params
- count:                  3030460677
- effective_count:        2506172677
- memory_bytes:           3711607822
- memory_gb:              3.4567041527479887
- effective_memory_bytes: 2663031822
- effective_memory_gb:    2.4801416527479887
- embedding_count:        524288000
- embedding_memory_bytes: 1048576000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 130.1101
- top_perf_time_ms:         7.6858
- dram_time_ms:             5.1239
- compute_time_ms_lofi:     0.1823
- compute_time_ms_hifi2:    0.3645
- compute_time_ms_hifi3:    0.5468
- compute_time_ms_hifi4:    0.7290

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
