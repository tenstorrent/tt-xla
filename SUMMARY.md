loader_path: third_party.tt_forge_models.cerebras_gpt_590m.causal_lm.pytorch.loader
variant_id: 590M
arch: p150
status: DONE_PASS
test_function: test_cerebras_gpt_590m
samples_per_second: 72.37
ttft_ms: 165.03
prefill_pcc: 0.999073
first_decode_pcc: 0.999567
top_perf_samples_per_sec: 406.79
pct_of_target: 17.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_cerebras_gpt_590m

## Test
tests/benchmark/test_llms.py::test_cerebras_gpt_590m

## Model
- HF name:    cerebras/Cerebras-GPT-590M
- Loader:     third_party.tt_forge_models.cerebras_gpt_590m.causal_lm.pytorch.loader
- Variant:    ModelVariant.GPT_590M (590M)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  72.37
- TTFT (ms):          165.03
- Prefill PCC:        0.999073
- First decode PCC:   0.999567
- Wall clock:         0:04:01
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_cerebras_gpt_590m_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 17.8% (72.37 / 406.79)

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
- total_flops:             638576689152
- breakdown.matmul:        83987890176
- breakdown.linear:        554588798976
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        226492416
- memory_bytes: 452984832
- memory_gb:    0.421875

### Params
- count:                  667505796
- effective_count:        587165316
- memory_bytes:           785381996
- memory_gb:              0.7314439825713634
- effective_memory_bytes: 624701036
- effective_memory_gb:    0.5817981772124767
- embedding_count:        80340480
- embedding_memory_bytes: 160680960

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 406.79254436090974
- top_perf_time_ms:         2.4583
- dram_time_ms:             1.6388
- compute_time_ms_lofi:     0.7257
- compute_time_ms_hifi2:    1.4513
- compute_time_ms_hifi3:    2.1770
- compute_time_ms_hifi4:    2.9026

## Files changed
- tests/benchmark/test_llms.py (added test_cerebras_gpt_590m)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added cerebras_gpt_590m CI entry)

## tt-forge-models submodule
no change
