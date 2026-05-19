loader_path: third_party.tt_forge_models.llama_3_1_8b_instruct_lion_numbers_ft.causal_lm.pytorch.loader
variant_id: 3.1_8B_Instruct_Lion_Numbers_FT
arch: p150
status: DONE_PASS
test_function: test_llama_3_1_8b_instruct_lion_numbers_ft
samples_per_second: 22.094
ttft_ms: 587.772844
prefill_pcc: 0.999115
first_decode_pcc: 0.998422
top_perf_samples_per_sec: 42.4655
pct_of_target: 52.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_llama_3_1_8b_instruct_lion_numbers_ft

## Test
tests/benchmark/test_llms.py::test_llama_3_1_8b_instruct_lion_numbers_ft

## Model
- HF name:    eekay/Llama-3.1-8B-Instruct-lion-numbers-ft
- Loader:     third_party.tt_forge_models.llama_3_1_8b_instruct_lion_numbers_ft.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_3_1_8B_INSTRUCT_LION_NUMBERS_FT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  22.094
- TTFT (ms):          587.772844
- Prefill PCC:        0.999115
- First decode PCC:   0.998422
- Wall clock:         0:19:09
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_3_1_8b_instruct_lion_numbers_ft_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 52.0% (22.094 / 42.4655)

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
- total_flops:             481640317056
- breakdown.matmul:        481640317056
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  8051232963
- effective_count:        7525896387
- memory_bytes:           9047188232
- memory_gb:              8.4258506372571
- effective_memory_bytes: 7996515080
- effective_memory_gb:    7.447335012257099
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.4655
- top_perf_time_ms:         23.5485
- dram_time_ms:             15.6990
- compute_time_ms_lofi:     0.5473
- compute_time_ms_hifi2:    1.0946
- compute_time_ms_hifi3:    1.6420
- compute_time_ms_hifi4:    2.1893

## Files changed
- tests/benchmark/test_llms.py (added test_llama_3_1_8b_instruct_lion_numbers_ft)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
