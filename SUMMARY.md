loader_path: third_party.tt_forge_models.darkidol_llama_3_1_8b_instruct.causal_lm.pytorch.loader
variant_id: Default
arch: p150
status: DONE_PASS
test_function: test_darkidol_llama_3_1_8b_instruct
samples_per_second: 4.579231759225871
ttft_ms: 818.763497
prefill_pcc: 0.998702
first_decode_pcc: 0.998651
top_perf_samples_per_sec: 42.5800
pct_of_target: 10.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_darkidol_llama_3_1_8b_instruct

## Test
tests/benchmark/test_llms.py::test_darkidol_llama_3_1_8b_instruct

## Model
- HF name:    aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.3-Uncensored
- Loader:     third_party.tt_forge_models.darkidol_llama_3_1_8b_instruct.causal_lm.pytorch.loader
- Variant:    ModelVariant.DARKIDOL_LLAMA_3_1_8B_INSTRUCT (Default)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  4.579231759225871
- TTFT (ms):          818.763497
- Prefill PCC:        0.998702
- First decode PCC:   0.998651
- Wall clock:         0:44:11
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_darkidol_llama_3_1_8b_instruct_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 10.8% (4.58 / 42.58)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             482445623424
- breakdown.matmul:        482445623424
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
- count:                  8030261446
- effective_count:        7504924870
- memory_bytes:           9024906004
- memory_gb:              8.40509869530797
- effective_memory_bytes: 7974232852
- effective_memory_gb:    7.42658307030797
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5800
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.4639
- compute_time_ms_hifi2:    0.9278
- compute_time_ms_hifi3:    1.3917
- compute_time_ms_hifi4:    1.8556

## Files changed
- tests/benchmark/test_llms.py (added test_darkidol_llama_3_1_8b_instruct)
- tests/benchmark/benchmarks/llm_benchmark.py (added hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added darkidol_llama_3_1_8b_instruct entry)

## tt-forge-models submodule
no change
