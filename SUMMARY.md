loader_path: third_party.tt_forge_models.arabic_orpo_llama_3_8b_instruct.causal_lm.pytorch.loader
variant_id: Arabic_Orpo_Llama_3_8B_Instruct
arch: p150
status: DONE_PASS
test_function: test_arabic_orpo_llama_3_8b_instruct
samples_per_second: 4.59333393210405
ttft_ms: 801.927582
prefill_pcc: 0.998528
first_decode_pcc: 0.998889
top_perf_samples_per_sec: 42.5800
pct_of_target: 10.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_arabic_orpo_llama_3_8b_instruct

## Test
tests/benchmark/test_llms.py::test_arabic_orpo_llama_3_8b_instruct

## Model
- HF name:    MohamedRashad/Arabic-Orpo-Llama-3-8B-Instruct
- Loader:     third_party.tt_forge_models.arabic_orpo_llama_3_8b_instruct.causal_lm.pytorch.loader
- Variant:    Arabic_Orpo_Llama_3_8B_Instruct

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  4.59333393210405
- TTFT (ms):          801.927582
- Prefill PCC:        0.998528
- First decode PCC:   0.998889
- Wall clock:         0:44:13
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_arabic_orpo_llama_3_8b_instruct_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 10.8%

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
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: add hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
