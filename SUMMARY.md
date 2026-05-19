loader_path: third_party.tt_forge_models.exaone_3_5_gguf.causal_lm.pytorch.loader
variant_id: 3.5_7.8B_Instruct_GGUF
arch: p150
status: DONE_PASS
test_function: test_exaone_3_5_7_8b_instruct_gguf
samples_per_second: 4.59712741946885
ttft_ms: 816.135836
prefill_pcc: 0.999600
first_decode_pcc: 0.999311
top_perf_samples_per_sec: 43.1682
pct_of_target: 10.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_exaone_3_5_7_8b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_exaone_3_5_7_8b_instruct_gguf

## Model
- HF name:    lmstudio-community/EXAONE-3.5-7.8B-Instruct-GGUF
- Loader:     third_party.tt_forge_models.exaone_3_5_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.EXAONE_3_5_7_8B_INSTRUCT_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  4.59712741946885
- TTFT (ms):          816.135836
- Prefill PCC:        0.999600
- First decode PCC:   0.999311
- Wall clock:         0:44:52
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_exaone_3_5_7_8b_instruct_gguf_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 10.6%

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
- total_flops:             475667628160
- breakdown.matmul:        475667628160
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
- count:                  7818449094
- effective_count:        7399018694
- memory_bytes:           8700568340
- memory_gb:              8.10303570702672
- effective_memory_bytes: 7861707540
- effective_memory_gb:    7.32178570702672
- embedding_count:        419430400
- embedding_memory_bytes: 838860800

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 43.1682
- top_perf_time_ms:         23.1652
- dram_time_ms:             15.4435
- compute_time_ms_lofi:     0.4574
- compute_time_ms_hifi2:    0.9147
- compute_time_ms_hifi3:    1.3721
- compute_time_ms_hifi4:    1.8295

## Files changed
- tests/benchmark/test_llms.py (added test_exaone_3_5_7_8b_instruct_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (added hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
