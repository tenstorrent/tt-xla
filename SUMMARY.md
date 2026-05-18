loader_path: third_party.tt_forge_models.apertus_8b_instruct_2509_gguf.causal_lm.pytorch.loader
variant_id: Apertus_8B_Instruct_2509_Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_apertus_8b_instruct_2509_gguf
samples_per_second: 7.699551377477467
ttft_ms: 461.33094
prefill_pcc: 1.000000
first_decode_pcc: 1.000000
top_perf_samples_per_sec: 42.5169
pct_of_target: 18.1
roofline_bound: dram
optimization_level: 2
trace_enabled: false
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_apertus_8b_instruct_2509_gguf

## Test
tests/benchmark/test_llms.py::test_apertus_8b_instruct_2509_gguf

## Model
- HF name:    unsloth/Apertus-8B-Instruct-2509-GGUF
- Loader:     third_party.tt_forge_models.apertus_8b_instruct_2509_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.APERTUS_8B_INSTRUCT_2509_Q4_K_M

## Test config landed
- optimization_level:        2
- trace_enabled:             false
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: trace_enabled=False because trace=True causes "system desc does not exist:
/tmp/tt_pjrt_system_descriptor" error during PCC benchmark compilation on p150
full model. The 1-layer run passes with trace=True but full 32-layer model fails.

## Measured (full model, defaults)
- Sample per second:  7.699551377477467
- TTFT (ms):          461.33094
- Prefill PCC:        1.000000
- First decode PCC:   1.000000
- Wall clock:         0:35:19
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_apertus_8b_instruct_2509_gguf_perf_metrics_4.json
Achieved vs top_perf_samples_per_sec: 18.1% (7.70 / 42.52)

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
- total_flops:             481036337280
- breakdown.matmul:        481036337280
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
- count:                  8053338304
- effective_count:        7516467392
- memory_bytes:           16106676608
- memory_gb:              15.000511527061462
- effective_memory_bytes: 15032934784
- effective_memory_gb:    14.000511527061462
- embedding_count:        536870912
- embedding_memory_bytes: 1073741824

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5169
- top_perf_time_ms:         23.5201
- dram_time_ms:             15.6800
- compute_time_ms_lofi:     0.5466
- compute_time_ms_hifi2:    1.0933
- compute_time_ms_hifi3:    1.6399
- compute_time_ms_hifi4:    2.1865

## Files changed
- tests/benchmark/test_llms.py (added test_apertus_8b_instruct_2509_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added apertus_8b_instruct_2509_gguf entry)

## tt-forge-models submodule
no change
