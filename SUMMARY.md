loader_path: third_party.tt_forge_models.granite_8b_code_instruct_gguf.causal_lm.pytorch.loader
variant_id: 8B_Code_Instruct_128K_GGUF
arch: p150
status: DONE_PASS
test_function: test_granite_8b_code_instruct_128k_gguf
samples_per_second: 23.56
ttft_ms: 381.55
prefill_pcc: 0.9452
first_decode_pcc: 0.9982
top_perf_samples_per_sec: 39.6141
pct_of_target: 59.5
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_granite_8b_code_instruct_128k_gguf

## Test
tests/benchmark/test_llms.py::test_granite_8b_code_instruct_128k_gguf

## Model
- HF name:    RichardErkhov/ibm-granite_-_granite-8b-code-instruct-128k-gguf
- Loader:     third_party.tt_forge_models.granite_8b_code_instruct_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GRANITE_8B_CODE_INSTRUCT_128K_GGUF

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  23.56
- TTFT (ms):          381.55
- Prefill PCC:        0.9452
- First decode PCC:   0.9982
- Wall clock:         0:04:40
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_granite_8b_code_instruct_128k_gguf_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 59.5% (23.56 / 39.61)

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
- total_flops:             515396075648
- breakdown.matmul:        515396075648
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
- count:                  8254689475
- effective_count:        8053362883
- memory_bytes:           8959632136
- memory_gb:              8.3443076685071
- effective_memory_bytes: 8556978952
- effective_memory_gb:    7.969307668507099
- embedding_count:        201326592
- embedding_memory_bytes: 402653184

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 39.6141
- top_perf_time_ms:         25.2435
- dram_time_ms:             16.8290
- compute_time_ms_lofi:     0.5857
- compute_time_ms_hifi2:    1.1714
- compute_time_ms_hifi3:    1.7570
- compute_time_ms_hifi4:    2.3427

## Files changed
- tests/benchmark/test_llms.py (added test_granite_8b_code_instruct_128k_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
