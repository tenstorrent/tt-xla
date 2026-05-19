loader_path: third_party.tt_forge_models.llamafactory_lora.causal_lm.pytorch.loader
variant_id: tiny_random_Llama_3_lora
arch: p150
status: DONE_PASS
test_function: test_llamafactory_lora_tiny_random_llama_3
samples_per_second: 553.1637117540834
ttft_ms: 24.263743
prefill_pcc: 1.0
first_decode_pcc: 0.999958
top_perf_samples_per_sec: 142505.5708
pct_of_target: 0.4
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_llamafactory_lora_tiny_random_llama_3

## Test
tests/benchmark/test_llms.py::test_llamafactory_lora_tiny_random_llama_3

## Model
- HF name:    llamafactory/tiny-random-Llama-3 (base), llamafactory/tiny-random-Llama-3-lora (adapter)
- Loader:     third_party.tt_forge_models.llamafactory_lora.causal_lm.pytorch.loader
- Variant:    tiny_random_Llama_3_lora

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 fails with "No circular buffer with id 0 exists in Program" runtime error (hangs/crashes at compile time). optimization_level=1 passes cleanly.

## Measured (full model, defaults)
- Sample per second:  553.1637117540834
- TTFT (ms):          24.263743
- Prefill PCC:        1.000000
- First decode PCC:   0.999958
- Wall clock:         0:00:31
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llamafactory_lora_tiny_random_llama_3_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 0.4%

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
- total_flops:             131858436
- breakdown.matmul:        131858436
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        262144
- memory_bytes: 524288
- memory_gb:    0.00048828125

### Params
- count:                  4112597
- effective_count:        2060501
- memory_bytes:           6293936
- memory_gb:              0.005861684679985046
- effective_memory_bytes: 2189744
- effective_memory_gb:    0.0020393580198287964
- embedding_count:        2052096
- embedding_memory_bytes: 4104192

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 142505.5708
- top_perf_time_ms:         0.0070
- dram_time_ms:             0.0047
- compute_time_ms_lofi:     0.0001
- compute_time_ms_hifi2:    0.0003
- compute_time_ms_hifi3:    0.0004
- compute_time_ms_hifi4:    0.0006

## Files changed
- tests/benchmark/test_llms.py (added test_llamafactory_lora_tiny_random_llama_3)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
