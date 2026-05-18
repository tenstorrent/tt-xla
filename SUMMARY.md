loader_path: third_party.tt_forge_models.gemma3_12b_gguf.causal_lm.pytorch.loader
variant_id: chatpdflocal_12B_IT_GGUF
arch: p150
status: DONE_PASS
test_function: test_chatpdflocal_gemma3_12b_it_gguf
samples_per_second: 13.558
ttft_ms: 963.60
prefill_pcc: 0.995491
first_decode_pcc: 0.992714
top_perf_samples_per_sec: 26.3294
pct_of_target: 51.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_chatpdflocal_gemma3_12b_it_gguf

## Test
tests/benchmark/test_llms.py::test_chatpdflocal_gemma3_12b_it_gguf

## Model
- HF name:    chatpdflocal/gemma-3-12b-it-gguf
- Loader:     third_party.tt_forge_models.gemma3_12b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.CHATPDFLOCAL_GEMMA_3_12B_IT_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  13.558
- TTFT (ms):          963.60
- Prefill PCC:        0.995491
- First decode PCC:   0.992714
- Wall clock:         0:28:00
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_chatpdflocal_gemma3_12b_it_gguf_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 51.5% (13.558 / 26.3294)

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
- total_flops:             752961454336
- breakdown.matmul:        752961454336
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        805306368
- memory_bytes: 1610612736
- memory_gb:    1.5

### Params
- count:                  12772421637
- effective_count:        11765788677
- memory_bytes:           14516666382
- memory_gb:              13.519699109718204
- effective_memory_bytes: 12503400462
- effective_memory_gb:    11.644699109718204
- embedding_count:        1006632960
- embedding_memory_bytes: 2013265920

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 26.3294
- top_perf_time_ms:         37.9803
- dram_time_ms:             25.3202
- compute_time_ms_lofi:     0.8556
- compute_time_ms_hifi2:    1.7113
- compute_time_ms_hifi3:    2.5669
- compute_time_ms_hifi4:    3.4226

## Files changed
- tests/benchmark/test_llms.py (new test function test_chatpdflocal_gemma3_12b_it_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: sync per-layer attention_type after config.layer_types override; fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added chatpdflocal_gemma3_12b_it_gguf entry)

## tt-forge-models submodule
no change
