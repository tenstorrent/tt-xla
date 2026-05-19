loader_path: third_party.tt_forge_models.mistral_7b_instruct_v0_3_gguf.causal_lm.pytorch.loader
variant_id: 7B_Instruct_v0.3_GGUF
arch: p150
status: DONE_PASS
test_function: test_mistral_7b_instruct_v0_3_gguf
samples_per_second: 35.71
ttft_ms: 293.87
prefill_pcc: 0.998991
first_decode_pcc: 0.996214
top_perf_samples_per_sec: 44.8360
pct_of_target: 79.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: mistral_7b_instruct_v0_3_gguf

## Test
tests/benchmark/test_llms.py::test_mistral_7b_instruct_v0_3_gguf

## Model
- HF name:    bartowski/Mistral-7B-Instruct-v0.3-GGUF
- Loader:     third_party.tt_forge_models.mistral_7b_instruct_v0_3_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.MISTRAL_7B_INSTRUCT_V03_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  35.71
- TTFT (ms):          293.87
- Prefill PCC:        0.998991
- First decode PCC:   0.996214
- Wall clock:         0:09:24
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mistral_7b_instruct_v0_3_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 79.6% (35.71 / 44.84)

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
- total_flops:             455266533504
- breakdown.matmul:        455266533504
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
- count:                  7248023747
- effective_count:        7113806019
- memory_bytes:           7827104520
- memory_gb:              7.29
- effective_memory_bytes: 7558669064
- effective_memory_gb:    7.04
- embedding_count:        134217728
- embedding_memory_bytes: 268435456

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 44.8360
- top_perf_time_ms:         22.3035
- dram_time_ms:             14.8690
- compute_time_ms_lofi:     0.5173
- compute_time_ms_hifi2:    1.0347
- compute_time_ms_hifi3:    1.5520
- compute_time_ms_hifi4:    2.0694

## Files changed
- tests/benchmark/test_llms.py (updated loader from mistral_gguf to mistral_7b_instruct_v0_3_gguf)
- .github/workflows/perf-bench-matrix.json (test entry already present)
- tests/benchmark/benchmarks/llm_benchmark.py (getattr fix for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
