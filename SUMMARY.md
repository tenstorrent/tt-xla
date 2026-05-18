loader_path: third_party.tt_forge_models.banana304343_quantum_vqe_mistral_gguf.causal_lm.pytorch.loader
variant_id: quantum_vqe_mistral_gguf
arch: p150
status: DONE_PASS
test_function: test_banana304343_quantum_vqe_mistral_gguf
samples_per_second: 35.96455598905523
ttft_ms: 294.351406
prefill_pcc: 0.989618
first_decode_pcc: 0.990557
top_perf_samples_per_sec: 44.8360
pct_of_target: 80.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: banana304343_quantum_vqe_mistral_gguf

## Test
tests/benchmark/test_llms.py::test_banana304343_quantum_vqe_mistral_gguf

## Model
- HF name:    Banana304343/quantum-vqe-mistral-gguf
- Loader:     third_party.tt_forge_models.banana304343_quantum_vqe_mistral_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QUANTUM_VQE_MISTRAL_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  35.96455598905523
- TTFT (ms):          294.351406
- Prefill PCC:        0.989618
- First decode PCC:   0.990557
- Wall clock:         0:08:03
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_banana304343_quantum_vqe_mistral_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 80.2%

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
- memory_gb:              7.289559133350849
- effective_memory_bytes: 7558669064
- effective_memory_gb:    7.039559133350849
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
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: graceful handling of optional get_weight_dtype_config_path method)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
