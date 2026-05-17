loader_path: third_party.tt_forge_models.codellama_7b_python_gguf.causal_lm.pytorch.loader
variant_id: CodeLlama_7B_Python_Q4_K_M_GGUF
arch: n150
status: DONE_FAIL
test_function: test_codellama_7b_python_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 24.2390
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "OOM: CodeLlama 7B uses MHA (num_kv_heads=32, no GQA), giving ~4x larger KV cache than equivalent GQA models; model (12.55 GB params) + KV cache (2 GB) exceeds n150 DRAM at batch_size=32"

# Benchmark added: test_codellama_7b_python_gguf

## Test
tests/benchmark/test_llms.py::test_codellama_7b_python_gguf

## Model
- HF name:    TheBloke/CodeLlama-7B-Python-GGUF
- Loader:     third_party.tt_forge_models.codellama_7b_python_gguf.causal_lm.pytorch.loader
- Variant:    CodeLlama_7B_Python_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2 (DEFAULT_OPTIMIZATION_LEVEL)
- trace_enabled:             true (DEFAULT_TRACE_ENABLED)
- experimental_weight_dtype: bfp_bf8 (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  null (OOM before measurement)
- TTFT (ms):          null
- Prefill PCC:        null
- First decode PCC:   null
- Wall clock:         ~10 min (OOM at runtime after compilation)
- Hardware:           n150 (Wormhole B0, left chip of n300)

## Failure Analysis
The full 7B model fails with OOM at runtime on n150 (batch_size=32):

```
Out of Memory: Not enough space to allocate 33554432 B DRAM buffer across 12 banks,
where each bank needs to store 2797568 B, but bank size is 1071821792 B
(allocated: 1060371776 B, free: 11450016 B, largest free block: 2097152 B)
```

Root cause: CodeLlama 7B Python uses MHA with num_kv_heads=32 (no GQA). Modern
models like Llama 3.x use GQA with num_kv_heads=8, giving 4x smaller KV cache.
At batch_size=32, seq_len=128, 32 layers, the KV cache is ~2 GB vs ~0.5 GB for GQA.
Combined with ~12.55 GB of model params (bfp_bf8) the total exceeds n150 DRAM.

Note: num_layers=1 test passes successfully (PCC ~0.999, 202 samples/sec).
The decode graph compilation succeeds (perf metrics JSON was written) but
execution fails when allocating the full KV cache + activations.

Also fixed a general harness bug in llm_benchmark.py: the benchmark called
model_loader.get_weight_dtype_config_path() unconditionally, but this method is
not present on all loaders. Fixed with hasattr() check (matching the pattern
already used in the runner at tests/runner/testers/torch/dynamic_torch_model_tester.py:82).

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_codellama_7b_python_gguf_perf_metrics_1.json
Note: Roofline computed from compilation output (full 7B decode graph compiled but OOM at runtime)
Achieved vs top_perf_samples_per_sec: N/A (model failed before measurement)

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             422852952192
- breakdown.matmul:        422852952192
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1073741824
- memory_bytes: 2147483648
- memory_gb:    2

### Params
- count:                  6738415680
- effective_count:        6607343680
- memory_bytes:           13476831360
- memory_gb:              12.55127727985382
- effective_memory_bytes: 13214687360
- effective_memory_gb:    12.30713665485382
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 24.2390
- top_perf_time_ms:         41.2558
- dram_time_ms:             27.5039
- compute_time_ms_lofi:     1.6518
- compute_time_ms_hifi2:    3.3035
- compute_time_ms_hifi3:    4.9553
- compute_time_ms_hifi4:    6.6071

## Files changed
- tests/benchmark/test_llms.py — added test_codellama_7b_python_gguf
- tests/benchmark/benchmarks/llm_benchmark.py — fixed hasattr() check for get_weight_dtype_config_path

## tt-forge-models submodule
no change
