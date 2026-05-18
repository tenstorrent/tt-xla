loader_path: third_party.tt_forge_models.clover_lm.causal_lm.pytorch.loader
variant_id: CloverLM
arch: p150
status: DONE_FAIL
test_function: test_clover_lm
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "compiler error: ttnn.scaled_dot_product_attention_decode failed validation: k_chunk_size % 32 == 0, maximum calculated k_chunk_size is: 2 (SDPA decode op constraint failure in TT-MLIR)"

# Benchmark added: test_clover_lm

## Test
tests/benchmark/test_llms.py::test_clover_lm

## Model
- HF name:    daslab-testing/CloverLM
- Loader:     third_party.tt_forge_models.clover_lm.causal_lm.pytorch.loader
- Variant:    ModelVariant.CLOVER_LM

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (DONE_FAIL)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details

The test fails during compilation (warmup phase) with a compiler-level error in
TT-MLIR's SDPA decode path:

```
loc("custom-call.601"): error: OperationValidationAndFallback:
Operation ttnn.scaled_dot_product_attention_decode failed validation
(MetalBackendError: TT_FATAL: k_chunk_size % 32 == 0
 Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2)
```

CloverLM uses an unusual GQA configuration (Q heads=32, KV groups=28,
d_head=128 with seq_len=1 for decode) which causes `k_chunk_size` to be
calculated as 2 — not a multiple of 32 as required by the SDPA decode
kernel. This is a compiler bug; no test-level workaround is available.

Additional infrastructure fix applied: `llm_benchmark.py` updated to use
`hasattr(model_loader, "get_weight_dtype_config_path")` guard before calling
the method (mirrors the existing pattern in `dynamic_torch_model_tester.py`),
so loaders that don't implement this optional method no longer raise
`AttributeError`.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (compilation failed before any decode graph was executed)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        N/A
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Compute
- total_flops:             N/A
- breakdown.matmul:        N/A
- breakdown.linear:        N/A
- breakdown.conv2d:        N/A
- breakdown.sparse_matmul: N/A

### Inputs
- count:        N/A
- memory_bytes: N/A

### KV cache
- count:        N/A
- memory_bytes: N/A
- memory_gb:    N/A

### Params
- count:                  N/A
- effective_count:        N/A
- memory_bytes:           N/A
- memory_gb:              N/A
- effective_memory_bytes: N/A
- effective_memory_gb:    N/A
- embedding_count:        N/A
- embedding_memory_bytes: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A
- dram_time_ms:             N/A
- compute_time_ms_lofi:     N/A
- compute_time_ms_hifi2:    N/A
- compute_time_ms_hifi3:    N/A
- compute_time_ms_hifi4:    N/A

## Files changed
- tests/benchmark/test_llms.py (added test_clover_lm)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
