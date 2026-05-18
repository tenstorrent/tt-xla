loader_path: third_party.tt_forge_models.biogpt.causal_lm.pytorch.loader
variant_id: Large
arch: p150
status: DONE_FAIL
test_function: test_biogpt_large
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
failure_reason: "compiler bug: failed to legalize stablehlo.reduce operation (loc reduce.33)"

# Benchmark added: test_biogpt_large

## Test
tests/benchmark/test_llms.py::test_biogpt_large

## Model
- HF name:    microsoft/BioGPT-Large
- Loader:     third_party.tt_forge_models.biogpt.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIOGPT_LARGE

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test failed during warm-up compilation with a compiler error:

    loc("reduce.33"): error: failed to legalize operation 'stablehlo.reduce'
    Failed to convert from SHLO to TTIR module
    ValueError: Error code: 13

The `stablehlo.reduce` operation in the BioGPT model graph could not be
legalized by the TT-MLIR compiler. This is a compiler bug outside the scope
of the benchmark test infrastructure. The test function has been added to
`test_llms.py` but is not currently producing measurable results.

Additional notes:
- Device: p150 (blackhole) — required TT_MESH_GRAPH_DESC_PATH to be set
- Missing Python dependency `sacremoses` was installed to enable BioGPT
  tokenizer initialization (BioGptTokenizer requires sacremoses)
- The 1-layer warm-up run confirmed the model loads successfully but fails
  at graph compilation time

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before generating perf metrics)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150
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
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
