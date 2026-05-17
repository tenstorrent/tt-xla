loader_path: third_party.tt_forge_models.aspire_v4_alt_8b_model_stock_gguf.causal_lm.pytorch.loader
variant_id: Aspire_V4_ALT_8B_Model_Stock_GGUF
arch: n150
status: DONE_FAIL
test_function: test_aspire_v4_alt_8b_model_stock_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "hardware device not available: persistent Ethernet core timeout on n300 Wormhole board (broadcast_tensix_risc_reset_to_cluster); multiple tt-smi resets did not resolve"

# Benchmark added: test_aspire_v4_alt_8b_model_stock_gguf

## Test
tests/benchmark/test_llms.py::test_aspire_v4_alt_8b_model_stock_gguf

## Model
- HF name:    mradermacher/Aspire_V4_ALT-8B-Model_Stock-i1-GGUF
- Loader:     third_party.tt_forge_models.aspire_v4_alt_8b_model_stock_gguf.causal_lm.pytorch.loader
- Variant:    Aspire_V4_ALT_8B_Model_Stock_GGUF

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
- Hardware:           n300 (Wormhole, single-chip n150 equivalent)

## Failure
The test function was successfully added to tests/benchmark/test_llms.py, but
the benchmark could not be executed due to a persistent hardware failure on the
n300 Wormhole board. The device's remote chip (chip 1) fails to respond over
Ethernet during device initialization:

  RuntimeError: Timeout waiting for Ethernet core service remote IO request flush.
  → tt::umd::Cluster::broadcast_tensix_risc_reset_to_cluster

Multiple tt-smi resets (single-chip and dual-chip) were attempted but did not
resolve the issue. The failure occurs before any model weight download or
compilation step.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test could not run)
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
- tests/benchmark/test_llms.py  (added test_aspire_v4_alt_8b_model_stock_gguf)

## tt-forge-models submodule
no change
