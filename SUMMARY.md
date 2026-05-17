loader_path: third_party.tt_forge_models.bartowski_arcee_ai_homunculus_gguf.causal_lm.pytorch.loader
variant_id: arcee_ai_Homunculus_GGUF
arch: n150
status: DONE_FAIL
test_function: test_bartowski_arcee_ai_homunculus_gguf
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
failure_reason: "n300 remote chip Ethernet initialization timeout: torch_xla.device() fails with 'Timeout waiting for Ethernet core service remote IO request flush' in Cluster::broadcast_tensix_risc_reset_to_cluster during PJRT client init; device hardware unavailable after multiple tt-smi resets"

# Benchmark added: test_bartowski_arcee_ai_homunculus_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_arcee_ai_homunculus_gguf

## Model
- HF name:    bartowski/arcee-ai_Homunculus-GGUF
- Loader:     third_party.tt_forge_models.bartowski_arcee_ai_homunculus_gguf.causal_lm.pytorch.loader
- Variant:    arcee_ai_Homunculus_GGUF

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
- Hardware:           n300 (n150 single-chip mode intended)

## Failure details
The test function was successfully added to `tests/benchmark/test_llms.py` and
the matrix entry was added to `.github/workflows/perf-bench-matrix.json`.
However, all benchmark runs failed at device initialization before any
model-specific code ran:

```
RuntimeError: Timeout waiting for Ethernet core service remote IO request flush.
 1. tt::umd::RemoteCommunicationLegacyFirmware::wait_for_non_mmio_flush
 2. tt::umd::Cluster::broadcast_tensix_risc_reset_to_cluster
 3. tt::umd::Cluster::deassert_resets_and_set_power_state
 4. tt::umd::Cluster::start_device
```

The n300's right (remote) chip is not responding via its Ethernet link. Multiple
`tt-smi --reset 0` calls did not recover it. This is a hardware-infrastructure
failure that blocks all TT tests on this machine, not a model-specific issue.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (device unavailable)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150
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
- tests/benchmark/test_llms.py (added test_bartowski_arcee_ai_homunculus_gguf)
- .github/workflows/perf-bench-matrix.json (added bartowski_arcee_ai_homunculus_gguf entry)
- SUMMARY.md

## tt-forge-models submodule
no change
