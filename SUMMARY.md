loader_path: third_party.tt_forge_models.m7.causal_lm.pytorch.loader
variant_id: 7B
arch: p150
status: DONE_FAIL
test_function: test_m7_7b
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
failure_reason: "compiler error: failed to legalize operation 'ttir.paged_update_cache' with optimization_level=2; full model OOM/killed with optimization_level=0 (~33min); root cause: M7-7B uses Mistral sliding window attention which requires paged KV cache not yet legalized in TTNN at opt2"

# Benchmark added: test_m7_7b

## Test
tests/benchmark/test_llms.py::test_m7_7b

## Model
- HF name:    liminerity/M7-7b
- Loader:     third_party.tt_forge_models.m7.causal_lm.pytorch.loader
- Variant:    ModelVariant.M7_7B (= "7B")

## Test config landed
- optimization_level:        2 (default — fails with paged_update_cache compiler bug)
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before performance benchmark)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         ~12 min (opt2), ~33 min (opt0, killed)
- Hardware:           p150 (Blackhole p300c)

## Bring-up results (num_layers=1)
- optimization_level=2, trace=True, num_layers=1: PASSED (PCC prefill=0.999106, first_decode=0.999428)
- optimization_level=0, trace=True, num_layers=1: PASSED (PCC prefill=0.998474, first_decode=0.999370)

## Failure analysis
The M7-7B model (based on Mistral architecture) uses sliding window attention.
The model loader explicitly handles `sliding_window` in `load_inputs()`. This requires
`paged_update_cache` for KV cache management in the decode graph.

At optimization_level=2 (default), the full 32-layer model fails during performance
benchmark compilation with:
  loc("scatter.11541"): error: failed to legalize operation 'ttir.paged_update_cache'

Single-layer (num_layers=1) passes at both opt0 and opt2 because the 1-layer decode
graph does not trigger the paged cache optimization path.

At optimization_level=0, the full model is killed by SIGKILL after approximately
33 minutes, suggesting either a compilation timeout or device hang during execution.

This is a compiler bug — `ttir.paged_update_cache` is not yet legalized in TTNN for
opt2. Fix belongs in the MLIR compiler, not in the test harness.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before producing perf metrics JSON)
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
- tests/benchmark/test_llms.py (added test_m7_7b)

## tt-forge-models submodule
old → 93218a34fc9fc6a671e0e41101da470c80891b2a → 7cf0e9b8df122b73c9a40dc67624f25d0232d3ee
(updated submodule to include m7 causal_lm loader)
