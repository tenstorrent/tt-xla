loader_path: third_party.tt_forge_models.fast_dllm_v2.causal_lm.pytorch.loader
variant_id: 7B
arch: n150
status: DONE_FAIL
test_function: test_fast_dllm_v2_7b
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
failure_reason: "TT device hung: Read 0xffffffff over PCIe ID 3; tt-smi --reset also failed — device requires host-level power cycle"

# Benchmark added: test_fast_dllm_v2_7b

## Test
tests/benchmark/test_llms.py::test_fast_dllm_v2_7b

## Model
- HF name:    Efficient-Large-Model/Fast_dLLM_v2_7B
- Loader:     third_party.tt_forge_models.fast_dllm_v2.causal_lm.pytorch.loader
- Variant:    ModelVariant.FAST_DLLM_V2_7B (value="7B")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (device hang)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (wormhole_b0)

## Failure Details
The test failed at device initialization (not model loading). The TT device at PCIe ID 3 returned 0xffffffff on every read, indicating a hard PCIe hang. `tt-smi --reset 0` was attempted but also failed with the same error. The device requires a host-level power cycle or system reboot to recover.

Error: `RuntimeError: Read 0xffffffff over PCIe ID 3: the board should be reset.`

The test function itself is correctly authored and structurally sound (it was collected by pytest successfully). Once the device is recovered, this test should be re-run.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — device failed before model execution
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150 (wormhole_b0)
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      true
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
- tests/benchmark/test_llms.py (test_fast_dllm_v2_7b added at line ~1094)

## tt-forge-models submodule
no change
