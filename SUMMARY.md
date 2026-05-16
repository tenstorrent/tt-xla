loader_path: third_party.tt_forge_models.acestep.causal_lm.pytorch.loader
variant_id: acestep_5hz_lm_0_6b
arch: n150
status: DONE_FAIL
test_function: test_acestep_5hz_lm_0_6b
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
failure_reason: "device initialization failed: N300 right chip ARC core unresponsive (ARC startup error at core 0-10 over NOC0, scratch_status=0xaa2c, postcode=0xc0de004d, Timed out after 300000ms); persistent across multiple tt-smi resets"

# Benchmark added: test_acestep_5hz_lm_0_6b

## Test
tests/benchmark/test_llms.py::test_acestep_5hz_lm_0_6b

## Model
- HF name:    ACE-Step/acestep-5Hz-lm-0.6B
- Loader:     third_party.tt_forge_models.acestep.causal_lm.pytorch.loader
- Variant:    ModelVariant.ACESTEP_5HZ_LM_0_6B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (device unavailable)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n300 (wormhole_b0, two-chip board, right chip ARC unresponsive)

## Failure detail
All 4 test attempts failed at device initialization (`torch_xla.device()`):
- Attempts 1–3: "Timeout waiting for Ethernet core service remote IO request flush"
  during `broadcast_tensix_risc_reset_to_cluster` (~6s timeout)
- Attempt 4 (after extended background init): "ARC startup error at core 0-10 over NOC0:
  scratch_status=0xaa2c, postcode=0xc0de004d, message_id=0x2c (Timed out after 300000 ms)"
  during `WormholeTTDevice::wait_arc_core_start` via `TopologyDiscovery::discover_remote_devices`

Both errors originate from the N300 right chip (remote chip) being unresponsive.
`tt-smi --reset 0` was run multiple times; it completed successfully but did not restore
the device. The test code itself is correct and should run once hardware is healthy.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test could not run due to device failure
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        wormhole_b0
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
no change — submodule stays at 73ef037570
