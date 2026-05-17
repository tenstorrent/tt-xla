loader_path: third_party.tt_forge_models.argilla_llama.causal_lm.pytorch.loader
variant_id: llama_3_2_1b_instruct_apigen_fc_v0_1
arch: n150
status: DONE_FAIL
test_function: test_argilla_llama_3_2_1b_instruct_apigen_fc_v0_1
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
failure_reason: "hardware failure: Timeout waiting for Ethernet core service remote IO request flush — n300 Ethernet link between chips is unresponsive at device init; no workaround available within this skill"

# Benchmark added: test_argilla_llama_3_2_1b_instruct_apigen_fc_v0_1

## Test
tests/benchmark/test_llms.py::test_argilla_llama_3_2_1b_instruct_apigen_fc_v0_1

## Model
- HF name:    argilla/Llama-3.2-1B-Instruct-APIGen-FC-v0.1
- Loader:     third_party.tt_forge_models.argilla_llama.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_3_2_1B_INSTRUCT_APIGEN_FC_V0_1

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (hardware failure)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n300 (Wormhole)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not complete)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        wormhole (n300)
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

## Failure detail
All three device-initialization attempts failed with:
```
RuntimeError: Timeout waiting for Ethernet core service remote IO request flush.
 1. tt::umd::RemoteCommunicationLegacyFirmware::wait_for_non_mmio_flush(...)
 2. tt::umd::Cluster::broadcast_tensix_risc_reset_to_cluster(...)
 3. tt::umd::Cluster::deassert_resets_and_set_power_state()
 4. tt::umd::Cluster::start_device(...)
```
`tt-smi --reset 0` was run between each attempt; chip 1 (Ethernet-connected)
has no PCI BDF and cannot be reset independently. The n300 Ethernet link is
broken at the hardware/firmware level. This is not a test or model issue.

## Files changed
- tests/benchmark/test_llms.py (test function added)
- SUMMARY.md (this file)

## tt-forge-models submodule
no change
