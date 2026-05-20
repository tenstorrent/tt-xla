loader_path: third_party.tt_forge_models.deepscaler_gguf.causal_lm.pytorch.loader
variant_id: 1.5B_Preview_GGUF
arch: n150
status: DONE_FAIL
test_function: test_deepscaler_1_5b_preview_gguf
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
failure_reason: "hardware failure: only TT device exposed to container (/dev/tenstorrent/3) is in PCIe hung state (Read 0xffffffff over PCIe ID 3); tt-smi --reset failed to recover the device; PCI-level reset not permitted from inside container; power cycle required"

# Benchmark added: test_deepscaler_1_5b_preview_gguf

## Test
tests/benchmark/test_llms.py::test_deepscaler_1_5b_preview_gguf

## Model
- HF name:    bartowski/agentica-org_DeepScaleR-1.5B-Preview-GGUF
- Loader:     third_party.tt_forge_models.deepscaler_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPSCALER_1_5B_PREVIEW_GGUF (1.5B_Preview_GGUF)

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
- Hardware:           n150

## Hardware Failure Details
The container running this benchmark only had access to `/dev/tenstorrent/3`
(host PCIe device 3 at BDF 0000:42:00.0). This device was in a hung state:

    RuntimeError: Read 0xffffffff over PCIe ID 3: the board should be reset.

Attempted recovery:
1. `tt-smi --reset` — failed: "Error when re-initializing chips! Read 0xffffffff over PCIe ID 3"
2. PCI sysfs reset — not permitted from inside container (read-only filesystem)

A host-level power cycle or physical PCIe reset is required to recover.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test did not run
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
- tests/benchmark/test_llms.py (added test_deepscaler_1_5b_preview_gguf)
- .github/workflows/perf-bench-matrix.json (added deepscaler_1_5b_preview_gguf entry)

## tt-forge-models submodule
no change
