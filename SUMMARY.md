loader_path: third_party.tt_forge_models.deepscaler_gguf.causal_lm.pytorch.loader
variant_id: 1.5B_Preview_GGUF
arch: p150
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
failure_reason: "PCIe device 3 (0000:c1:00.0) is hung (reads 0xffffffff); tt-metal topology discovery aborts on all 4-chip p150 enumeration; tt-smi --reset all failed to recover; hardware power cycle required"

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
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
Device 3 (PCIe 0000:c1:00.0, logical chip 3 of 4) on this p150 host reads
0xffffffff over PCIe — a PCIe hung-bus condition. tt-metal's topology discovery
(TopologyDiscovery::get_connected_devices) enumerates all physical devices and
aborts when it detects the hung device, preventing any PJRT client initialization.

Attempted recovery:
- tt-smi --reset 0     → failed (same 0xffffffff error post-reset)
- tt-smi -r all        → failed (only device 3 was seen by tt-smi; re-init failed)

The test function itself is correct and has been committed. Once the hardware is
power-cycled (or device 3 is physically reseated/replaced), rerunning this skill
should produce a successful benchmark.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not run)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150 (blackhole)
- chip_count_in_system_desc:   4 (PCIe: 0000:01:00.0, 0000:41:00.0, 0000:42:00.0, 0000:c1:00.0)
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

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
