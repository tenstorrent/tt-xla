loader_path: third_party.tt_forge_models.abcorrea_bw_v1.causal_lm.pytorch.loader
variant_id: bw_v1
arch: n150
status: DONE_FAIL
test_function: test_abcorrea_bw_v1
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
failure_reason: "device ETH training timed out (900s) on wormhole eth core 22,16 during PJRT client initialization; reproduced on two consecutive runs; tt-smi reset unavailable in container; device requires hardware reset"

# Benchmark added: test_abcorrea_bw_v1

## Test
tests/benchmark/test_llms.py::test_abcorrea_bw_v1

## Model
- HF name:    abcorrea/bw-v1
- Loader:     third_party.tt_forge_models.abcorrea_bw_v1.causal_lm.pytorch.loader
- Variant:    ModelVariant.BW_V1 = "bw_v1"

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (device init failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         ~21 min (15 min ETH training timeout + model load)
- Hardware:           n150 (wormhole_b0)

## Failure details
Both test runs (--num-layers 1 --max-output-tokens 3) failed with identical error:

    RuntimeError: ETH training timed out after 900000 ms, on eth core 22, 16
    Location: …/umd/device/tt_device/wormhole_tt_device.cpp:255
    tt::umd::TopologyDiscovery::wait_eth_cores_training

The failure occurs during PJRT client initialization (torch_xla.device() call),
before any model compilation or inference runs. The TT UMD is waiting for the
Wormhole ETH PHY link training to complete and timing out after 900 seconds.
This is a hardware/infrastructure issue, not a model-specific problem. The test
code itself is correct.

Reset attempts:
- tt-smi: not found in $PATH or $HOME/.tenstorrent-venv/bin/
- PCI sysfs reset (/sys/bus/pci/drivers/tenstorrent/*/reset): read-only from container

The device requires a host-level hardware reset to recover.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not complete)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150 (wormhole_b0)
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A

## Files changed
- tests/benchmark/test_llms.py (added test_abcorrea_bw_v1)
- .github/workflows/perf-bench-matrix.json (added abcorrea_bw_v1 entry)

## tt-forge-models submodule
no change
