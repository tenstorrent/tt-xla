loader_path: third_party.tt_forge_models.gemma3.causal_lm.pytorch.loader
variant_id: 1B_Instruct_awq_int4
arch: p150
status: DONE_FAIL
test_function: test_gemma3_1b_instruct_awq_int4
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
failure_reason: "TT device hardware fault: device 3 (blackhole/p150 at PCI BDF 0000:42:00.0) stuck in 'Query mappings failed' state; soft reset via tt-smi and ioctl unsuccessful; requires host-level PCIe rescan or system restart"

# Benchmark added: test_gemma3_1b_instruct_awq_int4

## Test
tests/benchmark/test_llms.py::test_gemma3_1b_instruct_awq_int4

## Model
- HF name:    gaunernst/gemma-3-1b-it-int4-awq
- Loader:     third_party.tt_forge_models.gemma3.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_1B_IT_AWQ_INT4 ("1B_Instruct_awq_int4")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (device fault)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150 (blackhole device 3 at 0000:42:00.0)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test could not run due to hardware fault)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        blackhole
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
The p150 device (blackhole, PCIe ID 3, BDF 0000:42:00.0, /dev/tenstorrent/3) was
in a hardware fault state at the start of this session. The device initially
returned "Read 0xffffffff over PCIe ID 3" (hung PCIe reads). Soft reset attempts
via `tt-smi -r all --no_reinit` and `tt-smi -r all --use_luwen` appeared to
partially recover the ASIC but left the PCIe BAR mappings inaccessible.
Subsequent ioctl-based resets (USER_RESET, ASIC_RESET, POST_RESET, RESTORE_STATE,
RESET_PCIE_LINK) could not restore QUERY_MAPPINGS functionality (ENODEV).
Recovery requires host-level PCIe rescan (`echo 1 > /sys/bus/pci/rescan`) or
full system restart — neither is possible from within the container.
Last successful benchmark runs were on 2026-05-19 (from perf_metrics_*.json
timestamps in the repo root).

## Files changed
- tests/benchmark/test_llms.py (test function added)
- SUMMARY.md

## tt-forge-models submodule
no change
