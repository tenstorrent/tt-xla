loader_path: third_party.tt_forge_models.aaryank_qwen3_5_9b_gguf.causal_lm.pytorch.loader
variant_id: 9B_GGUF
arch: n150
status: DONE_FAIL
test_function: test_aaryank_qwen3_5_9b_gguf
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
failure_reason: "device stuck: Timeout waiting for Ethernet core service remote IO request flush on n300 Wormhole board; R chip ETH firmware unresponsive after concurrent test collision — requires power cycle to recover"

# Benchmark added: test_aaryank_qwen3_5_9b_gguf

## Test
tests/benchmark/test_llms.py::test_aaryank_qwen3_5_9b_gguf

## Model
- HF name:    AaryanK/Qwen3.5-9B-GGUF
- Loader:     third_party.tt_forge_models.aaryank_qwen3_5_9b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_9B_GGUF (= "9B_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (device stuck)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n300 Wormhole (n150-equivalent single-chip)

## Device failure details
The test function was successfully added but could not be validated because the
n300 Wormhole device entered an unrecoverable state during bring-up attempts.

Root cause: Two concurrent test processes (PIDs 32994 and 33169) both attempted
to initialize the n300's ETH firmware simultaneously at ~13:32 UTC. This
corrupted the R chip's ETH core firmware state:
- First test:  `ETH training timed out after 900000 ms, on eth core 25, 16`
- Subsequent:  `Timeout waiting for Ethernet core service remote IO request flush`

Recovery attempts made:
- `tt-smi --reset 0` (multiple times) → device still broken
- `tt-smi --eth_train_skip --reset all` → device still broken
- `tt-smi -r 0` with full ETH training (completed in ~1.5 min) → device still broken
- `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1` env var → no effect
- `tt-smi --use_luwen --reset all` → timed out (luwen library also stuck)

All tests on the machine are broken (verified with test_device_initialization.py).
Device requires physical power cycle or kernel module reload (`rmmod`/`modprobe
tenstorrent`) which requires root access not available in this environment.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test did not execute
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        wormhole_b0 (n150)
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
- tests/benchmark/test_llms.py (added test_aaryank_qwen3_5_9b_gguf function)
- .github/workflows/perf-bench-matrix.json (added aaryank_qwen3_5_9b_gguf entry)

## tt-forge-models submodule
no change
