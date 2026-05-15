loader_path: third_party.tt_forge_models.qwen_1_5.causal_lm.pytorch.loader
variant_id: QWEN_1_5_7B
arch: n150
status: DONE_FAIL
test_function: test_qwen_1_5_7b
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: false
experimental_weight_dtype: null
failure_reason: "Device OOM — model (7.72B params bfp8=7.3GB + KV cache 2GB) is at the 12GB n150 DRAM capacity limit at batch_size=32; opt_level=2+trace=True OOMs during PCC benchmark; opt_level=1+trace=True OOMs at allocation; opt_level=2+trace=False untested (device became unrecoverable after repeated OOMs). Needs re-run after system reboot."

# Benchmark added: test_qwen_1_5_7b

## Test
tests/benchmark/test_llms.py::test_qwen_1_5_7b

## Model
- HF name:    Qwen/Qwen1.5-7B
- Loader:     third_party.tt_forge_models.qwen_1_5.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_1_5_7B (restored from commit 734d543c5f — variant existed but was absent from current submodule HEAD 93218a34fc)

## Test config landed
- optimization_level:        2 (DEFAULT)
- trace_enabled:             false
- experimental_weight_dtype: none (using weight_dtype_overrides={"default": "bfp_bf8"} instead)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  — (not obtained; test OOM'd before completing)
- TTFT (ms):          — (not obtained)
- Prefill PCC:        — (not obtained)
- First decode PCC:   — (not obtained)
- Wall clock:         N/A
- Hardware:           n300 L (wormhole_b0, single-chip mode)

## OOM debug history (all on n150, batch_size=32)

| Config | Result |
|--------|--------|
| opt_level=2, trace=True, experimental_weight_dtype=bfp_bf8 | Compiled OK, ran PCC benchmark, then TT_FATAL @ bank_manager.cpp:462 |
| opt_level=1, trace=True, experimental_weight_dtype=bfp_bf8 | DRAM OOM: 311 MB alloc fails with 12 banks × 28 MB free (fragmented) |
| opt_level=1, trace=True, weight_dtype_overrides=bfp_bf8 | Same DRAM OOM (same allocation sizes — different API path, identical HW result) |
| opt_level=2, trace=False, weight_dtype_overrides=bfp_bf8 | Device was already in bad state; crashed at init. Untested on clean device. |

The best candidate is opt_level=2 + trace=False + bfp_bf8. Disabling trace removes the trace capture buffer
which is the likely cause of the final OOM during the PCC benchmark iterations.

## TODO: after system reboot
1. Re-run: `pytest -svv tests/benchmark/test_llms.py::test_qwen_1_5_7b`
2. If DRAM OOM persists with trace=False: file an issue and update the `trace_enabled=False` comment with the URL.
3. If PASSED: update this SUMMARY.md with measured numbers and change status to DONE_PASS.

## Prefill roofline (from first decode run, full model — file 0)
Source JSON: tt_xla_qwen_1_5_7b_perf_metrics_0.json (decode metrics overwritten by later runs)
Note: This is the PREFILL graph roofline; full-model decode metrics were overwritten.

### Params (at bfp8)
- count:                  7721324608
- effective_count:        7098994752
- memory_bytes (bf16):    14.38 GB
- effective_memory_bytes_bfp8: 7.32 GB
- embedding_count:        622329856
- embedding_memory_bytes: 1244659712

### KV cache
- count:        1073741824
- memory_bytes: 2147483648 (2 GB bf16)
- memory_bytes_bfp8: 1107296256 (1.03 GB at bfp8)

### Memory footprint at bfp8 (why this is tight on 12 GB n150)
- bfp8 weights:  7.32 GB
- bfp8 KV cache: 1.03 GB
- Subtotal:      8.35 GB
- Execution buffers + activations: ~3–4 GB (observed: 12 GB filled at OOM)

### Prefill roofline
- bound:                    compute
- top_perf_samples_per_sec: 11.05
- top_perf_time_sec:        0.0905 (90.5 ms)
- dram_time_sec:            0.0293 (29.3 ms)
- compute_time_sec_lofi:    0.0302
- compute_time_sec_hifi2:   0.0603
- compute_time_sec_hifi3:   0.0905
- compute_time_sec_hifi4:   0.1207

### Compute
- total_flops:             7723202644096
- breakdown.matmul:        5970642077824
- breakdown.linear:        1752560566272
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

## Files changed
- tests/benchmark/test_llms.py (added test_qwen_1_5_7b)
- .github/workflows/perf-bench-matrix.json (added qwen_1_5_7b entry)
- third_party/tt_forge_models/qwen_1_5/causal_lm/pytorch/loader.py (restored QWEN_1_5_7B variant)

## tt-forge-models submodule
93218a34fc → (same HEAD, loader.py modified in working tree)
Restored QWEN_1_5_7B = "7B" → Qwen/Qwen1.5-7B variant that was present in commit
734d543c5f but absent from current submodule HEAD 93218a34fc.
