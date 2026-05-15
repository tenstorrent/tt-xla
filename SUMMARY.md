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
failure_reason: "Device unrecoverable — n300 ethernet fabric (L-R chip) failed to initialize; tt-smi --reset 0,1 returned 'Failed to get harvesting masks' (wormhole_tt_device.cpp:91). Full system reboot/power-cycle required before re-run."

# Benchmark added: test_qwen_1_5_7b

## Test
tests/benchmark/test_llms.py::test_qwen_1_5_7b

## Model
- HF name:    Qwen/Qwen1.5-7B
- Loader:     third_party.tt_forge_models.qwen_1_5.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_1_5_7B ("7B")

## Test config landed
- optimization_level:        2 (DEFAULT)
- trace_enabled:             false
- experimental_weight_dtype: none (using weight_dtype_overrides={"default": "bfp_bf8"} instead)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  — (not obtained; device unrecoverable before test ran)
- TTFT (ms):          — (not obtained)
- Prefill PCC:        — (not obtained)
- First decode PCC:   — (not obtained)
- Wall clock:         N/A
- Hardware:           n300 L (wormhole_b0, single-chip mode)

## Device failure history (this session)

| Attempt | Result |
|---------|--------|
| pytest --num-layers 1 --max-output-tokens 3 | RuntimeError: Timeout waiting for Ethernet core service remote IO request |
| tt-smi --reset 0 | Reset completed, but error persisted immediately |
| tt-smi --reset 0,1 | Error: Failed to get harvesting masks (wormhole_tt_device.cpp:91) — ethernet fabric broken |

## Prior session OOM history (before device became unrecoverable)

| Config | Result |
|--------|--------|
| opt_level=2, trace=True, experimental_weight_dtype=bfp_bf8 | Compiled OK, ran PCC benchmark, then TT_FATAL @ bank_manager.cpp:462 |
| opt_level=1, trace=True, experimental_weight_dtype=bfp_bf8 | DRAM OOM: 311 MB alloc fails with 12 banks × 28 MB free (fragmented) |
| opt_level=1, trace=True, weight_dtype_overrides=bfp_bf8 | Same DRAM OOM (same allocation sizes — different API path, identical HW result) |

The best candidate to try after reboot is **opt_level=2 + trace=False + weight_dtype_overrides=bfp_bf8**.
This is what the current test function encodes.

## TODO: after system power-cycle
1. Re-run: `pytest -svv tests/benchmark/test_llms.py::test_qwen_1_5_7b`
2. If DRAM OOM persists with trace=False: file an issue and update the `trace_enabled=False` comment with the URL.
3. If PASSED: update this SUMMARY.md with measured numbers and change status to DONE_PASS.

## Submodule note
This branch pins tt-forge-models at 136dbf842d (which contains QWEN_1_5_7B).
The parent branch (odjuricic/ai-benchmark-pipeline) is at 93218a34fc which removed the variant.
This branch must NOT be rebased onto the parent until the variant is restored upstream.

## Prefill roofline (from prior session, first decode run — file 0)
Source JSON: tt_xla_qwen_1_5_7b_perf_metrics_0.json (decode metrics overwritten by later runs)
Note: This is the PREFILL graph roofline; full-model decode metrics were never captured.

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
- Execution buffers + activations: ~3–4 GB (observed: 12 GB filled at OOM with trace=True)

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
- SUMMARY.md (this file)

## tt-forge-models submodule
Branch pins at 136dbf842d (has QWEN_1_5_7B). Parent branch is at 93218a34fc (variant removed).
