loader_path: third_party.tt_forge_models.jdchang_llama3_small.causal_lm.pytorch.loader
variant_id: llama3_small
arch: p150
status: DONE_FAIL
test_function: test_jdchang_llama3_small
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 2293.5890
pct_of_target: null
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "compiler bug: TT_FATAL: Invalid arguments to reshape during decode warmup run (reshape volume mismatch in TTNN runtime)"

# Benchmark added: test_jdchang_llama3_small

## Test
tests/benchmark/test_llms.py::test_jdchang_llama3_small

## Model
- HF name:    jdchang/llama3-small
- Loader:     third_party.tt_forge_models.jdchang_llama3_small.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA3_SMALL

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  null (test failed)
- TTFT (ms):          null (test failed)
- Prefill PCC:        null (test failed)
- First decode PCC:   null (test failed)
- Wall clock:         ~31s (failed during warmup)
- Hardware:           p150

## Failure details
The test failed with a compiler/runtime error during the decode warmup run:
- `TT_FATAL: Invalid arguments to reshape` in TTNN's reshape_common.cpp:50 (new_volume != old_volume assertion failure)
- `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`

This is a compiler/runtime bug outside the scope of this skill. The infrastructure fix applied (guarding `get_weight_dtype_config_path` with `hasattr` in `llm_benchmark.py`) is general and addresses a missing method on older loaders.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_jdchang_llama3_small_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: N/A (test failed before decode)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           110
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  880000000000000
- hifi2: 440000000000000
- hifi3: 293333333333333
- hifi4: 220000000000000

### Compute
- total_flops:             127892717856
- breakdown.matmul:        127892717856
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        594
- memory_bytes: 2376

### KV cache
- count:        2097152
- memory_bytes: 4194304
- memory_gb:    0.00390625

### Params
- count:                  176687755
- effective_count:        111020683
- memory_bytes:           249296424
- memory_gb:              0.23217538744211197
- effective_memory_bytes: 117962280
- effective_memory_gb:    0.10986093431711197
- embedding_count:        65667072
- embedding_memory_bytes: 131334144

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 2293.5890
- top_perf_time_ms:         0.4360
- dram_time_ms:             0.2278
- compute_time_ms_lofi:     0.1453
- compute_time_ms_hifi2:    0.2907
- compute_time_ms_hifi3:    0.4360
- compute_time_ms_hifi4:    0.5813

## Files changed
- tests/benchmark/test_llms.py (added test_jdchang_llama3_small)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr for older loaders)

## tt-forge-models submodule
no change (submodule at 3feb2058cf)
