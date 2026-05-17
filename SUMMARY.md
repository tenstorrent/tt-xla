loader_path: third_party.tt_forge_models.bielik.causal_lm.pytorch.loader
variant_id: 7B_Instruct_v0.1
arch: n150
status: DONE_FAIL
test_function: test_bielik_7b_instruct_v0_1
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 25.231
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "compiler bug: ttir.paged_update_cache fails to legalize in TTIRToTTNNCommon pipeline, triggered by torch._dynamo hitting recompile_limit(8) during decode warmup; affects both trace_enabled=True and trace_enabled=False"

# Benchmark added: test_bielik_7b_instruct_v0_1

## Test
tests/benchmark/test_llms.py::test_bielik_7b_instruct_v0_1

## Model
- HF name:    speakleash/Bielik-7B-Instruct-v0.1
- Loader:     third_party.tt_forge_models.bielik.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIELIK_7B_INSTRUCT_V0_1

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  null (test failed)
- TTFT (ms):          null
- Prefill PCC:        null
- First decode PCC:   null
- Wall clock:         ~21:25 (failed)
- Hardware:           n150 (wormhole_b0, n300 board single-chip)

## Failure Analysis
The full model run fails consistently regardless of `trace_enabled`. Root cause:

```
loc("scatter.11541"): error: failed to legalize operation 'ttir.paged_update_cache'
module_builder.cc:1172   ERR| Failed to run TTIRToTTNNCommon pipeline
```

This error is triggered when `torch._dynamo` hits `config.recompile_limit (8)` during
the decode warmup loop. The `paged_update_cache` operation (Mistral-style rolling-buffer
KV cache) is not supported by the current TT MLIR compiler. Each decode step changes
`past_key_values.layers[0].cumulative_length`, causing dynamo to recompile the forward
function. After 8 recompilations, the 9th triggers the `paged_update_cache` legalization
failure, leaving the device in an error state (RuntimeError: Error code: 13).

The 1-layer test (`--num-layers 1 --max-output-tokens 3`) passes because with only 3
output tokens the recompile limit is not reached.

Infrastructure fix included: `tests/benchmark/benchmarks/llm_benchmark.py` was updated
to guard `get_weight_dtype_config_path()` with `hasattr` (general fix, not bielik-specific).

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bielik_7b_instruct_v0_1_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (test failed)

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             455065206912
- breakdown.matmul:        455065206912
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        34
- memory_bytes: 134

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  7241732160
- effective_count:        7110660160
- memory_bytes:           14483464320
- memory_gb:              13.49
- effective_memory_bytes: 14221320320
- effective_memory_gb:    13.24
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 25.231
- top_perf_time_ms:         39.6338
- dram_time_ms:             26.4225
- compute_time_ms_lofi:     1.7776
- compute_time_ms_hifi2:    3.5552
- compute_time_ms_hifi3:    5.3328
- compute_time_ms_hifi4:    7.1104

## Files changed
- tests/benchmark/test_llms.py (added test_bielik_7b_instruct_v0_1 with # FAILED comment)
- tests/benchmark/benchmarks/llm_benchmark.py (general hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
