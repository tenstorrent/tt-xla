loader_path: third_party.tt_forge_models.configurable_hermes_7b.causal_lm.pytorch.loader
variant_id: ConfigurableHermes_7B
arch: n150
status: DONE_FAIL
test_function: test_configurable_hermes_7b
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 212.1348
pct_of_target: null
roofline_bound: compute
optimization_level: 2
trace_enabled: null
experimental_weight_dtype: "bfp_bf8"
failure_reason: "compiler bug: failed to legalize operation 'ttir.paged_update_cache' during full-model decode after 8 KV-cache recompiles"

# Benchmark added: test_configurable_hermes_7b

## Test
tests/benchmark/test_llms.py::test_configurable_hermes_7b

## Model
- HF name:    vicgalle/ConfigurableHermes-7B
- Loader:     third_party.tt_forge_models.configurable_hermes_7b.causal_lm.pytorch.loader
- Variant:    ConfigurableHermes_7B

## Test config landed
- optimization_level:        2
- trace_enabled:             (default)
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         0:21:45
- Hardware:           n150 (wormhole_b0, n300 card single-chip)

## Failure
The `num_layers=1` run passed (1:19 wall clock). The full-model run (32 decode
steps, batch_size=32, bfp_bf8) triggered 8 KV-cache recompiles due to
`cumulative_length` guard changes, then hit the dynamo `recompile_limit`.
Subsequent compilation of the 9th graph variant failed in the MLIR compiler:

```
loc("scatter.11541"): error: failed to legalize operation 'ttir.paged_update_cache'
Failed to run TTIRToTTNNCommon pipeline
```

The resulting `RuntimeError: Error code: 13` surfaces when the decode loop
tries `next_token_ids_replicated.to("cpu")` after the device is in a bad state
from the compilation failure.

This is a compiler-side bug (`ttir.paged_update_cache` legalization). The test
and loader are correct. The fix belongs in tt-mlir, not here.

Additionally, the llm_benchmark.py harness called `model_loader.get_weight_dtype_config_path()`
unconditionally; a general `hasattr` guard was added to make the harness robust
for loaders that don't implement this optional method.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_configurable_hermes_7b_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: N/A (test failed before sampling)

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
- total_flops:             402259970304
- breakdown.matmul:        402259970304
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        595
- memory_bytes: 2378

### KV cache
- count:        8388608
- memory_bytes: 16777216
- memory_gb:    0.015625

### Params
- count:                  480276676
- effective_count:        349196484
- memory_bytes:           633193740
- memory_gb:              0.5897076241672039
- effective_memory_bytes: 371033356
- effective_memory_gb:    0.3455517403781414
- embedding_count:        131080192
- embedding_memory_bytes: 262160384

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 212.1348
- top_perf_time_ms:         4.7140
- dram_time_ms:             1.2804
- compute_time_ms_lofi:     1.5713
- compute_time_ms_hifi2:    3.1427
- compute_time_ms_hifi3:    4.7140
- compute_time_ms_hifi4:    6.2853

## Files changed
- tests/benchmark/test_llms.py (added test_configurable_hermes_7b)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
