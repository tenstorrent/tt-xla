loader_path: third_party.tt_forge_models.carbon_beagle.causal_lm.pytorch.loader
variant_id: CarbonBeagle_11B_truthy
arch: p150
status: DONE_FAIL
test_function: test_carbon_beagle_11b_truthy
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "compiler bug: failed to legalize operation 'ttir.paged_update_cache' after torch.dynamo recompile limit (8) reached during warmup decode steps; RuntimeError: Error code: 13"

# Benchmark added: test_carbon_beagle_11b_truthy

## Test
tests/benchmark/test_llms.py::test_carbon_beagle_11b_truthy

## Model
- HF name:    vicgalle/CarbonBeagle-11B-truthy
- Loader:     third_party.tt_forge_models.carbon_beagle.causal_lm.pytorch.loader
- Variant:    ModelVariant.CARBON_BEAGLE_11B_TRUTHY

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A — test failed
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         ~12m (failed)
- Hardware:           p150

## Failure Details

The test fails consistently on the full model (all optimization levels 0/1/2) with:

```
torch._dynamo hit config.recompile_limit (8)
  last reason: past_key_values.layers[0].cumulative_length == 24 # is_full in cache_utils.py
loc("scatter.17125"): error: failed to legalize operation 'ttir.paged_update_cache'
Failed to run TTIRToTTNNCommon pipeline
RuntimeError: Error code: 13
```

Root cause: The Mistral-based CarbonBeagle-11B uses transformers' paged KV cache. Each decode step introduces a unique `cumulative_length` value, causing a new torch.dynamo trace. After 8 recompiles (the default recompile_limit), dynamo falls back to eager mode for subsequent decode lengths. In that fallback path, the `paged_update_cache` op is generated but the TTIRToTTNNCommon lowering pipeline does not support it, causing compilation failure and subsequently a device error (Error code: 13) when trying to move tokens to CPU.

The 1-layer test with --max-output-tokens 3 passes because only 3 recompilations occur (never hitting the limit).

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_carbon_beagle_11b_truthy_perf_metrics_0.json (1-layer only, not representative)
Achieved vs top_perf_samples_per_sec: N/A (full model test failed)

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

### Compute (1-layer only)
- total_flops:             403458492672
- breakdown.matmul:        403458492672
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs (1-layer only)
- count:        595
- memory_bytes: 2378

### KV cache (1-layer only)
- count:        8388608
- memory_bytes: 16777216
- memory_gb:    0.015625

### Params (1-layer only)
- count:                  480260295
- effective_count:        349188295
- memory_bytes:           633168664
- memory_gb:              0.590
- effective_memory_bytes: 371024664
- effective_memory_gb:    0.346
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline (1-layer only)
- bound:                    compute
- top_perf_samples_per_sec: 727.0471
- top_perf_time_ms:         1.3754
- dram_time_ms:             0.7202
- compute_time_ms_lofi:     0.4585
- compute_time_ms_hifi2:    0.9170
- compute_time_ms_hifi3:    1.3754
- compute_time_ms_hifi4:    1.8339

## Files changed
- tests/benchmark/test_llms.py (added test_carbon_beagle_11b_truthy)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- SUMMARY.md

## tt-forge-models submodule
no change
