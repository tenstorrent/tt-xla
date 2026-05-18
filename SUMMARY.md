loader_path: third_party.tt_forge_models.configurable_hermes_7b.causal_lm.pytorch.loader
variant_id: ConfigurableHermes_7B
arch: p150
status: DONE_FAIL
test_function: test_configurable_hermes_7b
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 44.8550
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: null
experimental_weight_dtype: bfp_bf8
failure_reason: "torch._dynamo recompile_limit (8) hit due to StaticCache.cumulative_length guard in transformers cache_utils.py; fallback to eager mode triggers unsupported ttir.paged_update_cache MLIR op → RuntimeError: Error code: 13 after 8 decode steps"

# Benchmark added: test_configurable_hermes_7b

## Test
tests/benchmark/test_llms.py::test_configurable_hermes_7b

## Model
- HF name:    vicgalle/ConfigurableHermes-7B
- Loader:     third_party.tt_forge_models.configurable_hermes_7b.causal_lm.pytorch.loader
- Variant:    ConfigurableHermes_7B

## Test config landed
- optimization_level:        2
- trace_enabled:             true (DEFAULT_TRACE_ENABLED)
- experimental_weight_dtype: bfp_bf8 (DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  null (test fails with default settings)
- TTFT (ms):          null
- Prefill PCC:        null
- First decode PCC:   null
- Wall clock:         N/A
- Hardware:           p150

## Failure Analysis
The test fails with `RuntimeError: Error code: 13` after 8 decode steps. Root cause:

1. `torch._dynamo` recompiles the decode graph on every step because
   `past_key_values.layers[0].cumulative_length` is used as a guard condition
   in `transformers/cache_utils.py:470` (`get_mask_sizes`). Each new value of
   `cumulative_length` triggers a recompile.

2. After 8 recompilations (dynamo's `config.recompile_limit`), dynamo falls back
   to eager (no-compile) mode.

3. The eager-mode graph contains `ttir.paged_update_cache` which the TT-MLIR
   `TTIRToTTNNCommon` pipeline cannot legalize:
   ```
   loc("scatter.722"): error: failed to legalize operation 'ttir.paged_update_cache'
   Failed to run TTIRToTTNNCommon pipeline
   ```

4. This results in `RuntimeError: Error code: 13`.

Diagnostic note: with `--max-output-tokens 3` (CLI override), the test passes
since only 3 decode steps are run (below the 8-recompile limit). Measured numbers
from that diagnostic run: samples_per_second=0.026 (heavily impacted by
per-step recompilation), TTFT=148s (compilation included), prefill_pcc=0.999537,
first_decode_pcc=0.994100. PCC is good; the model is numerically correct.

Fix required: mark `cumulative_length` as static in dynamo (changes to
transformers or the TT-XLA benchmark infrastructure), OR implement
`ttir.paged_update_cache` in the TT-MLIR TTNNCommon pipeline.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_configurable_hermes_7b_perf_metrics_9.json
Achieved vs top_perf_samples_per_sec: N/A (full test fails)

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
- total_flops:             455065731200
- breakdown.matmul:        455065731200
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
- count:                  7241748676
- effective_count:        7110668484
- memory_bytes:           7817495820
- memory_gb:              7.280610334128141
- effective_memory_bytes: 7555335436
- effective_memory_gb:    7.036454450339079
- embedding_count:        131080192
- embedding_memory_bytes: 262160384

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 44.8550
- top_perf_time_ms:         22.2940
- dram_time_ms:             14.8627
- compute_time_ms_lofi:     0.5171
- compute_time_ms_hifi2:    1.0342
- compute_time_ms_hifi3:    1.5514
- compute_time_ms_hifi4:    2.0685

## Files changed
- tests/benchmark/test_llms.py (added test_configurable_hermes_7b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: graceful handling when loader lacks get_weight_dtype_config_path method)

## tt-forge-models submodule
no change
