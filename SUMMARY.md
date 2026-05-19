loader_path: third_party.tt_forge_models.m7.causal_lm.pytorch.loader
variant_id: 7B
arch: p150
status: DONE_FAIL
test_function: test_m7_7b
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 44.8551
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: false
experimental_weight_dtype: bfp_bf8
failure_reason: "RuntimeError: Error code 13 on next_token_ids_replicated.to('cpu') during perf benchmark; root cause is ttir.paged_update_cache op failing to legalize in TTIRToTTNNCommon pipeline (Mistral sliding-window KV-cache scatter op not supported in TTNN backend on p150)"

# M7 7B Benchmark — DONE_FAIL

## Model
- **Loader**: `third_party.tt_forge_models.m7.causal_lm.pytorch.loader`
- **Variant**: `7B` (liminerity/M7-7b)
- **Architecture**: Mistral 7B with sliding-window attention
- **Parameters**: 7.24B (7,241,732,160), ~13.49 GB weights
- **KV Cache**: 0.5 GB

## Hardware
- **Device**: Blackhole p300c → arch `p150`
- **Worker grid**: 110 cores
- **DRAM bandwidth**: 512 GB/s

## Roofline (from compile-time analysis)
- **Decode graph**: DRAM-bound
- **top_perf_samples_per_sec**: 44.8551
- **top_perf_time_ms**: 22.2940 ms

## Test Configuration
- `optimization_level`: 2
- `trace_enabled`: False (trace compilation hangs indefinitely; see comment in test)
- `experimental_weight_dtype`: bfp_bf8 (default)
- `batch_size`: 32

## Failure Details

The test consistently fails with `RuntimeError: Error code: 13` during the performance
benchmark phase. The full sequence of events:

1. **Prefill compilation** (~5 min): Compiles and runs successfully; writes
   `tt_xla_m7_7b_perf_metrics_0.json`.
2. **Decode compilation** (~7 min): MLIR compilation fails with:
   ```
   loc("scatter.11541"): error: failed to legalize operation 'ttir.paged_update_cache'
   Failed to run TTIRToTTNNCommon pipeline
   ```
   The test falls back to a non-paged-cache execution path.
3. **Warmup** (16 tokens): Completes successfully using the fallback path.
4. **Performance benchmark**: Fails immediately at:
   ```python
   decoded = tokenizer.batch_decode(next_token_ids_replicated.to("cpu"))
   RuntimeError: Error code: 13
   ```

The `ttir.paged_update_cache` op is generated because M7 7B (Mistral architecture) uses
sliding-window attention, which the MLIR compiler represents via scatter-based KV-cache
updates. This op is not supported by the TTNN backend on p150 (Blackhole).

With `trace_enabled=True`, the trace compilation hangs indefinitely (45+ minutes with no
output) in the decode graph compilation phase.

## Single-Layer Pass (num_layers=1)

The model passes at `--num-layers 1` (38.61s total):
- Sample per second: 0.5413
- TTFT: 4581 ms
- Prefill PCC: 0.999106
- First decode PCC: 0.999428

Full 32-layer model fails at the `paged_update_cache` compilation step.

## Changes Made

1. **`tests/benchmark/test_llms.py`**: Added `test_m7_7b` with `trace_enabled=False`
   (trace hangs; comment references paged_update_cache issue) and a comment before the
   function noting the trace issue.
2. **`tests/benchmark/benchmarks/llm_benchmark.py`**: Added `hasattr` guard for
   `get_weight_dtype_config_path` to support ModelLoaders that don't implement this
   optional method.
3. **`third_party/tt_forge_models`**: Submodule updated to include `m7/causal_lm/pytorch/loader.py`.
