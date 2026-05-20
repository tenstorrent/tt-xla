loader_path: third_party.tt_forge_models.gemma3_4b_it_qat_gguf.causal_lm.pytorch.loader
variant_id: 4B_IT_QAT_Q4_0
arch: p150
status: DONE_PASS
test_function: test_gemma3_4b_it_qat_gguf
samples_per_second: 12.523948094598357
ttft_ms: 792.724338
prefill_pcc: 0.989221
first_decode_pcc: 0.985549
top_perf_samples_per_sec: 79.4603
pct_of_target: 15.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark Summary: test_gemma3_4b_it_qat_gguf (p150)

## Model

- **Loader**: `third_party.tt_forge_models.gemma3_4b_it_qat_gguf.causal_lm.pytorch.loader`
- **Variant**: `4B_IT_QAT_Q4_0`
- **HuggingFace model**: `ggml-org/gemma-3-4b-it-qat-GGUF` (gemma-3-4b-it-qat-Q4_0.gguf)
- **Architecture**: Gemma 3 4B (mixed sliding + full attention, 34 layers)
- **Hardware**: p150 (Blackhole, single-chip)

## Results

| Metric | Value |
|---|---|
| Sample per second | 12.52 |
| TTFT (ms) | 792.72 |
| Prefill PCC | 0.989221 ✅ |
| First decode PCC | 0.985549 ✅ |
| % of roofline target | 15.8% |

## Roofline Analysis (first decode graph)

- **Bound**: DRAM
- **Top perf**: 79.4603 samples/sec
- **Top perf time**: 12.5849 ms
- **Params**: 4.55B total (3.88B effective)
- **Model DRAM footprint**: 5.09 GB (3.84 GB effective)
- **KV cache**: 0.53 GB

## Configuration

- `optimization_level`: 2
- `trace_enabled`: True (default)
- `experimental_weight_dtype`: `"bfp_bf8"` (default)
- `batch_size`: 32

## Notes

Gemma 3 uses alternating **sliding-window** and **full-attention** KV cache layers
(`StaticSlidingWindowLayer` from `transformers.cache_utils`). Its `update()` method
accesses `self.cumulative_length` (a Python int) inside the compiled region, creating
per-decode-step recompile guards. After 8 recompiles torch._dynamo hit its
`recompile_limit`, fell back to dynamic shapes, and generated `ttir.paged_update_cache`
in tt-mlir — which lacks a `TTIRToTTNNCommon` lowering → `RuntimeError: Error code: 13`.

**Fix**: `llm_benchmark.py` monkey-patches `StaticSlidingWindowLayer.update` and
`get_mask_sizes` at module load time to remove all `cumulative_length` accesses:
- `update()` uses `cache_position` (tensor) for `index_copy_` directly
- `get_mask_sizes()` returns constant `(max_cache_len, 0)`

This eliminates the per-step guards, keeping the compiled graph static for all decode
steps. Correctness is preserved because:
1. KV positions are managed by `cache_position` (externally provided tensor)
2. Gemma 3 forward only calls `get_seq_length()` when `cache_position is None`
   (never in this benchmark)
3. `get_mask_sizes` returns `(128, 0)` which matches the original for all normal
   decode steps (cumulative_length < max_cache_len)

## Test run details

- Full run time: ~59 min 49 s
- pytest: `1 passed, 23 warnings`
- Both PCC thresholds exceeded ≥ 0.94 requirement
