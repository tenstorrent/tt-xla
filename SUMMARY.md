loader_path: third_party.tt_forge_models.biomistral_7b_dare.causal_lm.pytorch.loader
variant_id: biomistral_7b_dare
arch: p150
status: DONE_FAIL
test_function: test_biomistral_7b_dare
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 44.85
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "sliding window attention (StaticSlidingWindowLayer) generates ttir.paged_update_cache which fails TTIRToTTNNCommon on p150; TTStaticSlidingWindowLayer workaround hits RISC-V kernel compile error for seq_len=18 on blackhole"

# BioMistral-7B-DARE Benchmark — DONE_FAIL

## Model
- Loader: `third_party.tt_forge_models.biomistral_7b_dare.causal_lm.pytorch.loader`
- Variant: `biomistral_7b_dare` (HF: `BioMistral/BioMistral-7B-DARE`)
- Architecture: MistralForCausalLM (Mistral-7B fine-tune), 32 layers, 8 KV heads, sliding_window=4096
- Hardware: p150 (blackhole), single chip

## Failure (Dual Compiler Bug)

### Primary failure: paged_update_cache not supported

The full-model test (1:35:05) failed with:

```
loc("scatter.11541"): error: failed to legalize operation 'ttir.paged_update_cache'
Failed to run TTIRToTTNNCommon pipeline
RuntimeError: Error code: 13
```

**Root cause**: BioMistral uses `StaticSlidingWindowLayer` for all 32 KV cache layers
(since Mistral has `sliding_window=4096`). The `StaticSlidingWindowLayer.update()` method
uses `torch.tensor([-1])` tensor indexing for cache scatter, which the TTNN compiler converts
to `ttir.paged_update_cache`. The `PagedUpdateCacheOpConversionPattern` in TTIRToTTNN then
fails because the cache tensor has more than one user in the compiled graph.

This triggers after 8 recompilations (torch._dynamo's `recompile_limit`) of the decode step,
leaving the device tensor in an invalid state (Error code 13).

### Alternative approach: TTStaticSlidingWindowLayer also fails

Replacing `StaticSlidingWindowLayer` with `TTStaticSlidingWindowLayer`
(from `tt_torch.transformers_overrides`) avoids the scatter issue but causes a different error:

```
brisc compile failure: no matching function for call to 'get_noc_addr(const uint32_t&,
const TensorAccessor<DistributionSpec<2, 5, ..., ArrayStaticWrapper<long unsigned int, 18, 1>, ...>>)'
```

This is a tt-metal `writer_interleaved_rm_no_bcast` kernel compilation failure on blackhole
for the non-standard tensor shape with sequence length 18 (the tokenized prompt length).
The `get_noc_addr` overload for this specific `TensorAccessor` template specialization is
missing in the blackhole RISC-V firmware.

## Infrastructure changes included

1. `tests/benchmark/benchmarks/llm_benchmark.py`: Added `hasattr(model_loader, "get_weight_dtype_config_path")` guard (matches pattern in `dynamic_torch_model_tester.py`).
2. `python_package/tt_torch/transformers_overrides.py`: Added `override_mistral_sliding_window_causal_mask()` and `override_all_sliding_window_causal_masks()` for future use when tt-metal blackhole fixes the kernel issue.

## Roofline (decode graph, from perf metrics collected during failed run)

| Metric | Value |
|---|---|
| arch | blackhole (p150) |
| bound | dram |
| top_perf_samples_per_sec | 44.85 |
| top_perf_time_ms | 22.29 |
| params | 7.24B |
| kv_cache_gb | 0.5 |

## Measured performance

Not available — test failed before completing PCC verification or recording throughput.

## Configuration at failure

| Setting | Value |
|---|---|
| optimization_level | 2 (DEFAULT) |
| trace_enabled | true (DEFAULT) |
| experimental_weight_dtype | bfp_bf8 (DEFAULT) |
| batch_size | 32 |
