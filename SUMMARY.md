loader_path: third_party.tt_forge_models.heretic_3b_i1_gguf.causal_lm.pytorch.loader
variant_id: heretic_3B_I1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_heretic_3b_i1_gguf
samples_per_second: 37.303338
ttft_ms: 236.173922
prefill_pcc: 0.991956
first_decode_pcc: 0.553365
top_perf_samples_per_sec: 90.8682
pct_of_target: 41.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: First decode PCC fails at full model depth across all tested configurations (opt=1+bfp_bf8: 0.614377, opt=2+no-bfp: 0.553365; both < 0.94 required). 1-layer test passes (PCC 0.998721), suggesting KV cache accumulation error at full depth (32 layers) on p150.

---

# Benchmark: test_heretic_3b_i1_gguf (p150)

## Summary

**Status: DONE_FAIL** — Decode PCC consistently fails at full model depth.

| Configuration | Prefill PCC | Decode PCC | Throughput | TTFT |
|---|---|---|---|---|
| opt=1 + bfp_bf8 | 0.995402 ✓ | 0.614377 ✗ | 36.64 sps | 244ms |
| opt=2 + no-bfp | 0.991956 ✓ | 0.553365 ✗ | 37.30 sps | 236ms |
| 1-layer (opt=2) | 0.998584 ✓ | 0.998721 ✓ | 272.94 sps | 37ms |

## Failure Analysis

The 1-layer test passes with excellent PCC (0.999), but at full model depth (32 layers):
- Prefill PCC passes (~0.99) 
- **First decode PCC fails catastrophically (~0.55-0.61)**

The failure pattern (passes at 1 layer, fails at 32 layers) suggests KV cache state divergence accumulates across layers during device prefill. Even though the prefill logits are correct (PCC ~0.99), the intermediate KV cache values differ enough that the first autoregressive decode step produces completely different output compared to the CPU golden.

This appears to be a genuine numerical issue with GGUF Q4_K_M quantized models on p150, not a configuration error.

## Model

- **Loader**: `third_party.tt_forge_models.heretic_3b_i1_gguf.causal_lm.pytorch.loader`
- **Variant**: `heretic_3B_I1_GGUF` (mradermacher/heretic-3b-i1-GGUF, file: heretic-3b.i1-Q4_K_M.gguf)
- **Parameters**: 3.5B effective (3.93B total), 4.3 GB at bfp_bf8

## Roofline Analysis (Decode Graph)

| Metric | Value |
|---|---|
| Architecture | blackhole (p150) |
| DRAM bandwidth | 512 GB/s |
| Roofline bound | DRAM |
| Top perf (sps) | 90.87 |
| Measured (sps) | 37.30 |
| % of target | 41.1% |
| Top perf time (ms) | 11.00 |
| DRAM time (ms) | 7.34 |

## Final Test Configuration

```python
def test_heretic_3b_i1_gguf(
    output_file, num_layers, request, accuracy_testing,
    batch_size, max_output_tokens, decode_only, optimization_level,
):
    from third_party.tt_forge_models.heretic_3b_i1_gguf.causal_lm.pytorch.loader import (
        ModelLoader, ModelVariant,
    )
    variant = ModelVariant.HERETIC_3B_I1_GGUF
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant,
        output_file=output_file, num_layers=num_layers,
        request=request, accuracy_testing=accuracy_testing,
        batch_size=batch_size, max_output_tokens=max_output_tokens,
        decode_only=decode_only,
        optimization_level=(
            optimization_level if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )
```

Settings used: `optimization_level=2`, `trace_enabled=True`, `experimental_weight_dtype="bfp_bf8"` (all defaults).
