loader_path: third_party.tt_forge_models.document_validation_qwen2_5_vl_simple_v2_i1_gguf.causal_lm.pytorch.loader
variant_id: Simple_V2_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_document_validation_qwen2_5_vl_simple_v2_i1_gguf
samples_per_second: 4.174187238941504
ttft_ms: 1241.004509
prefill_pcc: 0.992996
first_decode_pcc: 0.997859
top_perf_samples_per_sec: 46.0471
pct_of_target: 9.1
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark: test_document_validation_qwen2_5_vl_simple_v2_i1_gguf (p150)

## Model
- **Model**: mradermacher Document-Validation-Qwen2.5-VL-Simple-V2-i1-GGUF
- **Variant**: Simple_V2_i1_GGUF
- **HF name**: hienphantt161/Document-Validation-Qwen2.5-VL-Simple-V2
- **Loader**: `third_party.tt_forge_models.document_validation_qwen2_5_vl_simple_v2_i1_gguf.causal_lm.pytorch.loader`
- **Arch**: p150 (Blackhole, device ID 0xb140)
- **Parameters**: ~7.6B (7.07B effective at bfp_bf8)

## Test Function Added

`tests/benchmark/test_llms.py::test_document_validation_qwen2_5_vl_simple_v2_i1_gguf`

```python
def test_document_validation_qwen2_5_vl_simple_v2_i1_gguf(
    output_file, num_layers, request, accuracy_testing,
    batch_size, max_output_tokens, decode_only, optimization_level,
):
    from third_party.tt_forge_models.document_validation_qwen2_5_vl_simple_v2_i1_gguf.causal_lm.pytorch.loader import (
        ModelLoader, ModelVariant,
    )
    variant = ModelVariant.DOCUMENT_VALIDATION_QWEN2_5_VL_SIMPLE_V2_I1_GGUF
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
        decode_only=decode_only,
        # optimization_level=1 fails PCC (0.690); optimization_level=2 crashes during MLIR compilation
        optimization_level=(
            optimization_level if optimization_level is not None else 0
        ),
    )
```

## Optimization Investigation (num_layers=1, batch_size=32)

Note: The loader has a known bug where `num_layers` parameter is ineffective
(it assigns `full_model.model.language_model` after creating a 1-layer config,
so the full 28-layer model runs regardless of `--num-layers 1`). All runs
used the full 7B Qwen2 language model.

| opt_level | trace | Result | PCC (prefill/decode) | sps | TTFT (ms) |
|-----------|-------|--------|----------------------|-----|-----------|
| 0 | False | ✅ PASS | 0.993 / 0.998 | 4.15 | 620 |
| 0 | True  | ✅ PASS | 0.993 / 0.998 | 4.17 | 1241 |
| 1 | False | ❌ FAIL | 0.690 / N/A | 19.9 | 168 |
| 2 | False | ❌ CRASH | — | — | — |

- **opt_level=1**: Prefill PCC drops to 0.690 (well below required 0.94). This is a
  real numerical accuracy issue introduced by the optimization passes — NOT lowered.
- **opt_level=2**: Process crashes during MLIR compilation with no error output
  (likely an assertion failure or segfault in the compiler).
- **opt_level=0**: Passes PCC at both prefill (0.993) and first decode (0.998).

Also fixed a general bug in `tests/benchmark/benchmarks/llm_benchmark.py` where
`get_weight_dtype_config_path()` was called unconditionally on the model loader,
causing `AttributeError` for loaders that don't have this method. Changed `else:` to
`elif hasattr(model_loader, "get_weight_dtype_config_path"):`.

## Performance Results (opt_level=0, trace=True)

| Metric | Value |
|--------|-------|
| Sample per second (measured) | 4.17 |
| TTFT (ms) | 1241 |
| Prefill PCC | 0.992996 |
| First decode PCC | 0.997859 |
| Roofline ceiling (sps) | 46.05 |
| % of roofline target | 9.1% |
| Roofline bound | DRAM |

The 9.1% roofline efficiency is expected: `optimization_level=0` does not
place weights in SRAM (all accesses from DRAM), so performance is far below
the DRAM bandwidth ceiling. The higher opt_levels that could reach >40 sps
fail with PCC regression (opt_level=1) or crash (opt_level=2).

## Harness Bug Fix

`tests/benchmark/benchmarks/llm_benchmark.py` line ~478: changed
```python
else:
    weight_dtype_config = model_loader.get_weight_dtype_config_path()
```
to
```python
elif hasattr(model_loader, "get_weight_dtype_config_path"):
    weight_dtype_config = model_loader.get_weight_dtype_config_path()
```
This affects any model loader that does not have `get_weight_dtype_config_path`
(e.g. this loader which extracts the LM from a conditional generation model).
