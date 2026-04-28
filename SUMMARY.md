# Test Remediation Summary

**Test:** `tests/runner/test_models.py::test_all_models_torch[bartowski_ai_sage_gigachat3_10b_a1_8b_gguf/causal_lm/pytorch-10B_A1_8B_GGUF-single_device-inference]`

**Model:** GigaChat3-10B-A1.8B GGUF (bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF, Q4_K_M quantization)

**Architecture:** DeepSeek V2 MoE (MLA attention, 10B total / 1.8B active params)

## Result: Tier B Compiler Bug

The loader bugs are fixed. The test now fails with a compiler-level shape mismatch during torch.compile FakeTensor propagation for DeepSeek V2 MLA attention. This is a Tier B issue.

## Root Cause Chain

### Loader Bug 1: deepseek2 GGUF architecture not registered (fixed)

The GGUF file declares `general.architecture = deepseek2`. Transformers 5.x does not include `deepseek2` in `GGUF_SUPPORTED_ARCHITECTURES` or `GGUF_TO_TRANSFORMERS_MAPPING["config"]`, causing `load_gguf_checkpoint` to fail with `ValueError: Unknown architecture`.

**Fix:** Registered `deepseek2` in both maps at import time (idempotent guard).

### Loader Bug 2: Broken load_gguf_checkpoint wrapper chain (fixed)

Transformers 5.2.0 added `model_to_load` keyword argument to `load_gguf_checkpoint`. Multiple other loaders in the repo patch `load_gguf_checkpoint` globally with closures that delegate to a captured `orig_load` but use `(*args, **kwargs)` and don't forward `model_to_load` through. Import order is non-deterministic (os.walk), so a broken wrapper may be installed when GigaChat3 loads.

**Fix:** Implemented `_unwrap_load_gguf_checkpoint` that walks the wrapper chain via both `__globals__` and `__closure__` cell inspection (cells named `orig_load`/`orig`) to find the real transformers function (identified by having `model_to_load` as an explicit named parameter). Re-installs our wrapper immediately before each GGUF load call to counteract later loaders.

### Loader Bug 3: deepseek_v2 not in GGUF_TO_FAST_CONVERTERS (fixed)

Our wrapper rewrites `model_type` from `deepseek2` to `deepseek_v2`. The tokenizer path calls `convert_gguf_tokenizer("deepseek_v2", ...)` which looks up `GGUF_TO_FAST_CONVERTERS["deepseek_v2"]` — absent, causing `KeyError`.

**Fix:** `GGUF_TO_FAST_CONVERTERS.setdefault("deepseek_v2", GGUFQwen2Converter)` at import time.

### Loader Bug 4: deepseek_v2 not in gguf-py MODEL_ARCH_NAMES (fixed)

`get_gguf_hf_weights_map` looks up `deepseek_v2` in gguf-py's `MODEL_ARCH_NAMES` — but gguf-py uses `deepseek2`, causing `NotImplementedError: Unknown gguf model_type: deepseek_v2`.

**Fix:** Patched `get_gguf_hf_weights_map` to translate `model_type=deepseek_v2` back to `deepseek2` before the gguf-py arch lookup.

### Remaining Failure: Tier B — MLA SDPA shape mismatch in FakeTensor propagation

After all loader fixes, the model loads and `torch.compile` begins tracing. DeepSeek V2 uses Multi-Head Latent Attention (MLA) where Q/K/V tensors have non-standard dimension relationships. During Dynamo's FakeTensor shape propagation, `scaled_dot_product_attention` is called with:

- Q: `(1, 32, 128, 256)` — 32 query heads
- K: `(1, 1024, 128, 256)` — 1024 "heads" (MLA expanded KV)
- V: `(1, 1024, 128, 192)` — 1024 "heads" with different head_dim

The TT XLA backend raises:
```
RuntimeError: Attempting to broadcast a dimension of length 1024 at -1!
Mismatching argument at index 1 had [1, 1024]; but expected shape should be broadcastable to [1, 32]
```

This occurs at `torch_overrides.py:34` inside `__torch_function__` — the XLA backend shape inference does not handle MLA's non-standard Q/K/V head dimension asymmetry. This is a Tier B compiler infrastructure bug.

**No loader workaround is applicable** — MLA requires these non-standard SDPA shapes by design. A compiler fix in the TT XLA backend is required.

## Files Changed

**tt-forge-models submodule** (branch: `remediation/bartowski_ai_sage_gigachat3_10b_a1_8b_gguf-causal_lm-pytorch-10B_A1_8B_GGUF-single_device-inference`):

- `bartowski_ai_sage_gigachat3_10b_a1_8b_gguf/causal_lm/pytorch/loader.py` — complete rewrite with deepseek2 GGUF support: registers arch+config mapping, robust `load_gguf_checkpoint` unwrapper with closure inspection, `get_gguf_hf_weights_map` patch, idempotent re-installation before each load call
- `glm_4_7_flash_gguf/causal_lm/pytorch/loader.py` — defensive fix: register `deepseek_v2` in `GGUF_TO_FAST_CONVERTERS`, patch `tokenization_utils_tokenizers`
- `glm_4_7_flash_ggml_org_gguf/causal_lm/pytorch/loader.py` — same defensive fix
