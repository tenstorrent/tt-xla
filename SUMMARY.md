# Remediation Summary: devquasar_ibm_granite_4_0_1b_gguf-causal_lm-pytorch-4_0_1B_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[devquasar_ibm_granite_4_0_1b_gguf/causal_lm/pytorch-4_0_1B_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS — granite GGUF architecture registration fix already on hf-bringup-7 branch; test passes 261.70s on n150

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-granite-arch-not-in-config-mapping

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original reported failure was:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Reproduction (with gguf 0.18.0 already installed) showed the real underlying failure:
```
ValueError: GGUF model with architecture granite is not supported yet.
```

The ImportError is raised by `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
when `is_gguf_available()` returns False (gguf not installed or version < 0.10.0). In the
reproduced environment gguf 0.18.0 was present, advancing to the architecture error.

## Root cause
Loader layer. IBM Granite 4.0 1B GGUF declares `general.architecture='granite'` in its GGUF
metadata, but `'granite'` is absent from transformers' `GGUF_CONFIG_MAPPING` and
`GGUF_SUPPORTED_ARCHITECTURES`. The check at line 477 of
`transformers/modeling_gguf_pytorch_utils.py`:

```python
if architecture not in GGUF_SUPPORTED_ARCHITECTURES and updated_architecture not in GGUF_SUPPORTED_ARCHITECTURES:
    raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
```

also rejects it because the `GGUF_TO_FAST_CONVERTERS` dict had no entry for `'granite'`.

Additionally, the GGUF file stores `granite.attention.head_count_kv` as a per-layer array
(40 × 4), while `GraniteConfig.num_key_value_heads` expects a scalar integer.

## Fix
Fix commit `011ae38a73` in `tt_forge_models` on branch
`remediation/devquasar_ibm_granite_4_0_1b_gguf-causal_lm-pytorch-4_0_1B_Q4_K_M_GGUF-single_device-inference`
(which is already on `origin/arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-7`).

The fix in `devquasar_ibm_granite_4_0_1b_gguf/causal_lm/pytorch/loader.py`:

1. Registers `'granite'` in `GGUF_SUPPORTED_ARCHITECTURES` and
   `GGUF_TO_TRANSFORMERS_MAPPING["config"]` with llama-style key mappings plus Granite-specific
   scaling fields (`attention.scale` → `attention_multiplier`, `embedding_scale` →
   `embedding_multiplier`, `residual_scale` → `residual_multiplier`).
2. Maps `'granite'` → `GGUFLlamaConverter` in `GGUF_TO_FAST_CONVERTERS` (Granite 4.0 uses
   LLaMA 3-style BPE tokenizer).
3. Patches `load_gguf_checkpoint` to flatten the per-layer `num_key_value_heads` list to a
   scalar (all layers share the same value; `kv_heads[0]` is used).
4. Patches all four call sites where `load_gguf_checkpoint` is referenced in transformers
   (`modeling_gguf_pytorch_utils`, `configuration_utils`, `tokenization_auto`,
   `tokenization_utils_tokenizers`).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    261.70s (0:04:21)
- Tier A attempts: N/A

## Files changed
- `devquasar_ibm_granite_4_0_1b_gguf/causal_lm/pytorch/loader.py` (tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 011ae38a733c0ef4b75b3b5aee09b53344341d6d |
