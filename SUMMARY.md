# Remediation Summary: mistral_nemo_base_2407_bnb_4bit-causal_lm-pytorch-Nemo_Base_2407_BNB_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral_nemo_base_2407_bnb_4bit/causal_lm/pytorch-Nemo_Base_2407_BNB_4bit-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
bnb-4bit-missing-requirements-and-dequantize

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise RuntimeError(

bitsandbytes `validate_environment` raises `RuntimeError` because CUDA is not
available in the TT test environment, which prevents `from_pretrained` from
loading the model with `quantization_config` set to BnB 4-bit. Additionally,
even if the model loaded, `bnb.nn.Linear4bit` modules are not executable on
TT hardware (no CUDA kernels).

## Root cause
The loader had two missing pieces:

1. **No requirements.txt**: bitsandbytes was not installed, so importing
   `transformers` with a BnB `quantization_config` raised `RuntimeError`
   from `quantizer_bnb_4bit.py:validate_environment` before any weights
   could load.

2. **No dequantization step**: After `from_pretrained`, all linear layers
   were `bnb.nn.Linear4bit` instances which require CUDA kernels. TT
   hardware cannot execute these; the model must be dequantized to
   standard `nn.Linear(bfloat16)` before being moved to the TT device.

The model is `unsloth/Mistral-Nemo-Base-2407-bnb-4bit`, an unsloth
checkpoint where weights are stored pre-dequantized as BF16 inside the
`Linear4bit` shell (no `quant_state` on the weight). The dequantize
function handles both the true-quantized and the unsloth-plain-BF16 cases.

## Fix
Two changes in `tt_forge_models/mistral_nemo_base_2407_bnb_4bit/causal_lm/pytorch/`:

1. **New file `requirements.txt`**: `bitsandbytes>=0.46.1`

2. **`loader.py`**: Added `_dequantize_bnb4_to_bf16(model)` function and
   called it after `from_pretrained` in `load_model()`. The function
   replaces every `bnb.nn.Linear4bit` with a standard `nn.Linear`
   (bfloat16). If `module.weight.quant_state` is present the weight is
   dequantized via `bnb.functional.dequantize_4bit`; otherwise (unsloth
   plain-BF16 case) it is cast directly to bfloat16.

   Files changed:
   - `tt_forge_models/mistral_nemo_base_2407_bnb_4bit/causal_lm/pytorch/loader.py`
   - `tt_forge_models/mistral_nemo_base_2407_bnb_4bit/causal_lm/pytorch/requirements.txt` (new)

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    240.22s (0:04:00)
- Tier A attempts: N/A

## Files changed
- tt_forge_models/mistral_nemo_base_2407_bnb_4bit/causal_lm/pytorch/loader.py
- tt_forge_models/mistral_nemo_base_2407_bnb_4bit/causal_lm/pytorch/requirements.txt (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9f26e9131a00d8525e9a7448a8155d12fb5579cf |
| tt-forge-models | 60d1f8881b69c74e765bb034e104b4b4bf50a274 |
