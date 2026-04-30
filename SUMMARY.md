# Remediation Summary: flux_kontext_gguf-pytorch-Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_kontext_gguf/pytorch-Q4_K_M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded

## Root cause
Three loader-layer bugs, identical in kind to the flux_dev_gguf fix:

1. **Gated-repo config detection** — `FluxTransformer2DModel.from_single_file` with a GGUF
   checkpoint reads raw quantized-byte tensor shapes to identify the model type. For this
   Q4_K_M checkpoint the `img_in.weight` packed-byte shape has `shape[1]==128`, which
   triggers the `flux-depth` branch in diffusers' `infer_diffusers_model_type` and sends it
   to the gated `black-forest-labs/FLUX.1-Depth-dev` repo — even though the semantic
   architecture is standard FLUX.1-dev (`in_channels=64`). Access is denied with a 403
   GatedRepoError before any weights are loaded.

2. **Missing `GGUFQuantizationConfig`** — without it, GGUFParameter objects (raw quantized
   bytes) are stored directly in the model's nn.Linear weights. The model forward pass
   works through `GGUFParameter.__torch_function__` on-the-fly dequantization, but
   TorchDynamo tracing calls `__torch_function__` recursively, immediately hitting Python's
   recursion limit.

3. **Missing `_dequantize_gguf_and_restore_linear`** — even with `GGUFQuantizationConfig`
   the model's GGUFLinear layers still hold quantized weights. Converting them to plain
   `nn.Linear` with float weights before `torch.compile` is required.

## Fix
In `third_party/tt_forge_models/flux_kontext_gguf/pytorch/loader.py` (remediation branch
`remediation/flux_kontext_gguf-pytorch-Q4_K_M-single_device-inference` of tt_forge_models):

- Added `_dequantize_gguf_and_restore_linear` import and call after `from_single_file`.
- Replaced the incorrect embedded config (`patch_size: 2`, `out_channels: 16`) with the
  correct FLUX.1-dev config (`patch_size: 1`, no `out_channels`). The config is written to
  a local temp directory and passed as `config=config_dir` to bypass the gated-repo lookup.
- Set `self.transformer.is_quantized = False` then cast to `compute_dtype` so the model
  has plain float weights before compilation.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    692.74s (0:11:32)
- Tier A attempts: N/A

## Files changed
- `flux_kontext_gguf/pytorch/loader.py` (tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b999150d4bef2098391872ce3e5a64985466a23c |
| tt-forge-models | 18c98b6dbdea43e9e170f13cf579fbba587904fd |
