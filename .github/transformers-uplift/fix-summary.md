transformers uplift: model-test-uplifts fixes — none applicable (maskformer compile error not attributable)

## Skipped (left for human review)
- tests/runner/test_models.py::test_all_models_torch[maskformer_swin_b/pytorch-Swin_Base_Coco-single_device-inference]: Fails with `ValueError: Error code: 13` from `torch_xla._XLAC._xla_warm_up_cache` (TT-MLIR compile/runtime error), not a Python/transformers API error. Diffed every file that builds this model's compiled graph between 5.5.1 and 5.8.1 — `modeling_maskformer.py`, `modeling_maskformer_swin.py` (Swin backbone), `modeling_swin.py`, and both maskformer image processors. The only changes are cosmetic `@dataclass`/`@auto_docstring` decorator reordering on the ModelOutput classes; attention interface, einsum/interpolate ops, and the input pipeline are functionally identical, and the image processors are byte-identical. The compiled graph is unchanged, so this compile failure is not caused by the transformers bump. No source-level transformers fix exists; needs investigation on the tt-mlir/compiler side (likely a compiler regression or flake independent of transformers).

## Stats
- Failures input: 1
- Fixed: 0
- Skipped: 1
