transformers uplift: model-test-uplifts fixes — no source-attributable fix

## Skipped (left for human review)
- tests/runner/test_models.py::test_all_models_torch[maskformer_swin_b/pytorch-Swin_Base_Coco-single_device-inference]: Fails with `ValueError: Error code: 13` at `torch_xla._XLAC._xla_warm_up_cache` (TT device compile failure during e2e-perf warm-up). Not attributable to the 5.5.1 -> 5.5.2 uplift: `modeling_maskformer.py` and `modeling_swin.py` are byte-identical between the two versions, and the maskformer image processor is unchanged. The only files that changed in 5.5.2 are weight-loading infra (`__init__.py`, `conversion_mapping.py`, `core_model_loading.py` — touch only mixtral/qwen2_moe/gemma3n/qwen3_5/llava/got_ocr2 rename rules) and `gemma4/*`; none affect this model's graph or weights. Likely a flaky/tt-mlir device-compile issue rather than a transformers regression. Left config as EXPECTED_PASSING; lowering PCC or marking xfail would be a blind fix with no transformers cause.

## Stats
- Failures input: 1
- Fixed: 0
- Skipped: 1
