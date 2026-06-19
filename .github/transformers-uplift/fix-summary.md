transformers uplift: model-test-uplifts fixes — no uplift-induced failures

## Skipped (left for human review)
- maskformer_swin_b/pytorch-Swin_Base_Coco-single_device-inference: pre-existing on baseline — same node id with identical root cause (`ValueError: Error code: 13` from `_xla_warm_up_cache`, same traceback through `modeling_maskformer.py:1915`) appears in baseline_failures.txt; not uplift-induced.

## Stats
- Failures input: 1
- Fixed: 0
- Skipped: 1
