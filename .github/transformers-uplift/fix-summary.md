transformers uplift: model-perf-uplift — no transformers-attributable fix

## Skipped (left for human review)
- All 27 perf regressions: systemic samples/sec drop across every model class, NOT transformers-attributable. Decisive: mnist (-56.8%) and test_vision.py import no transformers at all, yet regressed hard; resnet (-18%), mobilenetv2 (-8.7%), vovnet (-5.7%), unet (-5.5%) are pure-vision models untouched by the generation/cache/attention churn. Hypothesis: a tt-mlir/tt-metal/runtime change in the same nightly, not the 5.5.1->5.5.2 bump.
- tests/benchmark/test_llms.py::test_falcon3_1b: infra flake — `503 Service Unavailable` fetching httpcore from download.pytorch.org during venv install; not a transformers issue.
- tests/benchmark/test_llms.py::test_falcon3_7b: `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` at torch_xla `_xla_step_marker`; generic runtime/compiler internal error with no transformers API attribution.
- Affected regression entries (all skipped, same hypothesis): bert (-7.2%), llama_3_1_8b (-10.3%), llama_3_2_1b (-14.3%), llama_3_2_3b (-24.2%), phi-1_5 (-13.5%), phi-1 (-13.1%), phi-2 (-12.1%), ministral_8b (-51.3%), mnist (-56.8%), mobilenetv2 (-8.7%), qwen_2_5_0_5b (-28.9%), qwen_2_5_1_5b (-11.8%), qwen_2_5_7b (-7.1%), qwen_3_0_6b (-19.5%), qwen_3_1_7b (-23.5%), qwen_3_8b (-12.0%), resnet (-18.0%), resnet_jax (-65.1%), segformer (-33.8%), swin (-52.8%), falcon3-3b (-17.1%), unet (-5.5%), unet_for_conditional_generation (-45.4%), vovnet (-5.7%).

## Stats
- Failures input: 29
- Fixed: 0
- Skipped: 29
