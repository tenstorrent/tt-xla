transformers uplift: model-perf-uplift — no uplift-induced failures

## Skipped (left for human review)
- tests/benchmark/test_llms.py::test_phi2 (n150): fails with `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` raised inside `torch_xla.sync()` (`_xla_step_marker`), not from benchmark infra. The model loads and `model(**input_args)` traces fine; the error is a backend compile/runtime failure, not a transformers API change. The runner also hit a hugepages infra failure ("Failed to get requested 4 hugepages, only got 2"). The same "Error code: 13" class affects 84 tests on the baseline nightly, so this is a pre-existing backend issue / infra flake, not attributable to a transformers diff. No source-side fix.

## Stats
- Failures input: 1
- Fixed: 0
- Skipped: 1
