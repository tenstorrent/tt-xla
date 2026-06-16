transformers uplift: model-perf-uplift fixes — none applicable (runtime/infra failures)

## Skipped (left for human review)
- tests/benchmark/test_llms.py::test_phi2: device-runtime crash, not a transformers diff. The Python model call (decode_utils.py:324) runs; the failure is at the torch_xla step-marker sync — `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`, root-caused by tt-metal `TT_THROW: Statically allocated circular buffers ... clash with L1 buffers on core range [0-0 - 7-7]`. L1/circular-buffer allocation clash in the compiler/runtime; no transformers API attribution.
- tests/benchmark/test_vision.py::test_vovnet: CI infrastructure only — `tenstorrent-hugepages.service` failed and the wheel download returned `HTTP 502: Server Error` (xla-whl-release-b7be0a5). The test never executed; nothing transformers-related.

## Stats
- Failures input: 2
- Fixed: 0
- Skipped: 2
