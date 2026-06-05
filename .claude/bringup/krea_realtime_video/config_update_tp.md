# CONFIG_UPDATE_TP — krea_realtime_video / text_encoder

**Result:** PASSED (applied)
**Arch:** n300-llmbox  **Mesh:** (2,4) = 8 chips  **TT_VISIBLE_DEVICES:** 0,1,2,3

## Surface: pipeline component test (NOT runner YAML)

`test_path`: `tests/torch/models/krea_realtime/test_text_encoder.py::test_text_encoder_sharded`

Per the rules, pipeline component tests are **not** registered in
`test_config_inference_tensor_parallel.yaml`. They are selected by pytest
markers in nightly. No runner YAML change.

## Test final state (EXPECTED_PASSING)

The sharded node is in its final passing form — no xfail/skip:
- markers: `@pytest.mark.nightly`, `@pytest.mark.model_test`, `@pytest.mark.tensor_parallel`, `@pytest.mark.large`
- correctness: `run_graph_test(..., mesh=mesh, shard_spec_fn=loader.load_shard_spec, comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)))`
- bringup_status: implicit EXPECTED_PASSING (marker-only family, matching flux_2_dev); no `record_test_properties` added.

## Evidence

| Stage | Log | Result |
|-------|-----|--------|
| FIRST_RUN_TP (TT-only smoke) | logs/iter_1_run.log | PASSED 55.9s — mesh (2,4) created, sharded compile+exec OK |
| VERIFY_TP (CPU golden + PCC 0.99) | logs/iter_2_verify.log | PASSED 92.6s — PCC ≥ 0.99 asserted (assert_on_failure=True) |

## Provenance

- VALIDATE_TP: 218/242 params sharded, 24 replicated (per-layer relative_attention_bias), 0 problems (rank/axis/divisibility).
- TP pattern: B (FSDP-style 2D), ("batch","model") mesh.
- Shard spec source: `third_party/tt_forge_models/krea_realtime_video/pytorch/src/model_utils.py::shard_text_encoder_specs`.
