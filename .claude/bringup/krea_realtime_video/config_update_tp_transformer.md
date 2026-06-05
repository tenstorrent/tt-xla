# CONFIG_UPDATE_TP — krea_realtime_video / transformer

**Result:** ESCALATED (known PCC gap)  **Arch:** n300-llmbox  **Mesh:** (2,4) = 8 chips

## Surface: pipeline component test (NOT runner YAML)

`test_path`: `tests/torch/models/krea_realtime/test_transformer.py::test_transformer_sharded`

## Outcome

The sharded TP transformer **compiles and runs** on the 8-chip mesh after the
shard-spec repair (Megatron-1D, stem replicated). Numerical correctness vs the
CPU golden is **PCC 0.963 < 0.99**.

Per user decision: keep `required_pcc=0.99`, mark the node `@pytest.mark.xfail`
(strict=False) with the 0.963 reason, and track the numerical gap in a GitHub
issue. **No threshold relaxation.**

## Test final state

- markers: `@pytest.mark.nightly @model_test @tensor_parallel @large`
- `@pytest.mark.xfail(strict=False, reason="...PCC=0.963 < 0.99...Tracking: TODO(file tt-xla issue)")`
- correctness still wired: `run_graph_test(..., mesh, shard_spec_fn, ComparisonConfig(pcc=PccConfig(required_pcc=0.99)))`
- bringup_status: KNOWN_FAILURE_XFAIL (numerical, not functional)

## Tracking issue

Draft at `.claude/bringup/krea_realtime_video/transformer_pcc_issue_draft.md`.
`gh` is installed but not authenticated — file with `gh auth login` then
`gh issue create --repo tenstorrent/tt-xla` (or paste the draft). Replace the
`TODO(file tt-xla issue)` in the xfail reason with the issue URL once filed.

## Evidence

| Stage | Log | Result |
|-------|-----|--------|
| FIRST_RUN_TP iter1 | logs/iter_1_transformer_run.log | FAILED — reshape mismatch (time_projection unflatten, 6∤4) |
| REPAIR_SHARD | model_utils.py shard_transformer_specs | Megatron-1D, stem replicated |
| FIRST_RUN_TP iter2 | logs/iter_2_transformer_repair.log | PASSED 1380.8s (TT-only smoke) |
| VERIFY_TP iter3 | logs/iter_3_transformer_verify.log | FAILED_PCC 1214.2s — pcc=0.9630 vs 0.99 |
