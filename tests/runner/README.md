# Runner Utilities

## Frontend Training Failure Triage

Use `frontend_training_failure_triage.py` to generate bounded frontend triage artifacts for training failures whose current config reason suggests missing `unpack_forward_output` support.

Example:

```sh
python3 tests/runner/frontend_training_failure_triage.py \
  --output-root artifacts/frontend_training_triage_prd006_seed \
  --test-id yolov6/pytorch-N-single_device-training \
  --test-id distilbert/question_answering/pytorch-Base_Cased_Distilled_Squad-single_device-training \
  --test-id fpn/pytorch-ResNet50_Backbone_with_FPN_V2-single_device-training \
  --test-id maptr/pytorch-Tiny_R50_24e_Av2-single_device-training
```

Outputs:
- per-test folder with `manifest.json` and either `draft_issue.md` or `attempt.log`
- `summary.json` with run-level counts
- `review_report.md` for human review
- `config_refresh_recommendations.json` for stale-reason candidates
- `config_refresh_patch.yaml` as a reviewable proposed YAML refresh artifact

Selection rule:
- defaults to rows in `tests/runner/test_config/torch/test_config_training_single_device.yaml` whose reason is exactly:
  - `tt-forge-models doesn't implement unpack_forward_output for this model.`

Current inspection surface:
- training config source:
  - `tests/runner/test_config/torch/test_config_training_single_device.yaml`
- shared helper registry:
  - `third_party/tt_forge_models/training_utils.py`
- model-specific loader inspection:
  - `third_party/tt_forge_models/**/pytorch/loader.py`

Notes:
- this tool is draft-only; it does not file issues or mutate the training config
- stale-reason suspects are surfaced separately when the loader already implements `unpack_forward_output`

## Runtime Training Failure Reduction

Use `runtime_training_failure_reduction.py` to generate bounded runtime-reduction artifacts for training failures whose config metadata already indicates runtime or metal-style problems.

Example:

```sh
python3 tests/runner/runtime_training_failure_reduction.py \
  --output-root artifacts/runtime_training_reduction_prd006_seed \
  --debug-log-root artifacts/runtime_debug_logs \
  --test-id pointpillars/pytorch-pointpillars-single_device-training \
  --test-id beit/image_classification/pytorch-Base_Patch16_224-single_device-training \
  --test-id convnextv2/image_classification/pytorch-Atto-single_device-training \
  --test-id xlstm/pytorch-single_device-training
```

Outputs:
- per-test folder with `manifest.json` and either `draft_issue.md` or `attempt.log`
- optional per-test `debug_evidence.md` when a matching runtime debug log is supplied
- optional per-test `runtime_rerun.log` when `--execute-rerun` is used and no matching debug log exists
- `summary.json` with run-level counts
- `review_report.md` for human review

Debug evidence intake:
- optional `--debug-log-root` accepts either:
  - a directory containing per-test log files named after sanitized test ids
  - a single debug log file
- when debug logs are present, the tool extracts:
  - `Executing operation` lines
  - runtime signal lines such as `TT_FATAL` and `RuntimeError`
  - nearby `ttnn` / MLIR context lines
- this does not rerun the test, but it upgrades the bounded runtime packet from config-only routing to evidence-backed reduction artifacts

Bounded rerun path:
- optional `--execute-rerun` tries:
  - `pytest -vv -s tests/runner/test_models.py::test_all_models_torch[<test-id>]`
  - with `TTMLIR_LOGGER_LEVEL=DEBUG`
  - and `TT_RUNTIME_DEBUG=ON`
- `--pytest-bin` is restricted to the literal `pytest` command; select a repo-local
  or virtualenv pytest by activating the environment or prepending its `bin`
  directory to `PATH`
- this path writes `runtime_rerun.log` and then extracts debug evidence from that log if possible
- before launching pytest, the tool now probes the paired Python environment for:
  - `psutil`
  - `pytest`
  - `torch`
  - `torch_xla`
- if those imports fail, the tool records a `rerun precondition violation` attempt log instead of pretending a runtime draft is ready
- use `--force-run-skipped` only for bounded debug capture of selected `NOT_SUPPORTED_SKIP` rows; it sets `TT_XLA_FORCE_RUN_SKIPPED_TEST_IDS` for the subprocess and does not mutate source YAML

Selection rule:
- defaults to rows in `tests/runner/test_config/torch/test_config_training_single_device.yaml` whose `bringup_status` is `FAILED_RUNTIME` or whose reason contains a strong runtime keyword

Current owner heuristics:
- memory allocation, L1, DRAM, `TT_FATAL`, and device-buffer failures:
  - `tt-metal` draft candidate
- other bounded runtime failures:
  - `tt-alchemist` draft candidate
- hangs:
  - attempt log only until real debug-log capture exists

Notes:
- this tool is draft-only; it reruns tests only when `--execute-rerun` is explicitly supplied and does not file issues or mutate source YAML
- without `--debug-log-root`, the next manual step remains to capture real `TTMLIR_LOGGER_LEVEL=DEBUG` and `TT_RUNTIME_DEBUG=ON` evidence before filing

## Bounded Workflow Validation

Use `bounded_training_workflow_validation.py` to validate bounded frontend and runtime workflow outputs against the PRD-006 review rubric.

Example:

```sh
python3 tests/runner/bounded_training_workflow_validation.py \
  --frontend-output-root artifacts/frontend_training_triage_prd006_seed_v6 \
  --runtime-output-root artifacts/runtime_training_reduction_prd006_seed_v2 \
  --output-root artifacts/bounded_training_workflow_validation_prd006
```

Outputs:
- `validation_summary.json`
- `validation_review_packet.md`

Notes:
- this validator is evidence-based and artifact-only
- it does not file issues or rerun tests
