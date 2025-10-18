# Model Auto-Discovery Tests

## Overview

- **What**: A pytest-based runner that auto-discovers Torch models from `tt-forge-models` and generates tests for inference and training across parallelism modes.
- **Why**: Standardize model testing, reduce bespoke tests in repos, and scale coverage as models are added or updated.
- **Scope**: Discovers `loader.py` under `<model>/pytorch/` in `third_party/tt_forge_models`, queries variants, and runs each combination of:
  - **Run mode**: `inference`, `training`
  - **Parallelism**: `single_device`, `data_parallel`, `tensor_parallel`

Note: Discovery currently targets PyTorch models only. JAX model auto-discovery is planned.

## Prerequisites

- A working TT-XLA development environment, built and ready to run tests, with `pytest` installed.
- `third_party/tt_forge_models` git submodule initialized and up to date:
```bash
git submodule update --init --recursive third_party/tt_forge_models
```
- Device availability matching your chosen parallelism mode (e.g., multiple devices for data/tensor parallel).
- Optional internet access for per-model pip installs during test execution.
- env-var `IRD_LF_CACHE` set to point to large file cache / webserver for s3 bucket mirror. Reach out to team for details.

## Quick start / commonly used commands

Warning: Since the number of models and variants supported here is high (1000+), it is a good idea to run with `--collect-only` first to see what will be discovered/collected before running non-targeted pytest commands locally.

Also, running the full matrix can collect thousands of tests and may install per-model Python packages during execution. Prefer targeted runs locally using `-m`, `-k`, or an exact node id.

Tip: Use `-q --collect-only` to list tests with full path shown, remove `--collect-only` and use `-vv` when running.


- List all tests without running:
```bash
pytest --collect-only -q tests/runner/test_models.py |& tee collect.log
```

- List only tensor-parallel expected-passing on `n300-llmbox` (remove `--collect-only` to run):
```bash
pytest --collect-only -q tests/runner/test_models.py -m "tensor_parallel and expected_passing and n300_llmbox" --arch n300-llmbox |& tee tests.log
```

- Run a specific collected test node id exactly:
```bash
pytest -vv tests/runner/test_models.py::test_all_models[llama/sequence_classification/pytorch-llama_3_2_1b-single_device-full-inference] |& tee test.log
```

- Validate test_config files for typos, model name changes, useful when making updates:
```bash
pytest -svv --validate-test-config tests/runner/test_models.py |& tee validate.log
```

- List all expected passing llama inference tests for n150 (using substring `-k` and markers with `-m`):
```bash
  pytest -q --collect-only -k "llama" tests/runner/test_models.py \
   -m "n150 and expected_passing and inference" |& tee tests.log

tests/runner/test_models.py::test_all_models[deepcogito/pytorch-v1_preview_llama_3b-single_device-full-inference]
tests/runner/test_models.py::test_all_models[huggyllama/pytorch-llama_7b-single_device-full-inference]
tests/runner/test_models.py::test_all_models[llama/sequence_classification/pytorch-llama_3_8b_instruct-single_device-full-inference]
tests/runner/test_models.py::test_all_models[llama/sequence_classification/pytorch-llama_3_1_8b-single_device-full-inference]
tests/runner/test_models.py::test_all_models[llama/causal_lm/pytorch-llama_3_8b-single_device-full-inference]
tests/runner/test_models.py::test_all_models[llama/causal_lm/pytorch-llama_3_8b_instruct-single_device-full-inference]
tests/runner/test_models.py::test_all_models[llama/causal_lm/pytorch-llama_3_1_8b-single_device-full-inference]
<snip>
21/3048 tests collected (3027 deselected) in 3.53s
```

## How discovery and parametrization work

- The runner scans `third_party/tt_forge_models/**/pytorch/loader.py` (the git submodule) and imports `ModelLoader` to call `query_available_variants()`.
- For every discovered variant, the runner generates tests across run modes and parallelism.
- Implementation highlights:
  - Discovery and IDs: `tests/runner/test_utils.py` (`setup_test_discovery`, `discover_loader_paths`, `create_test_entries`, `create_test_id_generator`)
  - Main test: `tests/runner/test_models.py`

## Test IDs and filtering

- Test ID format: `<relative_model_path>-<variant_name>-<parallelism>-full-<run_mode>`
- Examples:
  - `squeezebert/pytorch-squeezebert-mnli-single_device-full-inference`
  - `...-data_parallel-full-training`
- Filter by substring with `-k` or by markers with `-m`:
```bash
pytest -q -k "qwen_2_5_vl/pytorch-3b_instruct" tests/runner/test_models.py
pytest -q -m "training and tensor_parallel" tests/runner/test_models.py
```

Take a look at `model-test-passing.json` and related `.json` files inside `.github/workflows/test-matrix-presets` for seeing how filtering works for CI jobs.

## Parallelism modes

- **single_device**: Standard execution on one device.
- **data_parallel**: Inputs are automatically batched to `xr.global_runtime_device_count()`; shard spec inferred on batch dim 0.
- **tensor_parallel**: Mesh derived from `loader.get_mesh_config(num_devices)`; execution sharded by model dimension.

## Per-model requirements

- If a model provides `requirements.txt` next to its `loader.py`, the runner will:
  1. Freeze the current environment
  2. Install those requirements (and optional `requirements.nodeps.txt` with `--no-deps`)
  3. Run tests
  4. Uninstall newly added packages and restore version changes

- Environment toggles:
  - `TT_XLA_DISABLE_MODEL_REQS=1` to disable install/uninstall management
  - `TT_XLA_REQS_DEBUG=1` to print pip operations for debugging

## Test configuration and statuses

- Central configuration is merged from `tests/runner/test_config/*` via `tests/runner/test_config/__init__.py`.
- Example: `tests/runner/test_config/test_config_inference_single_device.py` for all single device inference test tagging, and `tests/runner/test_config/test_config_inference_data_parallel.py` for data parallel inference test tagging.
- Each entry is keyed by the collected test ID and can specify:
  - **Status**: `EXPECTED_PASSING`, `KNOWN_FAILURE_XFAIL`, `NOT_SUPPORTED_SKIP`, `UNSPECIFIED`, `EXCLUDE_MODEL`
  - **Comparators**: `required_pcc`, `assert_pcc`, `assert_allclose`, `allclose_rtol`, `allclose_atol`
  - **Metadata**: `bringup_status`, `reason`, custom `markers` (e.g., `push`, `nightly`)
  - **Architecture scoping**: `supported_archs` used for filtering by CI job and optional `arch_overrides` used if test_config entries need to be modified based on arch.

## Model status and bringup_status guidance

Use `tests/runner/test_config/*` to declare intent for each collected test ID. Typical fields:

- `status` (from `ModelTestStatus`) controls filtering of tests in CI:
  - `EXPECTED_PASSING`: Test is green and should run in Nightly CI. Optionally set thresholds.
  - `KNOWN_FAILURE_XFAIL`: Known failure that should xfail; include `reason` and `bringup_status` to set them statically otherwise will attempt to be set dynamically at runtime.
  - `NOT_SUPPORTED_SKIP`: Skip on this architecture or generally unsupported; provide `reason` and (optionally) `bringup_status`.
  - `UNSPECIFIED`: Default for new tests; runs in Experimental Nightly until triaged.
  - `EXCLUDE_MODEL`: Deselect from auto-run entirely (rare; use for temporary exclusions).

- `bringup_status` (from `BringupStatus`) summarizes current health for Superset dashboard reporting:
  - `PASSED` (set automatically on pass),
  - `INCORRECT_RESULT` (e.g., PCC mismatch),
  - `FAILED_FE_COMPILATION` (frontend compile error),
  - `FAILED_TTMLIR_COMPILATION` (tt-mlir compile error),
  - `FAILED_RUNTIME` (runtime crash),
  - `NOT_STARTED`, `UNKNOWN`.

- `reason`: Short human-readable context, ideally with a link to a tracking issue.

- Comparator controls: prefer `required_pcc`; use `assert_pcc=False` sparingly as a temporary measure.

Examples

- Passing with a tuned PCC threshold if reasonable / understood decrease:
```python
"resnet/pytorch-resnet_50_hf-single_device-full-inference": {
  "status": ModelTestStatus.EXPECTED_PASSING,
  "required_pcc": 0.98,
}
```

- Known compile failure (xfail) with issue link:
```python
"clip/pytorch-openai/clip-vit-base-patch32-single_device-full-inference": {
  "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
  "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
  "reason": "Error Message - Github issue link",
}
```

- If minor unexpected PCC mismatch, open ticket, decrease threshold and set bringup_status/reason as:
```python
"wide_resnet/pytorch-wide_resnet101_2-single_device-full-inference": {
  "status": ModelTestStatus.EXPECTED_PASSING,
  "required_pcc": 0.96,
  "bringup_status": BringupStatus.INCORRECT_RESULT,
  "reason": "PCC regression after consteval changes - Github Issue Link",
}
```

- If severe unexpected PCC mismatch, open ticket, disable pcc check and set bringup_status/reason as:
```python
"gpt_neo/causal_lm/pytorch-gpt_neo_2_7B-single_device-full-inference": {
  "status": ModelTestStatus.EXPECTED_PASSING,
  "assert_pcc": False,
  "bringup_status": BringupStatus.INCORRECT_RESULT,
  "reason": "AssertionError: PCC comparison failed. Calculated: pcc=-1.0000001192092896. Required: pcc=0.99 - Github Issue Link",
}
```

- Architecture-specific overrides (e.g., pcc thresholds, status, etc):
```python
"qwen_3/embedding/pytorch-embedding_8b-single_device-full-inference": {
    "status": ModelTestStatus.EXPECTED_PASSING,
    "arch_overrides": {
        "n150": {
            "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
            "reason": "Too large for single chip",
            "bringup_status": BringupStatus.FAILED_RUNTIME,
        },
    },
},
```

## Targeting architectures

- Use `--arch {n150,p150,n300,n300-llmbox}` on pytest command line to enable `arch_overrides` resolution in config in case there are specific overrides (like PCC requirements, checking enablement, tagging) per arch.
- Tests are also marked with supported arch markers (or defaults), so you can select subsets using `-m`, example:
```bash
pytest -q -m n300 --arch n300 tests/runner/test_models.py
pytest -q -m n300_llmbox --arch n300-llmbox tests/runner/test_models.py
```

## Placeholder models (report-only)

- `PLACEHOLDER_MODELS` in `tests/runner/test_config/test_config_inference_single_device.py` lists important customer `ModelGroup.RED` models not yet merged, typically marked with `BringupStatus.NOT_STARTED`.
- `tests/runner/test_models.py::test_placeholder_models` emits report entries with the `placeholder` marker; used for reporting on Superset dashboard and run in tt-xla Nightly CI (typically via `model-test-xfail.json`).
- Be sure to remove the placeholder at the same time the real model is added to avoid duplicate reports.


## CI setup

- Push/PR: A small, fast subset runs on each pull request (e.g., tests marked `push`). This provides quick signal without large queues.
- Nightly: The broad model matrix (inference/training across supported parallelism) runs nightly and reports to the Superset dashboard. Tests are selected via markers and `tests/runner/test_config/*` statuses/arch tags like `ModelTestStatus.EXPECTED_PASSING`
- Experimental nightly: New or experimental models not yet promoted/tagged in `tests/runner/test_config/*` (typically `unspecified`) run separately. These do not report to Superset until promoted with proper status/markers.

## Adding a new model to run in Nightly CI

It is not difficult, but involves potentially 2 projects (`tt-xla` and `tt-forge-models`). If model is already added to `tt-forge-models` and uplifted to `tt-xla` then skip steps 1-4.

1. In `tt-forge-models/<model>/pytorch/loader.py`, implement a `ModelLoader` if doesn't already exist, exposing:
   - `query_available_variants()` and `get_model_info(variant=...)`
   - `load_model(...)` and `load_inputs(...)`
   - `load_shard_spec(...)` (if needed) and `get_mesh_config(num_devices)` (for tensor parallel)
2. Optionally add `requirements.txt` (and `requirements.nodeps.txt`) next to `loader.py` for per-model dependencies.
3. Contribute the model upstream: open a PR in the `tt-forge-models` repository and land it (see `tt-forge-models` repo: https://github.com/tenstorrent/tt-forge-models).
4. Uplift `third_party/tt_forge_models` submodule in `tt-xla` to the merged commit so the loader is discoverable:
   - Update the submodule and commit the pointer:
```bash
git submodule update --remote third_party/tt_forge_models
git add third_party/tt_forge_models
git commit -m "Uplift tt-forge-models submodule to <version> to include <model>"
```
5. Verify the test appears via `--collect-only` and run desired flavor locally if needed.
6. Add or update the corresponding entry in `tests/runner/test_config/*` to set status/thresholds/markers/arch support so that the model test is run in tt-xla Nightly CI. Look at existing tests for reference.
7. Remove any corresponding placeholder entry from `PLACEHOLDER_MODELS` if it exists.
8. Locally run `pytest -q --validate-test-config tests/runner/test_models.py` to validate `tests/runner/test_config/*` updates (on-PR jobs run it too).
9. Open a PR in `tt-xla` for changes, consider running full set of expected passing models on CI to qualify `tt_forge_models` uplift (if it is risky), and land the PR in `tt-xla` main when confident in changes.

## Troubleshooting

- Discovery/import errors show as: `Cannot import path: <loader.py>: <error>`; add per-model requirements or set `TT_XLA_DISABLE_MODEL_REQS=1` to isolate issues.
- Runtime/compilation failures are recorded with a bring-up status and reason in test properties; check the test report’s `tags` and `error_message`.
- Some models may be temporarily excluded from discovery; see logs printed during collection.
- Use `-vv` and `--collect-only` for detailed collection/ID debugging.

## Future enhancements

- Expand auto-discovery beyond PyTorch to include JAX models
- Automate updates of `tests/runner/test_config/*` potentially based on results of Nightly CI, automatic promotion of tests from Experimental Nightly to stable Nightly.
- Broader usability improvements and workflow polish tracked in [issue #1307](https://github.com/tenstorrent/tt-xla/issues/1307)

## Reference

- `tests/runner/test_models.py`: main parametrized pytest runner
- `tests/runner/test_utils.py`: discovery, IDs, `DynamicTorchModelTester`
- `tests/runner/requirements.py`: per-model requirements context manager
- `tests/runner/conftest.py`: config attachment, markers, `--arch`, config validation
- `tests/runner/test_config/*`: test config files to mark status and what is run in CI
- `third_party/tt_forge_models/config.py`: `Parallelism` and model metadata
