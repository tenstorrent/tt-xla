# Model Run Job Protocol

## Purpose

Define the repository-owned protocol for launching, validating, and reporting bounded model-run jobs from `tt-xla`. Agent-specific skills must reference this protocol instead of carrying independent launch logic.

## Supported Job Types

- `nvidia_validation`
  - Purpose: run a manifest through `tests/runner/test_models_nvidia.py` on a CUDA/NVIDIA host.
  - Required command family: `pytest ... tests/runner/test_models_nvidia.py --nvidia-cohort-json <manifest>`.
- `tt_model_validation`
  - Purpose: run normal TT model tests through the existing `tests/runner/test_models.py` lane.
  - Required command family: `pytest ... tests/runner/test_models.py::<test>`.
- `collectability_check`
  - Purpose: determine which manifest rows are collectable by the current harness before execution.
  - Required command family: `pytest --collect-only ... --nvidia-cohort-json <manifest>`.

## Manifest Contract

Every model-run job must start from a machine-readable manifest with:

- `manifest_type`
- `source_file`
- `source_sha256`
- `selection_rule`
- `selection_date`
- `models`

Each model row must include:

- `test_case_id`
- `source_status` when derived from a previous result file
- `source_branch` when available
- `source_hostname` when available
- `source_date` when available
- `source_test_name` or equivalent provenance field

Do not run an inferred or hand-selected cohort unless the manifest records the selection rule and reviewer approval source.

## Execution Contract

Before running:

- Confirm the repo ref and submodule refs.
- Confirm the runner host, hardware, OS, Python environment, CUDA or TT device availability, and disk headroom.
- Run a collectability check for manifest-driven NVIDIA jobs.
- Split rows into `runnable`, `blocked_collectability`, and `needs_mapping`.

During execution:

- Run bounded waves rather than a single unbounded batch.
- Record the exact command line for every wave.
- Preserve `pytest.log`, `junit.xml`, environment snapshot, and manifest path for every wave.
- Clean transient model/cache artifacts only through documented cleanup commands.

After execution:

- Normalize every row to one of:
  - `validated_pass`
  - `validated_fail`
  - `pipeline_error`
  - `blocked_collectability`
  - `needs_mapping`
  - `pending_terminalization`
- Preserve original harness details separately from normalized outcome.
- Do not classify infrastructure, auth, missing-artifact, or OOM capacity failures as model-quality failures without evidence.

## NVIDIA Validation Command Shape

Use this shape for NVIDIA validation:

```sh
pytest -q \
  tests/runner/test_models_nvidia.py \
  --nvidia-cohort-json "<manifest>" \
  --tb=short \
  --disable-warnings \
  --junitxml="<wave_dir>/junit.xml" \
  >"<wave_dir>/pytest.log" 2>&1
```

For single-row or wave-specific execution, select parametrized cases explicitly:

```sh
pytest -q \
  "tests/runner/test_models_nvidia.py::test_models_torch_nvidia[<test_case_id>]" \
  --nvidia-cohort-json "<manifest>" \
  --tb=short \
  --disable-warnings \
  --junitxml="<wave_dir>/junit.xml" \
  >"<wave_dir>/pytest.log" 2>&1
```

## Reporting Rules

- Do not post GitHub comments, update issues, or publish stakeholder reports without human review.
- Status updates must distinguish known source coverage from requested target coverage.
- If the requested target count is larger than the available source count, report the gap explicitly.
- Every external report must include source link, source SHA-256, repo ref, command line, and evidence paths.

## Agent Wrapper Rules

Claude, Codex, and other agent-specific skills must:

- Read this protocol first.
- Select one supported job type.
- Ask for or locate the manifest.
- Refuse to invent missing source rows.
- Produce commands and evidence expectations from this protocol.
- Keep any external posting behind human review.

Agent wrappers must not:

- Fork alternate launch logic.
- Embed machine-local secrets.
- Treat local-only refs as CI-ready.
- Hide blocked rows by excluding them from summaries.
