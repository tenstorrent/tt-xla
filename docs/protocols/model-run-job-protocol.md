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
- Create a launch record before executing the first row or wave.

During execution:

- Run bounded waves rather than a single unbounded batch.
- Record the exact command line for every wave.
- Preserve `pytest.log`, `junit.xml`, environment snapshot, and manifest path for every wave.
- Clean transient model/cache artifacts only through documented cleanup commands.
- Update local checkpoints for routine progress instead of posting GitHub status comments.

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
- Write normalized row results using `docs/protocols/model-run-job-normalized-results-template.csv`.

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

## Kickoff Procedure

Use this sequence for every model-run job:

1. Identify the job type, manifest path, target runner, repo ref, and expected hardware.
2. Verify the manifest contract and compute or confirm the manifest SHA-256.
3. Capture the repository identity:
   - `git rev-parse HEAD`
   - `git submodule status --recursive`
   - `git status --short`
4. Capture the runner identity:
   - hostname
   - OS release
   - Python executable and version
   - installed package environment needed by the selected test lane
   - accelerator inventory (`nvidia-smi` for NVIDIA, TT device inventory for TT jobs)
   - disk headroom for repository, scratch, model cache, and artifact directories
5. Run collectability or targeted dry-run validation for the selected job type.
6. Create per-run output directories before execution:
   - `artifacts/<run-id>/`
   - `artifacts/<run-id>/waves/`
   - `artifacts/<run-id>/normalized/`
7. Launch bounded waves or single-row runs with the protocol command shape.
8. Normalize terminal results and preserve raw outcomes.
9. Record local checkpoints with evidence paths after every major handoff, terminalization batch, stop condition, or blocker.
10. Prepare external report text only as a draft until a human approves posting.

## Launch Record Template

Every kickoff must produce a launch record using `docs/protocols/model-run-job-launch-record-template.md`. At minimum, the record must include these fields:

- `run_id`
- `job_type`
- `manifest_path`
- `manifest_sha256`
- `repo_ref`
- `submodule_refs`
- `runner_host`
- `hardware`
- `environment_summary`
- `artifact_root`
- `command_template`
- `row_scope`
- `wave_policy`
- `cleanup_policy`
- `stop_conditions`
- `human_review_required_before_external_post`

## Stop Conditions

Stop or pause dispatch when any of these conditions occurs:

- Runner host becomes unreachable or cannot complete login/session setup.
- Disk free space drops below the run-specific floor.
- Accelerator inventory is missing or changes unexpectedly.
- Authentication, dependency, or environment failures affect multiple rows and are likely pipeline-class failures.
- Artifact capture fails for a terminal row.
- The manifest, repo ref, or submodule ref no longer matches the launch record.
- A human-review gate is reached for issue comments, PR updates, or stakeholder reports.

When a stop condition blocks launch, execution, normalization, or reporting, create a blocker record from `docs/protocols/model-run-job-blocker-template.md` and link it from the launch record.

## Normalized Results Contract

Use `docs/protocols/model-run-job-normalized-results-template.csv` as the header for normalized row outputs. Required outcome fields:

- `status_raw`: original runner status, return code, or harness status.
- `outcome_normalized`: one of `validated_pass`, `validated_fail`, `pipeline_error`, `blocked_collectability`, `needs_mapping`, or `pending_terminalization`.
- `outcome_class`: `model_quality`, `pipeline`, `planning`, or `pending`.
- `evidence_path`: row artifact directory or wave artifact directory.

Preserve raw logs and JUnit paths even when normalization classifies the row as a pipeline or planning issue.

## Reporting Rules

- Do not post GitHub comments, update issues, or publish stakeholder reports without human review.
- Status updates must distinguish known source coverage from requested target coverage.
- If the requested target count is larger than the available source count, report the gap explicitly.
- Every external report must include source link, source SHA-256, repo ref, command line, and evidence paths.
- Routine progress checkpoints are local artifacts unless a human explicitly approves external posting.

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
