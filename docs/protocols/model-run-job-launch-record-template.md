# Model Run Job Launch Record

Use this template before launching a bounded model-run job. Keep the completed record with the run artifacts and update it when execution pauses, resumes, or stops.

## Identity

- `run_id`:
- `job_type`:
- `created_at_utc`:
- `created_by`:
- `human_review_required_before_external_post`: `true`

## Source

- `manifest_path`:
- `manifest_sha256`:
- `manifest_type`:
- `source_file`:
- `source_sha256`:
- `selection_rule`:
- `selection_date`:
- `row_scope`:

## Repository

- `repo_path`:
- `repo_ref`:
- `repo_status_short`:
- `submodule_refs`:
- `local_only_refs`:

## Runner

- `runner_host`:
- `hardware`:
- `os_release`:
- `python_executable`:
- `python_version`:
- `environment_summary`:
- `accelerator_inventory`:
- `disk_headroom`:

## Execution

- `artifact_root`:
- `command_template`:
- `collectability_command`:
- `wave_policy`:
- `timeout_policy`:
- `cleanup_policy`:
- `stop_conditions`:

## Evidence Paths

- `manifest_copy`:
- `environment_snapshot`:
- `collectability_log`:
- `pytest_logs`:
- `junit_files`:
- `normalized_results`:
- `checkpoint`:

## Outcome Summary

- `terminal_rows`:
- `validated_pass`:
- `validated_fail`:
- `pipeline_error`:
- `blocked_collectability`:
- `needs_mapping`:
- `pending_terminalization`:

## Human Review Gates

- `issue_comment_draft`:
- `pr_update_draft`:
- `stakeholder_report_draft`:
- `approved_external_posts`:
