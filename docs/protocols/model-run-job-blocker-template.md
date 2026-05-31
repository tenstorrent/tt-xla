# Model Run Job Blocker Record

Use this template when a model-run job cannot start, continue, normalize, or report because a precondition or invariant is not satisfied. Keep the completed record with the run artifacts and link it from the launch record.

## Identity

- `run_id`:
- `blocker_id`:
- `created_at_utc`:
- `job_type`:
- `manifest_path`:
- `repo_ref`:
- `runner_host`:

## Classification

- `failure_taxonomy`: `precondition_violation`
- `blocked_stage`: `launch | collectability | execution | normalization | reporting`
- `blocking_condition`:
- `first_observed_at_utc`:
- `latest_observed_at_utc`:
- `repeat_count`:

## Evidence

- `failed_command`:
- `exit_code`:
- `stderr_excerpt`:
- `stdout_excerpt`:
- `evidence_path`:
- `last_known_good_state`:

## Impact

- `terminal_rows_at_last_proof`:
- `pending_rows_at_last_proof`:
- `duplicate_dispatch_risk`: `true`
- `external_reporting_allowed`: `false`

## Correction Plan

- `next_safe_action`:
- `retry_policy`:
- `fallback_path`:
- `human_intervention_needed`:
- `resume_condition`:
