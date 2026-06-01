# Model Run Job Reboot Record

Use this template when a blocked model-run job needs runner reboot recovery. Keep the completed record with the run artifacts and link it from the launch or blocker record.

## Identity

- `run_id`:
- `reboot_id`:
- `created_at_utc`:
- `job_type`:
- `recovery_operation`: `runner_reboot_recovery`
- `manifest_path`:
- `repo_ref`:
- `runner_host`:

## Reboot Authority

- `automatic_reboot_allowed`: `false`
- `approval_source`:
- `approval_recorded_at_utc`:
- `trigger_threshold`:
- `rollback_or_escalation_owner`:

## Target

- `provider`:
- `instance_id`:
- `region`:
- `profile_or_account_scope`:
- `target_identity_source`:
- `single_instance_confirmed`: `false`

## Pre-Reboot State

- `last_known_good_state`:
- `terminal_rows_at_last_proof`:
- `pending_rows_at_last_proof`:
- `active_process_snapshot`:
- `failed_access_attempts`:
- `pre_reboot_health`:
- `read_only_cloud_status_evidence`:
- `duplicate_dispatch_risk`: `true`

## Reboot Command

- `reboot_command`:
- `started_at_utc`:
- `ended_at_utc`:
- `terminal_status`:
- `stdout_excerpt`:
- `stderr_excerpt`:
- `command_evidence_path`:

## Post-Reboot Poll

- `post_reboot_poll`:
- `poll_started_at_utc`:
- `poll_ended_at_utc`:
- `poll_terminal_status`:
- `runner_session_evidence`:
- `post_reboot_health`:
- `run_state_after_reboot`:

## Resume Gate

- `resume_condition`:
- `resume_condition_satisfied`: `false`
- `resumed_at_utc`:
- `blocker_record_if_not_resumed`:
- `external_reporting_allowed`: `false`
