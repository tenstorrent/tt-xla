---
name: tracy-ird-proof
description: Guide Claude or Codex agents through tt-xla Tracy performance proof work, especially the issue 5009 IRD profiling harness. Use when updating or validating tests/benchmark/scripts/ttxla_profile_pipeline.py, tests/benchmark/PROFILING.md, Tracy output artifacts, IRD lifecycle evidence, tt-perf-report output, or the related PR proof packet.
---

# Tracy IRD Proof

Use this skill when the work involves Tracy profiling, tt-perf-report output, IRD execution, or evidence for the issue 5009 profiling pipeline.

This file is the canonical runbook. Tool-specific skill files should stay as thin adapters that point here:

- `.claude/skills/tracy-ird-proof/SKILL.md`
- `.codex/skills/tracy-ird-proof/SKILL.md`
- `.opencode/skills/tracy-ird-proof/SKILL.md`

## Start With Repo Practice

Before editing or reporting status, inspect the repository-local guidance that applies to the change:

- `README.md`
- `.github/pull_request_template.md`
- `tests/benchmark/PROFILING.md`
- `tests/benchmark/scripts/ttxla_profile_pipeline.py`
- `tests/benchmark/test_ttxla_profile_pipeline.py`
- recent `.claude/skills/*/SKILL.md` files when changing Claude adapters
- `.agents/skills/*/SKILL.md` when changing shared agent runbooks

Follow the PR template exactly when updating the pull request body. Use full GitHub, Google Drive, or other external links in PR and issue text. Do not use local workstation paths in stakeholder-facing descriptions.

## Main Goal

The Tracy profiling proof workflow should let a developer:

- discover the benchmark manifest,
- run bounded model profiles locally or on IRD,
- collect IR, Tracy, and tt-perf-report evidence,
- clean IRD state before and after a short reservation or `ird run`,
- classify every discovered model with an explicit status,
- render `dashboard.html`, `claude-report-packet.html`, and `report.html`,
- provide reproducible evidence for the PR or GitHub issue.

## Implementation Grounding

Reuse repository terminology and constants. Do not invent new status strings when a matching repo term exists.

- Use `not_started` for a discovered model that has no terminal `status.json`.
- `not_started` follows the existing `BringupStatus.NOT_STARTED` style used by repository model status reporting.
- Do not use `pending_terminalization`.
- Prefer existing module constants in `tests/benchmark/scripts/ttxla_profile_pipeline.py`, such as `TAXONOMY_NOT_STARTED`, `TAXONOMY_NOT_RUN`, and `RUN_STATUS_NOT_RUN`, instead of repeated string literals.

When adding behavior, keep the script style consistent with the surrounding code:

- centralize taxonomy or terminal-state strings as module constants,
- write structured JSON artifacts through existing helpers,
- keep shell command execution observable in `command-trace.jsonl`,
- avoid broad refactors outside the profiling harness surface.

## IRD Execution Rules

Use the harness rather than n8n for this issue unless the user asks for external orchestration.

Preferred mode is a short `ird run` job because the scheduler owns teardown. If explicit reservation mode is required, use the harness `--ird-mode reserve` path and provide cleanup, run, release, and post-cleanup command templates.

Evidence must show cleanup before and after use when a reservation is involved:

- configured pre-cleanup command,
- reservation identity or tag,
- remote run command,
- release command,
- post-cleanup command,
- terminal return code,
- cleanup failure details if cleanup did not complete.

The expected local evidence files are:

- `manifest.json`
- `environment.json`
- `model-manifest.json`
- `requirements.json`
- `command-trace.jsonl`
- `ird/ird-lifecycle.json` for IRD runs
- `profiles/<model-id>/status.json`
- `profiles/<model-id>/ir/`
- `profiles/<model-id>/perf-report/`
- `dashboard.html`
- `claude-report-packet.html`
- `report.html`

## Tracy Output Meaning

A Tracy run can produce several artifact types:

- `ops_perf_results_<timestamp>.csv`: per-operation device performance rows used by `tt-perf-report`.
- `profile_log_device.csv`: raw device profiling data; this can be very large and is usually pruned by the harness.
- `tracy_profile_log_host.tracy`: host trace file for the Tracy GUI.
- `tt-perf-report` output: derived slow-op analysis, throughput context, and optimization hints based on the ops CSV.

The final stakeholder report should be generated from the source packet and evidence artifacts, not from free-form notes alone.

## Validation Commands

Run the focused validation that matches the touched files. The usual issue 5009 branch gate is:

```bash
.venv/bin/python -m black --check tests/benchmark/scripts/ttxla_profile_pipeline.py tests/benchmark/test_ttxla_profile_pipeline.py
.venv/bin/python -m isort --check-only --profile black tests/benchmark/scripts/ttxla_profile_pipeline.py tests/benchmark/test_ttxla_profile_pipeline.py
python3 -m py_compile tests/benchmark/scripts/ttxla_profile_pipeline.py tests/benchmark/test_ttxla_profile_pipeline.py
.venv/bin/python -m pytest --noconftest tests/benchmark/test_ttxla_profile_pipeline.py -q
.venv/bin/python -m radon cc tests/benchmark/scripts/ttxla_profile_pipeline.py tests/benchmark/test_ttxla_profile_pipeline.py -s
.venv/bin/python -m radon mi tests/benchmark/scripts/ttxla_profile_pipeline.py tests/benchmark/test_ttxla_profile_pipeline.py -s
```

If the local virtual environment uses a different path, adapt only the Python executable path and keep the command intent the same.

For generated HTML artifacts, also verify that:

- `dashboard.html` exists and has model rows,
- `claude-report-packet.html` exists and references the source evidence,
- `report.html` exists and can be opened as the final report,
- every report claim can be traced to a JSON, CSV, command trace, lifecycle file, or GitHub Actions link.

## PR And Issue Updates

When updating a PR or GitHub issue:

- use the repository PR template sections first,
- post multi-line GitHub text with `gh ... --body-file`,
- link tracked files with full GitHub URLs,
- link external artifacts through Google Drive or another accessible artifact repository,
- state whether current CI is still running or green for the exact head commit,
- call out blocked validation plainly when the environment is missing IRD, TT hardware, Tracy, or tt-perf-report.

For detached worktrees, confirm the current branch and push target before pushing. For the issue 5009 story branch, the expected push form is:

```bash
git push origin HEAD:story/issue-5009-ird-profile-harness
```
