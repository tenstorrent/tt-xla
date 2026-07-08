# TT-XLA Tools

## CI Failure Triage

`triage_ci_failure.py` creates a deterministic triage packet from a GitHub Actions job log or saved log file.

Prefer full job logs when posting a root-cause comment:

```bash
python3 tools/triage_ci_failure.py \
  --job-url https://github.com/tenstorrent/tt-xla/actions/runs/<run-id>/job/<job-id> \
  --run-url https://github.com/tenstorrent/tt-xla/actions/runs/<run-id> \
  --output-dir triage-output
```

Use `--input-source issue-body` only for local drafting from copied issue text:

```bash
python3 tools/triage_ci_failure.py \
  --job-log issue-body.txt \
  --input-source issue-body \
  --output-dir triage-output
```

Do not auto-post a packet as root cause unless `ready_to_post` is `true`. If `ready_to_post` is `false`, inspect `post_blockers` first. Common blockers are `full_log_evidence_missing` and `multiple_test_selectors_without_junit`.

When available, pass JUnit XML with `--junit` to bind failures to exact pytest selectors. Without JUnit, the tool binds matched evidence to the nearest preceding pytest selector in the log.
