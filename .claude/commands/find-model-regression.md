# Find model test regression boundary

Walk back through GitHub Actions runs to find two consecutive runs of the same workflow where a given test **passed** in the earlier run and **failed** in the later run. Save the two boundary commit SHAs to a file for use with `/bisect-model`.

## Usage

```
/find-model-regression <test_id> <workflow_name_or_id>
```

**Arguments** (from `$ARGUMENTS`):
- `test_id` — pytest test ID as seen in CI logs, e.g. `densenet/pytorch-169-single_device-inference`
- `workflow_name_or_id` — GitHub Actions workflow name (partial match OK) or numeric workflow ID, e.g. `forge_models_torch` or `12345678`

**Examples:**
```
/find-model-regression densenet/pytorch-169-single_device-inference forge_models_torch
/find-model-regression yolov9/pytorch-C-single_device-inference 12345678
```

---

## Instructions for Claude

The project root is `/localdev/vzeljkovic/tt-xla_superset`. The GitHub repo is `tenstorrent/tt-xla`.

---

### Phase 1 — Resolve the workflow

Parse `$ARGUMENTS`: first token = `test_id`, remainder = `workflow_name_or_id`.

If the second argument is numeric, use it directly as `workflow_id`.

If it is a string, list all workflows and find the matching one:
```bash
gh api repos/tenstorrent/tt-xla/actions/workflows --jq '.workflows[] | {id, name, path}'
```
Pick the workflow whose `name` or `path` contains the provided string (case-insensitive). If multiple match, list them and ask the user to be more specific. Save `workflow_id`, `workflow_name`, and `workflow_path` (the `.path` field, e.g. `.github/workflows/schedule-weekly.yml`).

Derive `workflow_url` as:
```
https://github.com/tenstorrent/tt-xla/actions/workflows/<filename>
```
where `<filename>` is the basename of `workflow_path` (e.g. `schedule-weekly.yml`).

> Note: GitHub's web UI does not accept numeric workflow IDs in URLs — only the filename works.

---

### Phase 2 — Fetch recent runs

Fetch the 20 most recent completed runs of this workflow, newest first:
```bash
gh api "repos/tenstorrent/tt-xla/actions/workflows/<workflow_id>/runs?per_page=20&status=completed" \
  --jq '.workflow_runs[] | {id: .id, head_sha: .head_sha, created_at: .created_at, conclusion: .conclusion}'
```

Discard runs with conclusion `cancelled` or `skipped` — they give no signal.

Print the list to the user so they can see what runs are being searched.

---

### Phase 3 — Check each run for the test result

For each run (newest first), call the helper below.

**Helper: check_test_in_run(run_id, test_id)**

1. Get all jobs for the run:
   ```bash
   gh api "repos/tenstorrent/tt-xla/actions/runs/<run_id>/jobs?per_page=100" \
     --jq '.jobs[] | {id, name, conclusion}'
   ```
   Paginate if `total_count` > 100 (`&page=2`, etc.).

2. For each job that looks relevant (name contains `forge_models` or `torch`), fetch logs and grep for the test:
   ```bash
   gh api "repos/tenstorrent/tt-xla/actions/jobs/<job_id>/logs" 2>/dev/null | \
     grep "<test_id>"
   ```

3. Classify the result:
   - Line ends with `PASSED` → **PASSED** — record `job_id`
   - Line ends with `FAILED` or `ERROR` → **FAILED** — record `job_id`
   - Line present but no PASSED/FAILED suffix → **NOT_RUN** (job timed out before this test)
   - No line in any job → **NOT_FOUND** (test didn't exist in this run)

   When a PASSED or FAILED result is found, save the `job_id` of the job that contained the result — this is needed for the job-level URL.

Build and print a progress table as you go:
```
Run 23375485557  (2026-03-21)  sha=afe4486f  → FAILED
Run 23084026025  (2026-03-14)  sha=1e5781dc  → FAILED
Run 22795337660  (2026-03-07)  sha=c77995f6  → PASSED  ← boundary found!
```

**Decision rules:**

| Status     | Action |
|------------|--------|
| FAILED     | Record as a failing run. Continue to the next (older) run. |
| PASSED     | **Boundary candidate found.** Verify the immediately newer run was FAILED (it should already be recorded). If yes, boundary is confirmed — stop. |
| NOT_RUN    | Skip (inconclusive — job timed out). Continue. |
| NOT_FOUND  | The test did not exist yet. Stop walking — no boundary exists in available history. Report to user. |

If you reach 20 runs without finding a PASSED, fetch 20 more runs and continue. Warn the user after 40 runs with no boundary found.

---

### Phase 4 — Save boundary to file

Once the boundary is found (`good_run` → `bad_run`):

Save the results to `investigation/regression_<test_id_slug>.json` where `test_id_slug` is the test_id with `/` replaced by `_` and spaces replaced by `-`.

The file content:
```json
{
  "test_id": "<test_id>",
  "workflow_id": <workflow_id>,
  "workflow_name": "<workflow_name>",
  "workflow_url": "<workflow_url>",
  "good_run_id": <good_run_id>,
  "good_run_url": "https://github.com/tenstorrent/tt-xla/actions/runs/<good_run_id>/job/<good_job_id>",
  "good_sha": "<good_sha>",
  "good_run_date": "<good_run_created_at>",
  "bad_run_id": <bad_run_id>,
  "bad_run_url": "https://github.com/tenstorrent/tt-xla/actions/runs/<bad_run_id>/job/<bad_job_id>",
  "bad_sha": "<bad_sha>",
  "bad_run_date": "<bad_run_created_at>"
}
```

Create the `investigation/` directory if it doesn't exist:
```bash
mkdir -p /localdev/vzeljkovic/tt-xla_superset/investigation
```

Write the file using the Write tool (not shell redirection).

---

### Phase 5 — Report to user

Print a summary:
```
Regression boundary found for: <test_id>

  PASSED  run <good_run_id>  sha <good_sha>  (<good_run_date>)
  FAILED  run <bad_run_id>   sha <bad_sha>   (<bad_run_date>)

Saved to: investigation/regression_<test_id_slug>.json

To bisect, run:
  /bisect-model <test_id> <bad_run_id>
```

---

## Error handling notes

- **No PASSED found in history**: Report how far back was checked. Ask the user to provide a known-good run ID or SHA manually, then write the file with that manually provided data.
- **NOT_FOUND before any PASSED**: The test was added and immediately failed, or was never passing in available history. Report this clearly.
- **NOT_RUN (timeout) everywhere**: The test has never been reliably executed in CI. Warn the user — bisect may not be reliable.
- **Multiple workflows match**: List all matches, ask user to be more specific.
