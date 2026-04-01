# Find regression boundaries for all failed tests

Given a failures JSON produced by `/collect-failures`, walk backward through previous CI runs of the same workflow for each failed test to find the boundary: the oldest run with the **same error signature** and the immediately preceding run where the error was **different or the test did not exist**.

Each test is investigated by a parallel subagent. Logs are downloaded once per run and cached in `bisection/logs/` for reuse across all subagents.

## Usage

```
/find-regression-boundaries <failures_json>
```

**Arguments** (from `$ARGUMENTS`):
- `failures_json` — path to a failures JSON file produced by `/collect-failures`, e.g. `bisection/run_23375485557_failures.json`

**Examples:**
```
/find-regression-boundaries bisection/run_23375485557_failures.json
/find-regression-boundaries bisection/run_23578377117_failures.json
```

---

## Instructions for Claude

Determine the repository root at the start:
```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
```

The GitHub repo is always `tenstorrent/tt-xla`. Use `GITHUB_REPO=tenstorrent/tt-xla` throughout.

All bisection data lives under `$REPO_ROOT/bisection/`:
- `bisection/logs/` — cached run logs (shared across all subagents)
- `bisection/logs/index.json` — source of truth: which runs are downloaded
- `bisection/logs/run_{run_id}/` — extracted logs for a run
- `bisection/regression_report_<stem>.json` — final output of this skill (stem derived from input filename, e.g. `run_23375485557_failures.json` → `regression_report_23375485557.json`)

---

### Phase 1 — Load the failures JSON

Read the file at the path provided in `$ARGUMENTS` (resolve relative to `$REPO_ROOT` if not absolute).

Extract:
- `run_id` — the starting run (the "bad" run where failures were observed)
- `workflow_id` — used to fetch previous runs of the same workflow
- `workflow_name` — for display
- `github_repo` — the `owner/repo` string (e.g. `tenstorrent/tt-xla`)
- `failed_tests` array — list of `{test_id, machine_type, machine_name, raw_error, ...}` objects

**Only process `failed_tests`.** The JSON may also contain a `timed_out_jobs` array — ignore it entirely. Timed-out jobs don't have individual test results, so regression boundaries cannot be determined for them.

Each entry in `failed_tests` is uniquely identified by its **(test_id, machine_type)** pair — the same `test_id` may appear more than once if it failed on different machine types. Treat every such pair as a separate investigation.

Print:
```
Loaded <N> failed tests from run <run_id> (<run_date>)
Workflow: <workflow_name>
Repo:     <github_repo>

Starting regression boundary search (max 10 runs back per test)...
```

---

### Phase 2 — Fetch the list of preceding runs

First, fetch the starting run's metadata (this works even if the run is still in_progress):
```bash
gh api "repos/<github_repo>/actions/runs/<run_id>" \
  --jq '{id: .id, head_sha: .head_sha, created_at: .created_at, conclusion: .conclusion}'
```

Save `starting_run_created_at` from this response. The starting run may still be in_progress (`conclusion == null`) — that is fine.

Then fetch up to 10 completed runs of this workflow that are older than or equal to the starting run's `created_at`, ordered newest first:

```bash
gh api "repos/<github_repo>/actions/workflows/<workflow_id>/runs?per_page=15&status=completed&created=<=%3C%3D<starting_run_created_at>" \
  --jq '.workflow_runs[] | {id: .id, head_sha: .head_sha, created_at: .created_at, conclusion: .conclusion}'
```

Discard runs with conclusion `skipped` only. Do **not** discard cancelled runs — a run with conclusion `cancelled` often means only a few jobs within it were cancelled while others completed normally; those completed jobs contain valid test data. Also discard the starting run itself if it appears in this list (it may appear if it just completed). Take up to 10 of the remaining runs — these are the runs to walk back through.

Save the ordered list of run IDs to walk: `[run_id, prev_run_1, prev_run_2, ..., prev_run_10]` (newest to oldest).

Print the run list:
```
Runs to search (newest → oldest):
  [0] run 23375485557  2026-03-21  sha=afe4486f  ← starting run (known failures)
  [1] run 23084026025  2026-03-14  sha=1e5781dc
  [2] run 22795337660  2026-03-07  sha=c77995f6
  ...
```

---

### Phase 3 — Verify logs are cached

**This skill does not download logs.** The starting run's logs were already downloaded by `/collect-failures`. Logs for all *previous* runs (the ones to walk back through) must also be present before this skill runs — download them manually or use `.claude/scripts/run_regression_batches.py` which handles downloading and then invokes this skill per batch.

Check that `$REPO_ROOT/bisection/logs/index.json` contains entries for all runs in the list. If any run is missing, print a warning and skip it during investigation (treat as `DOWNLOAD_FAILED`).

---

### Phase 4 — Investigate each test in parallel

Launch one subagent per **(test_id, machine_type)** pair using the Agent tool with `run_in_background: false`. Run all subagents **in parallel** (send all Agent tool calls in a single message).

Each subagent receives the following prompt (fill in the actual values):

---

**Subagent prompt template:**

```
You are investigating the regression boundary for one failed test on a specific machine type.

Repository root: <REPO_ROOT>

## Your task

Find the boundary: the two consecutive runs of the same workflow where the test error signature changed (or the test went from non-existent to failing), **on the same machine type**.

## Test to investigate

test_id: <test_id>   ← full pytest path, e.g. tests/runner/test_models.py::test_all_models_torch[yolov9/pytorch-S-single_device-inference]
machine_type: <machine_type>
machine_name: <machine_name>
known_error (raw, from the starting run): <raw_error>

## Ordered runs to search (newest → oldest, index 0 = starting run)

<paste the full run list from Phase 2>

All logs are already downloaded and extracted. Do NOT download any logs — use only what is cached.

## Log cache location

<REPO_ROOT>/bisection/logs/run_{run_id}/ contains log files for each run. Depending on how the run was downloaded, logs may be structured as:
- Flat files: `job_{id}.txt` (downloaded by `/collect-failures`)
- Flat files with prefix: `{n}_{job_name}.txt` (old ZIP-based downloads)
- Subdirectories: `{job_name}/1_job.txt` (downloaded by `run_regression_batches.py`)

Always use recursive grep (`grep -rln`) when searching — it handles all three layouts.

## Search procedure

For each run (starting at index 0, going to older runs):

1. Search the cached logs for the test_id **on the correct machine type**.

   `test_id` is the full pytest path (e.g. `tests/runner/test_models.py::test_all_models_torch[yolov9/...]`). **Always search for the exact `test_id` string — never attempt partial matches, fuzzy matching, or substitutions.** If the exact string is not present in the logs, the test does not exist in that run. Do not try to find renamed or similar variants; that is not your job and will produce incorrect boundaries.

   Log file names encode the machine type in their job name (e.g. `...test n150...`, `...test n300...`). First narrow to log files for the correct machine type:
   ```bash
   grep -rln "<test_id>" <REPO_ROOT>/bisection/logs/run_<run_id>/ | grep -i "<machine_type>"
   ```
   If that returns nothing, fall back to searching all log files:
   ```bash
   grep -rln "<test_id>" <REPO_ROOT>/bisection/logs/run_<run_id>/
   ```
   In the fallback case, check the `Machine name:` header of each candidate file to confirm it matches the expected machine type before using it. If no file for this machine type is found, treat the run as **NO_LOGS** (see step 3).

2. Classify the result:
   - Line contains `PASSED` → **PASSED** — **⚠ IMMEDIATE STOP. Do not read another line. Do not check another run. Return your result right now with last_good = this run.**
   - Line contains `FAILED` or `ERROR` or `CRASHED with signal 6` → **FAILED** — extract the raw error using the procedure in **Error extraction** below
   - Line contains `CRASHED with signal 9` → **NOT_RUN** (job was externally killed — the test was in-flight when the job was cancelled; no reliable pass/fail conclusion can be drawn) — skip this run, continue to older
   - Line found but no PASSED/FAILED/ERROR/CRASHED suffix → **NOT_RUN** (job timed out mid-run before this test executed) — skip this run, continue to older
   - No line at all → do **step 3** before deciding

## Error extraction

Find the failure detail block (`___ test_name ___` section) and extract the `E   ` line.

In GHA logs the pytest separator has only 1 underscore before the test name (`_ func_and_params __`). Derive `func_and_params` from `test_id` by taking the part after `::`, e.g.:
- `tests/runner/test_models.py::test_all_models_torch[yolov9/...]` → `test_all_models_torch[yolov9/...]`
- `tests/jax/single_chip/ops/test_slice.py::test_slice[3-97-1]` → `test_slice[3-97-1]`

```bash
grep -F -A 100 "_ <func_and_params>" <log_file> | grep "Z E   " | head -3
```

**Special case — `XlaRuntimeError: INTERNAL: Error code: 13`:**
This is an opaque wrapper. The real error is a `TT_FATAL` message printed to the test output just before the failure. Get the line number of the test's running log entry and search forward for `TT_FATAL`:
```bash
# Find the line where the test started running (use func_and_params = part after "::")
grep -n "Running.*<func_and_params>" <log_file>  # get line number N

# Search forward from there for TT_FATAL
sed -n "${N},$((N+500))p" <log_file> | grep "TT_FATAL" | head -3
```
The `TT_FATAL` line contains the actual error (e.g. `Out of Memory: Not enough space to allocate...`). Use that as `raw_error` instead of `Error code: 13`.

3. If no line found at all — distinguish three cases:

   **3a. Check whether any log files for this machine type were downloaded for the run:**
   ```bash
   ls <REPO_ROOT>/bisection/logs/run_<run_id>/ | grep -i "<machine_type>"
   ```
   - If **no files exist** for this machine type → **NO_LOGS** — the test ran on a job that was never downloaded. **Skip this run, continue to older.** Do NOT classify as NOT_FOUND. (This is common when only failing jobs' logs are cached.)

   **3b. Log files for this machine type exist — check if the test was collected but the job was killed before it ran:**
   The pytest log prints all collected test IDs at the very start of the job (with no status suffix). If a job is killed mid-run, tests that were collected but not yet executed will appear in the collection output but have no result line.

   Search for the full test_id anywhere in the log (collection line has no trailing status):
   ```bash
   grep -rn "<test_id>" <REPO_ROOT>/bisection/logs/run_<run_id>/
   ```
   - If a line is found with no PASSED/FAILED/ERROR/CRASHED suffix → **NOT_RUN** — skip this run, continue to older
   - If no line found at all in any log for this run (logs ARE present but test is absent) → **NOT_FOUND** — **skip this run, continue to older.** The test was not in the CI matrix for this run (e.g. newly added test, matrix variation).

4. If FAILED: compare the raw error to the `known_error`.

   **PCC errors** — both errors contain `PCC comparison failed`:
   - Extract the calculated PCC value from each error (the number after `pcc=`)
   - Compute the absolute difference between the two PCC values
   - Difference < 0.1 → record as **FAILED** with the error message, continue to older run
   - Difference >= 0.1 → record as **DIFFERENT_ERROR**, stop immediately

   **Non-PCC errors** (crash, OOM, compile error, etc.):
   - Same error type and root cause → record as **FAILED** with the error message, continue to older run
   - Different error type or root cause → record as **DIFFERENT_ERROR**, stop immediately

   **Mixed** (one is PCC, other is not):
   - Always record as **DIFFERENT_ERROR**, stop immediately

**⛔ HARD STOP RULE: The moment you classify a run as PASSED or DIFFERENT_ERROR — stop immediately. Do NOT add any more entries to `runs_checked`. Do NOT examine any more runs. Return your JSON result at once.**

Statuses that cause you to **skip** (continue to the next older run, do not stop): NOT_RUN, NO_LOGS, NOT_FOUND.

## Stop conditions

These are hard stops — the moment any of these is hit, record it and return the result immediately:

| Condition | boundary_found | first_bad | last_good |
|-----------|---------------|-----------|-----------|
| Current run is PASSED | **true** | oldest FAILED run in the walk | this run (PASSED) |
| Current run is DIFFERENT_ERROR | **true** | oldest FAILED run in the walk | this run (DIFFERENT_ERROR) |
| Reached 10 runs back with only FAILED/NOT_RUN/NO_LOGS/NOT_FOUND | **false** | — | regression predates available history |

**"oldest FAILED run in the walk"** — this is always at minimum the starting run (index 0), since the starting run is always FAILED (it is the source run where the failure was observed). Even if the very first run checked (index 1) is PASSED or DIFFERENT_ERROR, the boundary is still valid: first_bad = starting run, last_good = run at index 1.

**`runs_checked` must stop at the stop condition.** The last entry should be the run that triggered the stop — do not include any runs examined after it.

## Output

Return a JSON object (do not write any files):
```json
{
  "test_id": "<test_id>",
  "machine_type": "<machine_type>",
  "machine_name": "<machine_name>",
  "boundary_found": true,
  "first_bad_run_id": <run_id>,
  "first_bad_run_date": "<date>",
  "first_bad_sha": "<sha>",
  "last_good_run_id": <run_id>,        // null when last_good_status is NOT_FOUND
  "last_good_run_date": "<date>",      // null when last_good_status is NOT_FOUND
  "last_good_sha": "<sha>",            // null when last_good_status is NOT_FOUND
  "last_good_status": "PASSED | DIFFERENT_ERROR | NOT_FOUND",
  "last_good_raw_error": "<raw error from last_good run, or null if PASSED/NOT_FOUND>",
  "known_error": "<the original error from the starting run>",
  "runs_checked": [
    {"run_id": ..., "date": "...", "status": "FAILED | DIFFERENT_ERROR | PASSED | NOT_RUN | NOT_FOUND", "raw_error": "<always include the error message, or null for PASSED/NOT_RUN/NOT_FOUND>"}
  ]

IMPORTANT: every entry in runs_checked MUST have a raw_error field. For FAILED and DIFFERENT_ERROR always include the actual error message extracted from that run's logs — never omit it.
}
```

If no boundary found (ran out of history):
```json
{
  "test_id": "<test_id>",
  "machine_type": "<machine_type>",
  "machine_name": "<machine_name>",
  "boundary_found": false,
  "reason": "Reached 10 run limit with no boundary",
  "known_error": "<original error>",
  "first_bad_run_id": <run_id of the earliest run where test failed>,
  "first_bad_run_date": "<date of first_bad run>",
  "first_bad_sha": "<sha of first_bad run>",
  "last_good_run_id": null,
  "last_good_run_date": null,
  "last_good_sha": null,
  "runs_checked": [...]
}
```
```

---

### Phase 5 — Collect results and save report

After all subagents complete, collect their JSON outputs.

**Derive the output filename from the input filename** (not from `run_id`):
- Take the basename of the input file (e.g. `batch_0_23375485557_failures.json`)
- Strip the `_failures.json` suffix → `batch_0_23375485557`
- Strip a leading `run_` prefix if present → e.g. `run_23375485557` → `23375485557`
- Output path: `$REPO_ROOT/bisection/regression_report_<stem>.json`

Examples:
- `batch_0_23375485557_failures.json` → `bisection/regression_report_batch_0_23375485557.json`
- `run_23375485557_failures.json` → `bisection/regression_report_23375485557.json`

Save to that path:
```json
{
  "source_run_id": <run_id>,
  "source_run_date": "<date>",
  "workflow_name": "<workflow_name>",
  "github_repo": "<github_repo>",
  "generated_at": "<current UTC timestamp>",
  "total_tests": <N>,
  "boundaries_found": <count>,
  "boundaries_not_found": <count>,
  "results": [
    {
      "test_id": "tests/runner/test_models.py::test_all_models_torch[densenet/pytorch-169-single_device-inference]",
      "machine_type": "n150",
      "machine_name": "forge-n150-12",
      "boundary_found": true,
      "first_bad_run_id": 23375485557,
      "first_bad_run_date": "2026-03-21",
      "first_bad_sha": "afe4486f",
      "last_good_run_id": 22795337660,
      "last_good_run_date": "2026-03-07",
      "last_good_sha": "c77995f6",
      "last_good_status": "PASSED",
      "last_good_raw_error": null,
      "known_error": "FAILED ...::test_all_models_torch[densenet/...] - AssertionError: PCC...",
      "runs_checked": [...]
    },
    ...
  ]
}
```

Use the Write tool to save this file.

---

### Phase 6 — Print summary table

```
Regression boundaries for: <workflow_name>  run <run_id>  (<run_date>)
<N> tests investigated

Test                                                                                                     | Machine  | First bad run   | Last good run  | Last good status
---------------------------------------------------------------------------------------------------------|----------|-----------------|----------------|------------------
tests/runner/test_models.py::test_all_models_torch[densenet/pytorch-169-single_device-inference]         | n150     | 2026-03-21 (#23375485557) | 2026-03-07 (#22795337660) | PASSED
tests/runner/test_models.py::test_all_models_torch[densenet/pytorch-169-single_device-inference]         | n300     | 2026-03-14 (#23084026025) | 2026-03-07 (#22795337660) | DIFFERENT_ERROR
tests/jax/single_chip/ops/test_slice.py::test_slice[3-97-1]                                             | n150     | 2026-03-14 (#23084026025) | 2026-03-07 (#22795337660) | DIFFERENT_ERROR
tests/runner/test_models.py::test_all_models_torch[mobilenetv3/pytorch-small-single_device-inference]    | n150     | (boundary not found — reached limit)
...

Saved to: bisection/regression_report_<stem>.json
```

---

## Error handling

| Situation | Action |
|-----------|--------|
| `git rev-parse` fails (not inside a repo) | Stop and tell the user to run this from inside the repository |
| Failures JSON not found | Report path tried, tell user to run `/collect-failures` first |
| Fewer than 2 runs available in history | Report to user — not enough history to find a boundary |
| Subagent returns malformed JSON | Use empty runs_checked and mark boundary_found: false with reason "subagent error" |
| All tests hit the 10-run limit | Warn user — the regression may predate available history; suggest fetching more runs manually |
