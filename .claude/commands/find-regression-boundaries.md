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
- `bisection/regression_report_{run_id}.json` — final output of this skill

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

Fetch up to 15 completed runs of this workflow, ordered newest first (we need 10 runs before the starting run, plus the starting run itself):

```bash
gh api "repos/<github_repo>/actions/workflows/<workflow_id>/runs?per_page=15&status=completed" \
  --jq '.workflow_runs[] | {id: .id, head_sha: .head_sha, created_at: .created_at, conclusion: .conclusion}'
```

Discard runs with conclusion `cancelled` or `skipped`.

Find the position of `run_id` (the starting run) in this list. The runs **before it** (older) are the runs to walk back through — take up to 10 of them.

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

### Phase 3 — Ensure logs are cached for all runs

For each run in the list, check `$REPO_ROOT/bisection/logs/index.json`. For any run not yet cached, download and extract the logs **before** launching subagents — this avoids race conditions where multiple subagents try to download the same run simultaneously.

**Cache procedure for a single run** (repeat for each uncached run):

1. Fetch run metadata (sha, date):
```bash
gh api "repos/<github_repo>/actions/runs/<run_id>" \
  --jq '{id: .id, workflow_id: .workflow_id, workflow_name: (.name), head_sha: .head_sha, created_at: .created_at}'
```

2. Create the directory:
```bash
mkdir -p "$REPO_ROOT/bisection/logs/run_<run_id>"
```

3. Download the full logs ZIP:
```bash
gh api "repos/<github_repo>/actions/runs/<run_id>/logs" \
  > "$REPO_ROOT/bisection/logs/run_<run_id>.zip"
```

4. Extract:
```bash
cd "$REPO_ROOT/bisection/logs/run_<run_id>"
unzip -o ../run_<run_id>.zip
```

5. Remove ZIP:
```bash
rm "$REPO_ROOT/bisection/logs/run_<run_id>.zip"
```

6. Update `$REPO_ROOT/bisection/logs/index.json` with the new entry (merge — do not overwrite existing entries).

Print progress as each run is downloaded: `Downloaded logs for run <run_id> (<date>)`.

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

test_id: <test_id>
machine_type: <machine_type>
machine_name: <machine_name>
known_error (raw, from the starting run): <raw_error>

## Ordered runs to search (newest → oldest, index 0 = starting run)

<paste the full run list from Phase 2>

All logs are already downloaded and extracted. Do NOT download any logs — use only what is cached.

## Log cache location

<REPO_ROOT>/bisection/logs/run_{run_id}/ contains .txt files with the logs for each run.

## Search procedure

For each run (starting at index 0, going to older runs):

1. Search the cached logs for the test_id **on the correct machine type**.

   Log file names encode the machine type in their job name (e.g. `...test n150...`, `...test n300...`). First narrow to log files for the correct machine type:
   ```bash
   grep -rln "<test_id>" <REPO_ROOT>/bisection/logs/run_<run_id>/ | grep -i "<machine_type>"
   ```
   If that returns nothing, fall back to searching all log files:
   ```bash
   grep -rln "<test_id>" <REPO_ROOT>/bisection/logs/run_<run_id>/
   ```
   In the fallback case, check the `Machine name:` header of each candidate file to confirm it matches the expected machine type before using it. If no file for this machine type is found, treat the run as **NOT_FOUND** for this (test_id, machine_type) pair.

2. Classify the result:
   - Line contains `PASSED` → **PASSED**
   - Line contains `FAILED` or `ERROR` → **FAILED** — extract the raw error using the procedure in **Error extraction** below
   - Line found but no PASSED/FAILED/ERROR suffix → **NOT_RUN** (job timed out mid-run before this test executed) — skip this run, continue to older
   - No line at all → do **step 3** before deciding

## Error extraction

Find the failure detail block (`___ test_name ___` section) and extract the `E   ` line:
```bash
grep -A 30 "____.*<test_id>.*____" <log_file> | grep "^[^Z]*E   " | head -3
```

**Special case — `XlaRuntimeError: INTERNAL: Error code: 13`:**
This is an opaque wrapper. The real error is a `TT_FATAL` message printed to the test output just before the failure. Get the line number of the test's running log entry and search forward for `TT_FATAL`:
```bash
# Find the line where the test started running
grep -n "Running.*<test_id>" <log_file>  # get line number N

# Search forward from there for TT_FATAL
sed -n "${N},$((N+500))p" <log_file> | grep "TT_FATAL" | head -3
```
The `TT_FATAL` line contains the actual error (e.g. `Out of Memory: Not enough space to allocate...`). Use that as `raw_error` instead of `Error code: 13`.

3. If no line found at all — distinguish two cases:

   **3a. Check if the test was collected but the job was killed before it ran:**
   The pytest log prints all collected test IDs at the very start of the job (with no status suffix). If a job is killed mid-run, tests that were collected but not yet executed will appear in the collection output but have no result line.

   Search for the test_id anywhere in the log (collection line has no trailing status):
   ```bash
   grep -rn "test_all_models_torch\[<test_id>\]" <REPO_ROOT>/bisection/logs/run_<run_id>/
   ```
   - If a line is found with no PASSED/FAILED/ERROR suffix → **NOT_RUN** — skip this run, continue to older
   - If no line found at all → **NOT_FOUND** — stop here

4. If FAILED: compare the raw error to the `known_error`.

   **PCC errors** — both errors contain `PCC comparison failed`:
   - Extract the calculated PCC value from each error (the number after `pcc=`)
   - Compute the absolute difference between the two PCC values
   - Difference < 0.1 → **SAME_ERROR**, continue to older run
   - Difference >= 0.1 → **DIFFERENT_ERROR**, stop immediately

   **Non-PCC errors** (crash, OOM, compile error, etc.):
   - Same error type and root cause → **SAME_ERROR**, continue to older run
   - Different error type or root cause → **DIFFERENT_ERROR**, stop immediately

   **Mixed** (one is PCC, other is not):
   - Always **DIFFERENT_ERROR**, stop immediately

   - If PASSED: record as PASSED, **stop immediately — do not check any more runs**

**IMPORTANT: As soon as you record PASSED, DIFFERENT_ERROR, or NOT_FOUND, you MUST stop. Do not process any further runs.**

## Stop conditions

These are hard stops — the moment any of these is hit, record it and return the result immediately:

| Condition | boundary_found | first_bad | last_good |
|-----------|---------------|-----------|-----------|
| Current run is PASSED | **true** | most recent SAME_ERROR run | this run (PASSED) |
| Current run is DIFFERENT_ERROR | **true** | most recent SAME_ERROR run | this run (DIFFERENT_ERROR) |
| Current run is NOT_FOUND | **true** | starting run | NOT_FOUND |
| Reached 10 runs back with only SAME_ERROR/NOT_RUN | **false** | — | regression predates available history |

**"most recent SAME_ERROR run"** — this is always at minimum the starting run (index 0). Even if the very first run checked (index 1) is PASSED or DIFFERENT_ERROR, the boundary is still valid: first_bad = starting run, last_good = run at index 1.

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
  "last_good_run_id": <run_id>,
  "last_good_run_date": "<date>",
  "last_good_sha": "<sha>",
  "last_good_status": "PASSED | DIFFERENT_ERROR | NOT_FOUND",
  "last_good_raw_error": "<raw error from last_good run, or null if PASSED/NOT_FOUND>",
  "known_error": "<the original error from the starting run>",
  "runs_checked": [
    {"run_id": ..., "date": "...", "status": "SAME_ERROR | DIFFERENT_ERROR | PASSED | NOT_RUN | NOT_FOUND", "raw_error": "<always include the error message, or null for PASSED/NOT_RUN/NOT_FOUND>"}
  ]

IMPORTANT: every entry in runs_checked MUST have a raw_error field. For SAME_ERROR and DIFFERENT_ERROR always include the actual error message extracted from that run's logs — never omit it.
}
```

If no boundary found (ran out of history):
```json
{
  "test_id": "<test_id>",
  "machine_type": "<machine_type>",
  "machine_name": "<machine_name>",
  "boundary_found": false,
  "reason": "Reached 10 run limit with no boundary | NOT_FOUND before any SAME_ERROR",
  "known_error": "<original error>",
  "runs_checked": [...]
}
```
```

---

### Phase 5 — Collect results and save report

After all subagents complete, collect their JSON outputs.

Save to `$REPO_ROOT/bisection/regression_report_<run_id>.json`:
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
      "test_id": "densenet/pytorch-169-single_device-inference",
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

Test                                                      | Machine  | First bad run   | Last good run  | Last good status
----------------------------------------------------------|----------|-----------------|----------------|------------------
densenet/pytorch-169-single_device-inference              | n150     | 2026-03-21 (#23375485557) | 2026-03-07 (#22795337660) | PASSED
densenet/pytorch-169-single_device-inference              | n300     | 2026-03-14 (#23084026025) | 2026-03-07 (#22795337660) | DIFFERENT_ERROR
yoloworld/pytorch-Xlarge_640-single_device-inference      | n150     | 2026-03-14 (#23084026025) | 2026-03-07 (#22795337660) | DIFFERENT_ERROR
mobilenetv3/pytorch-small-single_device-inference         | n150     | (boundary not found — reached limit)
...

Saved to: bisection/regression_report_<run_id>.json
```

---

## Error handling

| Situation | Action |
|-----------|--------|
| `git rev-parse` fails (not inside a repo) | Stop and tell the user to run this from inside the repository |
| Failures JSON not found | Report path tried, tell user to run `/collect-failures` first |
| Fewer than 2 runs available in history | Report to user — not enough history to find a boundary |
| Log ZIP download fails for a run | Skip that run in the walk, note it in the result as `DOWNLOAD_FAILED` |
| Subagent returns malformed JSON | Use empty runs_checked and mark boundary_found: false with reason "subagent error" |
| All tests hit the 10-run limit | Warn user — the regression may predate available history; suggest fetching more runs manually |
