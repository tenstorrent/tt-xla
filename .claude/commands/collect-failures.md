# Collect failed tests from a CI run

Fetch all non-succeeded jobs from a GitHub Actions run using the Jobs API as the source of truth, then enrich jobs with log-parsed details based on their failure type.

## Usage

```
/collect-failures <run_url>
```

**Arguments** (from `$ARGUMENTS`):
- `run_url` — GitHub Actions run URL, e.g. `https://github.com/tenstorrent/tt-xla/actions/runs/23375485557`

**Examples:**
```
/collect-failures https://github.com/tenstorrent/tt-xla/actions/runs/23375485557
/collect-failures https://github.com/tenstorrent/tt-xla/actions/runs/23578377117
```

---

## Instructions for Claude

Determine the repository root at the start:
```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
```

The GitHub repo is always `tenstorrent/tt-xla`. Use `GITHUB_REPO=tenstorrent/tt-xla` throughout.

All bisection data lives under `$REPO_ROOT/bisection/`:
- `bisection/logs/` — cached run logs and index
- `bisection/logs/index.json` — source of truth for which runs have been downloaded
- `bisection/logs/run_{run_id}/` — extracted log files for a run
- `bisection/run_{run_id}_failures.json` — output of this skill

---

### Phase 1 — Parse the run URL

Parse `$ARGUMENTS` as a GitHub Actions run URL. Extract `run_id` from the URL — it is the numeric segment after `/runs/`:

```
https://github.com/tenstorrent/tt-xla/actions/runs/23375485557
                                                      ^^^^^^^^^^^
                                                      run_id = 23375485557
```

Fetch run metadata:
```bash
gh api "repos/$GITHUB_REPO/actions/runs/<run_id>" \
  --jq '{id: .id, workflow_id: .workflow_id, workflow_name: (.name), head_sha: .head_sha, created_at: .created_at, conclusion: .conclusion}'
```

Save: `run_id`, `workflow_id`, `workflow_name`, `run_sha`, `run_date`.

Print to user:
```
Workflow: <workflow_name>
Run ID:   <run_id>
Date:     <run_date>
SHA:      <run_sha>
```

---

### Phase 2 — Categorize all non-succeeded jobs via the Jobs API

This is the **only** API call needed to determine job outcomes. Do NOT rely on log parsing to determine job conclusions.

Fetch all jobs for the run (paginate if needed — use `per_page=100`):
```bash
gh api "repos/$GITHUB_REPO/actions/runs/<run_id>/jobs?per_page=100" \
  --jq '.jobs[] | {id: .id, name: .name, conclusion: .conclusion, html_url: .html_url}'
```

If there are more than 100 jobs, fetch additional pages until `.jobs` is empty.

Partition into four active buckets based on `status` and `conclusion`:
- **failed** — `conclusion == "failure"` → full log parsing (individual tests + error messages)
- **cancelled** — `conclusion == "cancelled"` → full log parsing (individual tests + error messages); these are jobs killed with SIGKILL (signal 9)
- **timed_out** — `conclusion == "timed_out"` → limited log parsing (see Phase 5)
- **not_finished** — `status == "in_progress"` and `conclusion == null` (job is still running) → no log parsing; record as-is

Jobs with `conclusion == "success"`, `conclusion == "neutral"` and `conclusion == "skipped"` are ignored entirely.

Print a summary after categorization:
```
Job breakdown:
  failed:       <N>
  cancelled:    <N>
  timed_out:    <N>
  not_finished: <N>
```

---

### Phase 3 — Check cache, download logs if needed

Logs are needed for **failed**, **cancelled**, and **timed_out** jobs. If there are none of these, skip this phase.

1. Ensure the log directory exists:
```bash
mkdir -p "$REPO_ROOT/bisection/logs/run_<run_id>"
```

2. Read `$REPO_ROOT/bisection/logs/index.json` (if it exists). It tracks which job logs have already been downloaded under `run_<run_id>.downloaded_job_ids` (an array of job IDs).

3. For each job in the **failed**, **cancelled**, and **timed_out** buckets, check whether its log is already cached:

```bash
# Log file path for a job
LOG_FILE="$REPO_ROOT/bisection/logs/run_<run_id>/job_<job_id>.txt"
```

- **If the file already exists:** skip download for this job, print `  [cached] job <job_id> (<job_name>)`.
- **If the file does not exist:** download it:
```bash
gh api "repos/$GITHUB_REPO/actions/jobs/<job_id>/logs" \
  > "$REPO_ROOT/bisection/logs/run_<run_id>/job_<job_id>.txt"
```
  Print `  [downloaded] job <job_id> (<job_name>)`.

4. After downloading all needed logs, update `bisection/logs/index.json` — merge the entry (create the file if it doesn't exist, do not overwrite other run entries):
```json
{
  "run_<run_id>": {
    "run_id": <run_id>,
    "sha": "<run_sha>",
    "date": "<run_date>",
    "workflow_id": <workflow_id>,
    "workflow_name": "<workflow_name>",
    "downloaded_at": "<current UTC timestamp>",
    "downloaded_job_ids": [<job_id_1>, <job_id_2>, ...]
  }
}
```
`downloaded_job_ids` should be the union of any previously recorded IDs and the newly downloaded ones.

Use the Write tool to save this file.

---

### Phase 4 — Process failed and cancelled jobs (full log parsing)

Apply this phase to all jobs in the **failed** and **cancelled** buckets.

For each job, the log file is:
```
$REPO_ROOT/bisection/logs/run_<run_id>/job_<job_id>.txt
```

If the file does not exist (download failed), record `log_file` as `"not available"` and skip log parsing for that job.

#### Step 0 — Extract machine info from the log header

Every log file starts with runner metadata. Extract:

```bash
grep "Machine name:" "<log_file>" | head -1
# Output: Machine name: 'forge-n150-12'
```

Parse `machine_name` as the value inside the quotes (e.g. `forge-n150-12`).

Derive `machine_type` from `machine_name` by extracting the first token from the set `{n150, n300, p150, llmbox}` found anywhere in the name:
- `forge-n150-12` → `n150`
- `forge-n300-5` → `n300`
- `forge-p150-3` → `p150`
- `forge-llmbox-n300-1` → `llmbox`
- `forge-mlir-n150-7` → `n150`
- `forge-torch-p150-2` → `p150`

The only valid machine types are: `n150`, `n300`, `p150`, `llmbox`. Any prefix segments (e.g. `mlir`, `torch`) are ignored.

If the `Machine name:` line is missing, fall back to extracting machine type from the job name (look for the first parenthesized token matching a known type, e.g. `(n150, run_forge_models_torch, 1)` → `n150`).

#### Step 0b — Detect in-job timeouts and reclassify

Before parsing individual test results, check whether **any step** in the job timed out (even though the Jobs API reported it as `failure` rather than `timed_out`):

```bash
grep -ic "has timed out\|timed out after\|The operation was canceled" "<log_file>"
```

If this matches (count > 0), **do not process it in Phase 4.** Move it to the `timed_out` bucket and process it in Phase 5.

Save the full timeout message line as the timeout reason (used as `last_test_executing` context):
```bash
grep -i "has timed out\|timed out after\|The operation was canceled" "<log_file>" | head -3
```

To find the last test that was executing before the timeout, look for a "Last test:" hint near the timeout message:
```bash
grep -i "has timed out\|timed out after\|Last test:" "<log_file>" | tail -5
```
If "Last test: <test_id>" appears, use that as `last_test_executing`. Otherwise fall back to the standard Phase 5 log search.

#### Step 1 — Identify individual failed tests

**Include** (collect these):
- `FAILED` — test failed for any reason
- `CRASHED with signal 6/9` — process aborted or killed

**Exclude** (ignore these):
- `PASSED`, `XFAIL`, `SKIPPED`

```bash
grep -n "FAILED tests/" "<log_file>"
grep -n "CRASHED with signal" "<log_file>"
```

**Extract the full pytest test ID** from each `FAILED` line. The line looks like:
```
2024-01-01T00:00:00.000Z FAILED tests/runner/test_models.py::test_all_models_torch[yolov9/pytorch-S-single_device-inference] - tests/runner/...
```
Extract the portion starting at `tests/` and ending at the closing `]` (inclusive). This is the `test_id`:
```
tests/runner/test_models.py::test_all_models_torch[yolov9/pytorch-S-single_device-inference]
```

This format works for all test types:
- `tests/runner/test_models.py::test_all_models_torch[yolov9/pytorch-S-single_device-inference]`
- `tests/jax/single_chip/ops/test_slice.py::test_slice[3-97-1]`
- `tests/torch/graphs/test_attention.py::test_gemma_attention[2_27B_IT-1024-single_device]`

Deduplicate by **(test_id, machine_name)** pair — not by test_id alone. If the same test fails on two different machines, include a separate entry for each machine. If the same (test_id, machine_name) pair appears in multiple log files, keep only one — prefer the entry with the most informative error message.

#### Step 2 — Extract the real error message for each failed test

The pytest log has two distinct sections:

**A) Failure detail block** (earlier in the log) — contains the actual `AssertionError`:
```
_______ test_all_models_torch[yolov9/pytorch-S-single_device-inference] _______
... traceback ...
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9487. Required: pcc=0.98.
```

**B) Short test summary** (at the end) — only contains the call chain, NOT the error:
```
FAILED tests/runner/test_models.py::test_all_models_torch[yolov9/...] - tests/runner/test_models.py:294: in test_all_models_torch
```

**Always extract the error from section A**, not section B.

To match the `_ ` separator (GHA logs have only 1 underscore before the test name), use only the portion of `test_id` **after** `::` (the function name and params, without the file path prefix). Derive it as `func_and_params = test_id.split("::")[-1]`, e.g.:
- `tests/runner/test_models.py::test_all_models_torch[yolov9/...]` → `test_all_models_torch[yolov9/...]`
- `tests/jax/single_chip/ops/test_slice.py::test_slice[3-97-1]` → `test_slice[3-97-1]`

```bash
grep -F -A 100 "_ <func_and_params>" <log_file> | grep "Z E   "
```

Strip the leading timestamp and `E   ` prefix. If no `E   AssertionError` line is found, fall back to the `FAILED` summary line.

---

### Phase 5 — Process timed_out jobs (limited log parsing)

Apply this phase to all jobs in the **timed_out** bucket.

For each timed_out job, the log file is `$REPO_ROOT/bisection/logs/run_<run_id>/job_<job_id>.txt`. Then extract exactly three things:

**1. Collected test count** — look for the `collect Tests` section in the log, which lists lines like:
```
tests/runner/test_models.py::test_all_models_torch[efficientdet/pytorch-D7-single_device-inference]
tests/runner/test_models.py::test_all_models_torch[gpt2/pytorch-Default-single_device-inference]
```
Count how many such lines appear in the collect section:
```bash
grep -c "test_all_models_torch\[" "<log_file>"
```
Note: This counts ALL occurrences. To isolate only the collection section (before tests start running), look for the block between `collected` and the first `PASSED`/`FAILED`/`RUNNING` line. Use the total count from the collection header line if present (e.g. `collected 47 items`), otherwise count the `test_all_models_torch[` lines in the collection block.

**2. Last two tests that were running** — look for test execution lines in the running section. These appear as lines with `RUNNING` or as pytest's live output showing the test node ID being executed:
```bash
grep -n "RUNNING\|tests/runner/test_models\.py::test_all_models_torch\[" "<log_file>" | tail -3
```
The last match is the last test that was executing. The second-to-last is the test before it. Extract only the test_id (the part inside `[...]`).

---

### Phase 6 — Build output JSON and save

Save to `$REPO_ROOT/bisection/run_<run_id>_failures.json`:

```json
{
  "run_id": <run_id>,
  "run_date": "<run_date>",
  "sha": "<run_sha>",
  "workflow_id": <workflow_id>,
  "workflow_name": "<workflow_name>",
  "github_repo": "<github_repo>",
  "failed_tests": [
    {
      "test_id": "tests/runner/test_models.py::test_all_models_torch[yolov9/pytorch-S-single_device-inference]",
      "status": "FAILED",
      "raw_error": "AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9487420481169414. Required: pcc=0.98.",
      "machine_type": "n150",
      "machine_name": "forge-n150-12",
      "job_id": 12345678,
      "job_name": "test forge models passing / test n150",
      "job_url": "https://github.com/tenstorrent/tt-xla/actions/runs/.../jobs/...",
      "job_conclusion": "failure",
      "log_file": "job_12345678.txt"
    }
  ],
  "timed_out_jobs": [
    {
      "job_id": 12345681,
      "job_name": "test forge models nightly / test n150",
      "job_url": "https://github.com/tenstorrent/tt-xla/actions/runs/.../jobs/...",
      "tests_collected": 47,
      "last_test_executing": "densenet/pytorch-169-single_device-inference",
      "test_before_last": "resnet/pytorch-50-single_device-inference"
    }
  ],
  "not_finished_jobs": [
    {
      "job_id": 12345682,
      "job_name": "test forge models nightly / test n300",
      "job_url": "https://github.com/tenstorrent/tt-xla/actions/runs/.../jobs/..."
    }
  ]
}
```

Note: `failed_tests` contains tests from both **failed** and **cancelled** jobs. Each entry has `job_conclusion` to indicate which it came from. The same `test_id` may appear more than once if it failed on different machines — each (test_id, machine_name) pair gets its own entry.

Before saving, sort `failed_tests`:
1. Primary sort: `test_id` alphabetically (A→Z)
2. Secondary sort: `machine_type` in the fixed order `n150 → p150 → n300 → llmbox`

Use the Write tool to save this file.

---

### Phase 7 — Report to user

```
Failures collected for: <workflow_name>  run <run_id>  (<run_date>)

Failed / Cancelled Jobs (<N> tests total):
  - densenet/pytorch-169-single_device-inference  [n150 / forge-n150-12  (failure)]
      Error: AssertionError: PCC comparison failed. Calculated: pcc=0.9487. Required: pcc=0.98.
  - yolov9/pytorch-S-single_device-inference  [n150 / forge-n150-5  (cancelled)]
      Error: running the test CRASHED with signal 9
  ...

Timed Out Jobs (<N>):
  - <job_name>  <job_url>
      Tests collected:     47
      Last test executing: densenet/pytorch-169-single_device-inference
      Test before last:    resnet/pytorch-50-single_device-inference
  ...

Not Finished Jobs (<N>)  [still running — re-run /collect-failures later to capture results]:
  - <job_name>  <job_url>
  ...

Saved to: bisection/run_<run_id>_failures.json

To find regression boundaries for failed tests, run:
  /find-regression-boundaries bisection/run_<run_id>_failures.json
```

---

## Error handling

| Situation | Action |
|-----------|--------|
| `git rev-parse` fails (not inside a repo) | Stop and tell the user to run this from inside the repository |
| Jobs API returns empty | Report that no jobs were found for this run ID |
| No non-succeeded jobs found | Report that all jobs in this run succeeded |
| Per-job log download fails (403/404) | GitHub may not expose logs for this job. Print a warning, mark `log_file` as `"not available"`, continue with remaining jobs |
| Downloaded log file is empty | Warn the user, mark `log_file` as `"not available"`, continue |
