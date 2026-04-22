---
name: compare-nightly
description: Compare two nightly GitHub Actions runs and classify failures as new, persisting, or fixed
disable-model-invocation: true
allowed-tools: Read, Read(/tmp/**), Glob, Grep, Bash(git clone *), Bash(gh run view *), Bash(gh run list *), Bash(gh run download *), Bash(gh pr view *), Bash(tee *), Bash(gh api *), Bash(gh api * > /tmp/**), Bash(wc -l /tmp/**), Bash(jq *), Bash(mkdir -p /tmp/**), Bash(rm -rf /tmp/**), Bash(for *), Bash(python3 *)
context: fork
argument-hint: baseline-run-id comparison-run-id
model: opus
---

Compare two nightly GitHub Actions runs and classify every test failure as:
- **New**: failing in the comparison run but not in the baseline run
- **Persisting**: failing in both runs
- **Fixed**: failing in the baseline run but not in the comparison run

The baseline run-id is $0 (typically the older/yesterday run).
The comparison run-id is $1 (typically the newer/today run).
The GitHub repo is github.com/tenstorrent/tt-xla.

## Step 1 — Collect failed tests for each run

Repeat this process for both runs independently.

For each run:
1. Fetch all jobs: `gh run view {run-id} --repo tenstorrent/tt-xla --json jobs --jq '.jobs[] | {id: .databaseId, name: .name, conclusion: .conclusion}'`
   If that fails (HTTP 502 for older runs), paginate with:
   `gh api "repos/tenstorrent/tt-xla/actions/runs/{run-id}/jobs?per_page=30&page={N}"`
2. Discard jobs with conclusion: success, skipped, cancelled, or null (in progress).
   Also discard jobs whose names contain "fail-notify", "Collect data", "Failure Inspector",
   "Nightly Update Release Notes", "generate", "filter", "check-if-docker".
3. For each remaining failed job, fetch its raw log:
   `gh api "repos/tenstorrent/tt-xla/actions/jobs/{job-id}/logs"`
4. In the log, find the pytest short test summary section by searching for
   "short test summary info". Extract every line that starts with "FAILED" or "ERROR"
   from that section. These lines identify the specific test that failed and its
   first assertion/error message.
5. If the log has no pytest summary (e.g. a benchmark or demo job), extract the
   test name and error from the log using these signals:
   - Lines containing "FAILED", "CRASHED with signal", "TT_FATAL", "TIMEOUT",
     "hang detected", "Out of Memory", "Bad StatusOr", "AssertionError",
     "PCC comparison failed".
   - Lines containing "Killed", "Fatal Python error", "Segmentation fault",
     "Segfault", "SIGSEGV", "SIGABRT", "Aborted", "Crashed".
   - Search backward from the error line to find the test function name
     (lines matching "test_" or "::test_").
6. For each extracted failure record:
   - test_id: the pytest node id (e.g. "tests/torch/graphs/test_attention.py::test_glm_4_attention_decode[4.5-single_device]")
     or a model name (e.g. "test_all_models_torch[glm/causal_lm/pytorch-4.5_Air-single_device-training]")
   - root_cause: short phrase (e.g. "TT_FATAL MUL_BCAST_GRANULARITY", "OOM signal 9", "PCC nan",
     "device timeout", "Killed", "Fatal Python error", "Segmentation fault", "SIGABRT")
   - arch: hardware runner from the job name (e.g. n150, p150, n300-llmbox, galaxy-wh-6u, n150-perf)
   - job_url: "https://github.com/tenstorrent/tt-xla/actions/runs/{run-id}/job/{job-id}"

Build two lists: BASELINE_FAILURES (from run $0) and COMPARISON_FAILURES (from run $1).

## Step 2 — Collect commits merged between the two runs

Fetch the head SHA that each run was triggered on:
`gh api "repos/tenstorrent/tt-xla/actions/runs/{run-id}" --jq '.head_sha'`

Do this for both $0 (baseline SHA) and $1 (comparison SHA).

Then fetch the list of commits on the main branch between those two SHAs (exclusive
of the baseline SHA, inclusive of the comparison SHA):
`gh api "repos/tenstorrent/tt-xla/compare/{baseline-sha}...{comparison-sha}" --jq '.commits[] | {sha: .sha[0:8], author: .commit.author.name, message: .commit.message | split("\n")[0]}'`

If the comparison SHA is newer than the baseline SHA (the normal case) this returns
commits that landed between the two nightlies. If the API returns an empty list or
an error, note it and continue.

Store the result for the report in Step 5.

## Step 4 — Match failures across runs

A failure in COMPARISON_FAILURES matches a failure in BASELINE_FAILURES when:
- The test_id strings are identical, OR
- The test_id strings differ only in the hardware arch suffix and refer to the same
  logical test (treat the arch as a separate dimension, not part of the test identity).

Use Python 3 (via `python3 -c "..."`) to compute the three sets after you have
collected both lists. Write a small script that reads JSON from stdin or from
two temporary files in /tmp/ and outputs the three categorized groups. Clean up
any temporary files in /tmp/ when done.

## Step 5 — Produce the report

Output the following Markdown report. Do not emit a section if it is empty.

```
# Commits Merged Between Runs

| SHA | Author | Message |
|-----|--------|---------|
| {sha} | {author} | {message} |

_(baseline: {baseline-sha}, comparison: {comparison-sha})_

# New Failures (in $1, not in $0)

## {root-cause}
- {test_id} ({arch}) -> [job-link]({job_url})

# Persisting Failures (in both $0 and $1)

## {root-cause}
- {test_id} ({arch}) -> [job-link]({job_url in $1}) [also in baseline]({job_url in $0})

# Fixed Since Baseline (in $0, not in $1)

## {root-cause}
- {test_id} ({arch}) -> [baseline job]({job_url in $0})
```

Group failures within each section by root cause (same heading = same root cause).
If the same test fails on multiple hardware arches with the same root cause, list it
once with all arches as a comma-separated list in the ({arch}) field.

## Constraints

- Never run any `gh` command that modifies repository state (no issue/PR creation,
  branch changes, etc.). Read-only calls only.
- Never execute multiple commands separated by a semicolon.
- Always use for loops in bash when iterating over multiple job IDs.
- If log fetching fails for a job, note it and continue — do not abort.
- If a run has more than 100 jobs, paginate through all pages before concluding
  which jobs failed.
