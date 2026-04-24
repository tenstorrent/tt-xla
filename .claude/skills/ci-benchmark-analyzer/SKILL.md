---
name: ci-benchmark-analyzer
description: Analyze CI benchmark workflow runs from GitHub Actions for the tt-xla project. Produces a markdown report covering failed jobs (with root-cause error extraction via logs and Glean), successful model performance metrics (samples/sec, TTFT, device perf), perf regressions/improvements vs previous nightly, and the full dependency commit chain (tt-xla, tt-mlir, tt-metal). Use this skill whenever the user wants to analyze a CI run, review nightly benchmark results, investigate CI failures, check benchmark performance from a workflow run, or asks about "latest nightly" results. Also trigger when the user pastes a GitHub Actions run URL or mentions a run ID in the context of performance analysis, or asks about perf regressions.
---

# CI Benchmark Analyzer

Analyzes GitHub Actions benchmark workflow runs for the `tenstorrent/tt-xla` repository. Produces a structured markdown report with an actionable summary, failure diagnosis, performance comparison against the previous nightly, and detailed per-model metrics.

## Input Handling

The user specifies a run in one of these ways:

| Input | How to resolve |
|-------|---------------|
| GitHub Actions URL | Extract run ID from `actions/runs/<ID>` in the URL |
| Numeric run ID | Use directly |
| `latest nightly` | Query the latest completed run of the "On nightly" workflow (ID `184422752`) |
| `latest benchmark` | Query the latest completed run of "Performance Benchmark" workflow (ID `184422748`) |
| `latest experimental` | Query the latest completed run of "On nightly Experimental" workflow (ID `187864533`) |

To resolve "latest" variants:
```bash
gh api "repos/tenstorrent/tt-xla/actions/workflows/<WORKFLOW_ID>/runs?per_page=1&status=completed" --jq '.workflow_runs[0].id'
```

If the run is still in progress, note that in the report header and analyze only the completed jobs.

## Report Storage

Save reports to `nightly-reports/` directory (relative to cwd) so that previous results accumulate and are accessible for future comparison. Naming convention:

```
nightly-reports/
  nightly_<RUN_ID>_<YYYY-MM-DD>.md
```

Always produce a fresh report file named after the current run ID — each run gets its own new file. The filename already encodes the run date, so there's no need for an "updated on" or "last updated" line in the report body.

Before generating, check `nightly-reports/` for the most recent prior report and use it as comparison context (read-only). If no prior report exists, fetch the previous nightly run from GitHub (see step 4) to establish a baseline.

## Workflow

### 1. Gather run metadata

```bash
gh api repos/tenstorrent/tt-xla/actions/runs/<RUN_ID> \
  --jq '{id, name, head_sha, head_branch, event, status, conclusion, created_at, updated_at, html_url}'
```

### 2. Resolve the dependency commit chain

This is important context for anyone reading the report — it tells them exactly which versions of the three core repos were used.

**tt-xla commit**: `head_sha` from step 1.

**tt-mlir commit**: Read `third_party/CMakeLists.txt` at that commit and extract `TT_MLIR_VERSION`:
```bash
gh api repos/tenstorrent/tt-xla/contents/third_party/CMakeLists.txt?ref=<HEAD_SHA> \
  --jq '.content' | base64 -d | grep 'set(TT_MLIR_VERSION'
```
Parse the SHA from the `set(TT_MLIR_VERSION "...")` line.

**tt-metal commit**: Read tt-mlir's `third_party/CMakeLists.txt` at the tt-mlir commit:
```bash
gh api repos/tenstorrent/tt-mlir/contents/third_party/CMakeLists.txt?ref=<TT_MLIR_SHA> \
  --jq '.content' | base64 -d | grep 'set(TT_METAL_VERSION'
```

### 3. List all jobs and classify them

```bash
gh api "repos/tenstorrent/tt-xla/actions/runs/<RUN_ID>/jobs?per_page=100" --paginate
```

The response may span multiple pages. Collect all jobs.

**Focus exclusively on perf benchmark jobs** (name contains `perf-benchmark / run-perf-benchmarks / perf`). Ignore all other job categories (build, test, forge, release, notify, etc.) entirely — they are not relevant to this report.

Separate failed perf jobs into two categories:
- **Model failures**: the benchmark itself failed (compilation error, runtime crash, device error)
- **Perf regression failures**: the job failed because the `Check perf regression` step detected >5% regression — these are NOT errors, they are perf signals

You can distinguish them by checking the job logs: regression failures have `"Performance regression > 5% detected!"` in the log, while model failures have stack traces, `TT_FATAL`, `AssertionError`, etc.

### 4. Get previous nightly for comparison

**Always compare against the immediately preceding nightly from GitHub** — never use local report files as a substitute. Nightlies run daily and a local report may be days old, making the diff misleading.

Find the nightly run that ran just before the current one:
```bash
gh api "repos/tenstorrent/tt-xla/actions/workflows/184422752/runs?per_page=5&status=completed" \
  --jq '.workflow_runs[] | {id, created_at, head_sha}'
```
Pick the run whose `created_at` is the most recent date **earlier** than the current run's `created_at`. Then use the comparison script to compute diffs:
```bash
python <skill-path>/scripts/compare_nightlies.py <CURRENT_RUN_ID> <PREVIOUS_RUN_ID> \
  --output-dir /tmp/perf_compare_<RUN_ID> --threshold 10
```

The script outputs `comparison.json` with per-model diffs categorized as `regression`, `improvement`, `stable`, `new`, or `missing`.

Local reports in `nightly-reports/` are read-only context (e.g. to check if a failure is recurring) — they are **not** the baseline for perf diffs.

**Perf diff convention** (used consistently everywhere in the report):
- **Positive % = improvement** (model got faster, more samples/sec)
- **Negative % = regression** (model got slower, fewer samples/sec)

Example: "llama_3_2_1b: +15.3%" means it's 15.3% faster than the previous nightly.

### 5. Analyze failed perf benchmark jobs

For each failed perf benchmark job (model failures only, not regression check failures):

1. **Fetch the job log**:
```bash
gh api repos/tenstorrent/tt-xla/actions/jobs/<JOB_DATABASE_ID>/logs
```

2. **Extract the root error** by scanning the log for these patterns (in priority order):
   - `TT_FATAL` or `TT_THROW` — device/runtime fatal error. Extract the error message and walk up the stack trace for the nearest `ttnn::` or `tt::tt_metal::` reference.
   - `AssertionError:` or `AssertionError:` — Python assertion failures. Extract the full assertion message.
   - `error:` lines in MLIR output — compilation failures.
   - `FAILED` in pytest output — test failure. Look for the short test summary.
   - `RuntimeError:` — Python runtime errors.
   - `Performance regression > 5% detected!` — this is a perf regression, NOT an error. Classify it separately.
   - Generic: last `##[error]` line before job end.

3. **Use Glean extensively for deeper context**: Search Glean with the core error message to find related issues, discussions, or documentation. This is critical — it tells the reader whether this is a known issue, whether someone is already working on a fix, and what the broader impact might be.
   ```
   mcp__glean_default__search: "<core error message> tt-xla" or "<op name> failure"
   mcp__glean_default__chat: "What is known about this error in tt-xla/tt-mlir: <error message>"
   ```
   Include any relevant findings (linked issues, Slack threads, related PRs) in the failure section. If Glean returns nothing, note that explicitly — a novel error with no prior context is important information.

### 6. Analyze successful perf benchmark jobs

For each successful perf benchmark job, download and parse the perf report artifact using the bundled script:

```bash
python <skill-path>/scripts/fetch_perf_reports.py <RUN_ID> --output-dir /tmp/perf_reports_<RUN_ID>
```

This produces `summary.json` with extracted metrics for all jobs. Key metrics per model:
- **Samples/sec** = `total_samples / total_time`
- **TTFT (ms)** = `ttft` value (LLM models only; absent for vision/encoder models)
- **Device FW duration (s)** = `device_fw_duration` (only when device perf ran)
- **Device type**: from `device_info.device_type`

### 7. Generate the report

Write to `nightly-reports/nightly_<RUN_ID>_<YYYY-MM-DD>.md`. Create the directory if needed.

## Report Template

The summary is the most important section — it's what people read first and sometimes the only thing they read. Make it actionable.

```markdown
# CI Benchmark Report — Run <RUN_ID>

## Summary

**Perf benchmarks**: X/Y passed

### Failures
Models that failed to run (excluding perf regression check failures).
Each entry links to its detailed section below.
- [model_a (n150)](#model_a-n150) — <one-line error summary>
- [model_b (n300-llmbox)](#model_b-n300-llmbox) — <one-line error summary>

[If no failures, write "None".]
[Format anchors as: model_display_name-device_type, e.g. #llama_3_2_1b-n150]

Keep the summary focused on models that actually failed to run — perf regressions and improvements are covered in the Performance Results tables below.
```

The Run Info and Dependency Chain sections go at the END of the report (see "Run Info & Dependency Chain" below).

## Failed Jobs

For each failed model, create a subsection with an anchor matching the summary link:

```markdown
### <a id="model_a-n150"></a>model_a (n150)
- **Job**: [link to job](<job_url>)
- **Error**: <concise root cause>
  ```
  <relevant error snippet, 3-5 lines max>
  ```
- **Context**: <Glean findings — related issues, known problem, etc.>
```

## Performance Results

Group by device type. Include the perf diff column showing change vs previous nightly.

**Filter: Only include models with ≥5% change in Samples/sec OR ≥5% change in TTFT vs the previous nightly.** Models with smaller changes in both metrics are considered stable and should be omitted from the tables. Always include models marked "new" (not present in previous nightly) and models with `missing`/"—" previous values where a diff can't be computed. At the end of each device section, add a line like: `_N stable models omitted (<5% change in both Samples/sec and TTFT)._`

```markdown
### n150 Models

| Model | Samples/sec | Diff | TTFT (ms) | TTFT Diff | Device FW (s) | Job |
|-------|------------|------|-----------|-----------|---------------|-----|
| model_a | 36.5 | +15.3% | 147.8 | +8.2% | 1.89 | [link](...) |
| model_b | 12.3 | -5.2% | 780.6 | -12.1% | — | [link](...) |
| model_c | 27.9 | **-23.5%** | 200.1 | +3.0% | 1.45 | [link](...) |

[Bold diffs that exceed 10% in either direction]
[Use "—" for metrics not present (vision/encoder models have no TTFT)]
[Use "new" in Diff columns for models not in the previous nightly]
[TTFT Diff: positive = TTFT got lower/faster (improvement), negative = TTFT got higher/slower (regression)]
```

Adjust the template based on what's actually in the run — if there are no failures, write "None" under Failures.

## Run Info & Dependency Chain (report footer)

Place the run metadata and dependency chain at the END of the report, after Performance Results. Use bulleted form, omit the event/Status fields, and show both nightly URLs as full URLs (no `[report](...)` link):

```markdown
## Run Info

- **Workflow**: <workflow name>
- **Branch**: <head_branch>
- **URL**: <html_url>
- **Previous nightly**: <prev_html_url>

## Dependency Chain

| Repo | Commit | Link |
|------|--------|------|
| tt-xla | `<short_sha>` | [full SHA](https://github.com/tenstorrent/tt-xla/commit/<sha>) |
| tt-mlir | `<short_sha>` | [full SHA](https://github.com/tenstorrent/tt-mlir/commit/<sha>) |
| tt-metal | `<short_sha>` | [full SHA](https://github.com/tenstorrent/tt-metal/commit/<sha>) |
```

The filename already encodes the run date — no need for an "updated on" line in the body.

## Scripts

Two helper scripts are bundled:

### `scripts/fetch_perf_reports.py`
Batch-downloads all perf report artifacts for a run and extracts metrics.
```bash
python <skill-path>/scripts/fetch_perf_reports.py <RUN_ID> --output-dir /tmp/perf_reports_<RUN_ID>
```
Output: `summary.json` with per-model metrics.

### `scripts/compare_nightlies.py`
Compares perf between two runs and categorizes models as regression/improvement/stable.
```bash
python <skill-path>/scripts/compare_nightlies.py <CURRENT_RUN_ID> <PREVIOUS_RUN_ID> \
  --output-dir /tmp/perf_compare_<RUN_ID> --threshold 10
```
Output: `comparison.json` with per-model diffs. Convention: positive % = faster (improvement), negative % = slower (regression).

## Notes

- The `gh` CLI must be authenticated with access to `tenstorrent/tt-xla` (and `tenstorrent/tt-mlir` for commit chain resolution).
- Artifact downloads require appropriate permissions. If an artifact has expired (default 90 days), note it in the report.
- For very large runs (nightly has 50+ jobs), use the bundled scripts to batch-process artifacts rather than downloading one by one.
- Device perf measurements (`device_fw_duration`) are only present when the workflow ran with `skip-device-perf: false` AND the device perf step succeeded. Their absence is not an error.
- TTFT is only meaningful for LLM models (test_llms.py tests). Vision and encoder models won't have it.
- When Glean returns no results for a failure, note that explicitly in the report — a novel error with no prior context is itself important.
- Perf regression check failures in CI use a 5% threshold, but for the report summary we use 10% to highlight only significant changes. Models between 5-10% regression may still appear in the detailed tables.
