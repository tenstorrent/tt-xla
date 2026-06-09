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

**Always run the perf comparison yourself** — the CI `Check perf regression` step is unreliable and may fail silently or use stale baselines. The comparison script is the source of truth for regressions and improvements in this report.

**Always compare against the immediately preceding nightly from GitHub** — never use local report files as a substitute. Nightlies run daily and a local report may be days old, making the diff misleading.

Find the nightly run that ran just before the current one:
```bash
gh api "repos/tenstorrent/tt-xla/actions/workflows/184422752/runs?per_page=5&status=completed" \
  --jq '.workflow_runs[] | {id, created_at, head_sha}'
```
Pick the run whose `created_at` is the most recent date **earlier** than the current run's `created_at`. Then run the comparison script with a 5% threshold so both summary-relevant directions are captured:
```bash
python <skill-path>/scripts/compare_nightlies.py <CURRENT_RUN_ID> <PREVIOUS_RUN_ID> \
  --output-dir /tmp/perf_compare_<RUN_ID> --threshold 5
```

The script outputs `comparison.json` with per-model diffs categorized as `regression`, `improvement`, `stable`, `new`, or `missing`. Use `diff_percent` to apply the report's asymmetric thresholds:
- **Summary regressions**: `diff_percent <= -10`
- **Summary improvements**: `diff_percent >= +5`

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
   - `AssertionError:` — Python assertion failures. Extract the full assertion message.
   - `error:` lines in MLIR output — compilation failures.
   - `FAILED` in pytest output — test failure. Look for the short test summary.
   - `RuntimeError:` — Python runtime errors.
   - `Performance regression > 5% detected!` — this is a perf signal, NOT a job failure to report under "Failures".
   - Generic: last `##[error]` line before job end.

3. **Classify with a short, universal label.** The reader scans these — they need to recognize the failure mode at a glance, not read prose. Don't editorialize ("marginally", "extremely", "very low"), don't speculate about meaning ("logits diverged from reference", "optimizer assigned L1-resident layout"), and don't restate the obvious. Just name the failure mode and, if useful, append the bare numbers from the log. Examples:

   | Log signal | Label |
   |------------|-------|
   | `AssertionError: First decode PCC failed. PCC=X, Required=Y` | `First decode PCC failed. PCC=X, Required=Y` |
   | `AssertionError: <other> PCC failed. PCC=X, Required=Y` | `<other> PCC failed. PCC=X, Required=Y` |
   | `TT_FATAL: Out of Memory: ... L1 buffer ...` | `L1 OOM: <size> B across <N> banks` |
   | `TT_THROW: ... circular buffers ... clash with L1 buffers ...` | `CB/L1 clash: <core range>` |
   | `error:` from MLIR output | `Compilation error: <op or pass name>` |
   | Other `RuntimeError` / `TT_FATAL` | `Runtime error: <one-clause excerpt>` |

   **PCC labels**: use the AssertionError message verbatim (after `AssertionError: `) as the label — do NOT strip the step qualifier ("First decode", "Prefill", etc.). The step where PCC failed is essential diagnostic context.

   The same label is used in the Summary bullet AND in the Failed Jobs subsection's `**Error**` line — they must match.

4. **For device timeout / hang failures**, the hanging op is critical context. After identifying the `TIMEOUT: device timeout in fetch queue wait` error, scan the job log for the op being executed at the time of hang. Look for lines like:
   - `BinaryNg`, `UnaryWithParam`, `MatmulMultiCore`, or other kernel/op names appearing shortly before the timeout
   - tt-triage output lines that list the op name and its arguments
   - Patterns like `Running op: <op_name>` or `Dispatching kernel: <name>`

   Include the hanging op in the **Context** subsection under a "Hang details" note, e.g.: `Hang details: timed out during BinaryNg op (from job log)`. This makes it immediately actionable for the tt-metal/tt-mlir owners debugging the hang.

5. **Use Glean to find prior context — but stay factual.** Search for the error and surface what's already documented:
   ```
   mcp__glean_default__search: "<core error message> tt-xla" or "<op name> failure"
   mcp__glean_default__chat: "What is known about this error in tt-xla/tt-mlir: <error message>"
   ```
   Include only what the linked sources say: issue numbers, filer, file date, related PRs, Slack threads, owner. Do **not** draw conclusions Glean didn't state — no "no fix has merged yet", no "PCC=0.41 indicates the logits are nearly random", no characterizations of severity. If the issue itself says a fix is in flight, you can quote that; if it doesn't, leave it out. If Glean returns nothing, note that explicitly.

### 6. Analyze successful perf benchmark jobs

For each successful perf benchmark job, download and parse the perf report artifact using the bundled script:

```bash
python <skill-path>/scripts/fetch_perf_reports.py <RUN_ID> --output-dir /tmp/perf_reports_<RUN_ID>
```

This produces `summary.json` with extracted metrics for all jobs. Key metrics per model:
- **Samples/sec** = `samples_per_sec` measurement if present, else `total_samples / total_time`
- **TTFT (ms)** = `ttft` value (LLM models only; absent for vision/encoder models)
- **Device FW duration (s)** = `device_fw_duration` (only when device perf ran)
- **Device type**: from `device_info.device_type`

### 7. Generate the report

Write to `nightly-reports/nightly_<RUN_ID>_<YYYY-MM-DD>.md`. Create the directory if needed.

## Report Template

The summary is the most important section — it's what people read first and sometimes the only thing they read. Keep it short, factual, and scannable. Three sections only: Failures, Perf Regressions, Perf Improvements. No prose preambles, no "Note:" callouts about commits being identical or measurement variance — the dependency chain at the bottom already shows the SHAs, the reader can compare.

```markdown
# CI Benchmark Report — Run <RUN_ID>

## Summary

**Perf benchmarks**: X/Y passed

### Failures
- [model_a (n150)](#model_a-n150) — <short universal label>
- [model_b (n300-llmbox)](#model_b-n300-llmbox) — <short universal label>

[If none, write "None".]
[Anchor format: model_display_name-device_type, e.g. #llama_3_2_1b-n150]

### Perf Regressions
Models with `diff_percent <= -10` in `comparison.json` (Samples/sec dropped >10% vs previous nightly).
- model_c (n150): -15.3%

[If none, write "None". Always derive from the comparison script — never trust the CI "Check perf regression" step.]

### Perf Improvements
Models with `diff_percent >= +5` in `comparison.json` (Samples/sec improved >5% vs previous nightly).
- model_d (n150): +12.1%

[If none, write "None".]
```

The Run Info and Dependency Chain sections go at the END of the report (see "Run Info & Dependency Chain" below).

## Failed Jobs

For each failed model, create a subsection with an anchor matching the summary link:

```markdown
### <a id="model_a-n150"></a>model_a (n150)
- **Job**: [link to job](<job_url>)
- **Error**: <short universal label — same wording as the Summary bullet>
  ```
  <relevant error snippet, 3-5 lines max>
  ```
- **Context**: <factual references only — issue numbers, filer + file date, related PRs, owners, Slack threads. State whether it's a known issue and link it. Do not write "no fix has merged yet", do not interpret PCC values, do not speculate about cause unless the linked issue states it explicitly.>
```

**All issue and PR references must be hyperlinked** — never write bare `tt-xla#NNNN` text. Use full markdown links:
- tt-xla issues/PRs: `[tt-xla#NNNN](https://github.com/tenstorrent/tt-xla/issues/NNNN)` (use `/pull/NNNN` for PRs)
- tt-mlir issues/PRs: `[tt-mlir#NNNN](https://github.com/tenstorrent/tt-mlir/issues/NNNN)`
- tt-metal issues/PRs: `[tt-metal#NNNN](https://github.com/tenstorrent/tt-metal/issues/NNNN)`

**Good Context examples** (terse, factual, link-driven):
- `Tracked in [tt-xla#4394](https://github.com/tenstorrent/tt-xla/issues/4394), [tt-mlir#8044](https://github.com/tenstorrent/tt-mlir/issues/8044) (filed by pdeviTT, 2026-04-21, open). Owner: Vladan Kovacevic, pdeviTT.`
- `Known issue from [tt-xla#4318](https://github.com/tenstorrent/tt-xla/pull/4318) (new decode PCC check, merged 2026-04-24). Active investigation in #tt-forge-models-code-changes Slack thread "Qwen PCC Drop Debugging". Owner: pdeviTT.`

**Avoid**:
- "No fix has merged yet" (unstated assumption — leave it out, the issue link tells the reader)
- "PCC=0.41 indicates the logits are nearly random" (interpretation Glean didn't state)
- "Root cause: ..." unless the linked issue explicitly identifies it

## Performance Results

Group by device type. Tables show Samples/sec and the diff vs previous nightly.

**TTFT is intentionally omitted for now** — it will be reintroduced once we're tracking it properly. Don't add a TTFT column, don't include TTFT diffs, and don't add explanatory notes about TTFT/measurement variance.

**Filter: Only include models with ≥5% change in Samples/sec vs the previous nightly.** Models with smaller changes are stable and should be omitted from the tables. Always include models marked `new` (not present in previous nightly) and `missing` rows where a diff can't be computed. At the end of each device section, add: `_N stable models omitted (<5% change in Samples/sec)._`

```markdown
### n150 Models

| Model | Samples/sec | Diff | Device FW (s) | Job |
|-------|------------|------|---------------|-----|
| model_a | 36.5 | +15.3% | 1.89 | [link](...) |
| model_b | 12.3 | -5.2% | — | [link](...) |
| model_c | 27.9 | **-23.5%** | 1.45 | [link](...) |

[Bold diffs ≤ -10% — these are the regressions surfaced in the Summary]
[Use "—" for metrics not present (Device FW only when device perf step ran successfully)]
[Use "new" in Diff column for models not in the previous nightly]
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
  --output-dir /tmp/perf_compare_<RUN_ID> --threshold 5
```
Output: `comparison.json` with per-model diffs. Convention: positive % = faster (improvement), negative % = slower (regression). Use `--threshold 5` so the report can apply asymmetric thresholds (regressions ≤ -10%, improvements ≥ +5%) at filter time.

## Notes

- The `gh` CLI must be authenticated with access to `tenstorrent/tt-xla` (and `tenstorrent/tt-mlir` for commit chain resolution).
- Artifact downloads require appropriate permissions. If an artifact has expired (default 90 days), note it in the report.
- For very large runs (nightly has 50+ jobs), use the bundled scripts to batch-process artifacts rather than downloading one by one.
- Device perf measurements (`device_fw_duration`) are only present when the workflow ran with `skip-device-perf: false` AND the device perf step succeeded. Their absence is not an error.
- When Glean returns no results for a failure, note that explicitly in the report — a novel error with no prior context is itself important.
- The CI `Check perf regression` job is unreliable; always run `compare_nightlies.py` to derive regressions and improvements yourself. Summary thresholds are asymmetric: regressions surface at `diff_percent <= -10`, improvements at `diff_percent >= +5`. Detail tables show all `|diff_percent| >= 5`.
