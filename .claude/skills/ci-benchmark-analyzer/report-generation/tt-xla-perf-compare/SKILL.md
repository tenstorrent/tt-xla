---
name: tt-xla-perf-compare
description: Compare perf-benchmark "Sample per second" metrics across two tenstorrent/tt-xla GitHub Actions nightly pipeline runs. Use when the user provides two GitHub Actions pipeline URLs (yesterday and today) and asks to compare performance, check regressions, improvements, or failures in the perf-benchmark jobs.
---

# tt-xla Perf Benchmark Comparison

Compares "Sample per second" values from the `perf-benchmark / run-perf-benchmarks` jobs across two nightly pipeline runs.

## Inputs

- **Yesterday URL**: e.g. `https://github.com/tenstorrent/tt-xla/actions/runs/24485208389`
- **Today URL**: e.g. `https://github.com/tenstorrent/tt-xla/actions/runs/24540958167`

Both full URLs or bare run IDs are accepted.

---

## Workflow

**Execute all steps immediately without asking for permission or confirmation at any point. Never pause to ask, just do it.**

### Step 1 — Check for fresh results

Check whether `/tmp/perf_comparison.json` exists and contains matching run IDs:

```bash
python3 -c "
import json, sys, os
if not os.path.exists('/tmp/perf_comparison.json'):
    print('missing')
else:
    d = json.load(open('/tmp/perf_comparison.json'))
    print(d.get('yesterday_id',''), d.get('today_id',''))
"
```

- If the run IDs **match** what the user provided → skip to Step 3.
- If **missing or mismatched** → go to Step 2.

### Step 2 — Ask user to run the script

Tell the user exactly:

> "Please run this in your terminal, then let me know when it's done:"
> ```
> python3 ~/.cursor/skills/tt-xla-perf-compare/scripts/compare_perf.py <yesterday_url> <today_url>
> ```
> "It takes about 60–90 seconds."

Wait for the user to confirm it's done, then continue to Step 3.

### Step 3 — Read results and produce report

Read `/tmp/perf_comparison.json` and do both:
1. **Print the text report** in the format below
2. **Update the canvas** at the canvas path

---

## Key Details

- **Repo**: always `tenstorrent/tt-xla`
- **Script**: `~/.cursor/skills/tt-xla-perf-compare/scripts/compare_perf.py`
- **JSON output**: `/tmp/perf_comparison.json`
- **Canvas path**: `~/.cursor/projects/<workspace>/canvases/perf-benchmark-comparison.canvas.tsx`
- **Log pattern (sps)**: `Sample per second: <float>` (last occurrence per job wins)
- **Log pattern (regression)**: `Performance regression > 5% detected! Performance dropped by X%`
- **Improvement threshold (text callout)**: `pct >= 10` (change% ≥ 10%), sorted descending by delta
- **Regression threshold**: only when the above log line is present — do NOT use delta
- **Hardware groups**: `n150-perf`, `n300-llmbox`, `galaxy-wh-6u`

---

## Output Format

### Section order (omit sections with no entries)

```
0. Commit Details        (always shown)
1. Failed
2. Perf Regression
3. Failed -> Passed
4. Perf Improvement
5. In Progress / Queued  (if any)
6. Failures In Detail    (tabular, always shown when there are failures)
```

### Headings: bold + underlined (no section numbers)

### Commit Details (always first)

```
Commit Details:
  Nightly Pipeline : https://github.com/tenstorrent/tt-xla/actions/runs/24540958167
  tt-xla commit    : eef6ae1cd3b97d9be39a113bae1cb653852219f9
  tt-mlir commit   : a53ca15ef0d7e17ca7ecbac52698302539bfaa71
  tt-metal commit  : 3fa4d753550dba1d4aacc9af45b111ae540f63fc
```

- `Nightly Pipeline` = today's run HTML URL (`run_info.today.url` from JSON)
- `tt-xla commit`  = `run_info.today.sha`
- `tt-mlir commit` = `run_info.today.mlir_sha` — from `TT_MLIR_VERSION` in `tenstorrent/tt-xla @ third_party/CMakeLists.txt` at the xla commit
- `tt-metal commit` = `run_info.today.metal_sha` — from `TT_METAL_VERSION` in `tenstorrent/tt-mlir @ third_party/CMakeLists.txt` at the mlir commit

### Failed (inline list only — no error text here)

```
Failed:
  bert (n150-perf) [new]
  mistral_nemo_instruct_2407_tp (n300-llmbox) [new]
  test_gpt_oss_120b_tp_galaxy_batch_size_64 (galaxy-wh-6u) [old]
```

- `[new]` = passed yesterday, failed today — OR failed both days but different failed step
- `[old]` = failed both days with the same failed step
- **Do NOT include jobs that appear under Perf Regression here** — they are reported there only

### Perf Regression

```
Perf Regression:
  ministral_8b [Performance dropped by 45.14%]
```

Only when today's log contains `"Performance regression > 5% detected! Performance dropped by X%"`.

### Failed -> Passed

```
Failed -> Passed:
  test_gpt_oss_120b_tp_galaxy_batch_size_64
```

### Perf Improvement

```
Perf Improvement:
  mnist [delta: 84.68, change%: +0.5%]
  resnet [delta: 0.96, change%: +0.1%]
  qwen_3_4b_embedding [delta: 0.69, change%: +1.4%]
  qwen_2_5_1_5b_instruct [delta: 0.50, change%: +1.0%]
```

### In Progress / Queued

```
In Progress / Queued:
  some_model (n150-perf) [in_progress]
  another_model (n300-llmbox) [queued]
```

### Failures In Detail (tabular — 5 columns)

```
Failures In Detail:
  Job                                    Tag     Failure Reason (first error)              Step                    URL
  -------------------------------------- ------- ----------------------------------------- ----------------------- -------------------------------------------
  bert (n150-perf)                       [new]   E: Failed to fetch http://...             Check perf regression   https://github.com/tenstorrent/tt-xla/actions/runs/<run_id>/job/<job_id>
  mistral_nemo_instruct_2407_tp          [new]   E: Failed to fetch http://...             Check perf regression   https://...
```

- **Failure Reason** = `t_first_error` from JSON, extracted with priority: 1st `E:`, 2nd `Error:`, 3rd any line with `error` (case-insensitive)
- **Step** = `t_fail_step` from JSON (first step with `conclusion: "failure"`)
- **URL** = `https://github.com/tenstorrent/tt-xla/actions/runs/<today_id>/job/<t_job_id>`

---

## Canvas `JobRow` type

```typescript
type JobRow = {
  name: string; hw: string; ys: string; ts: string;
  y: number | null; t: number | null;
  d: number | null; p: number | null;
  r: string; pd: string; tag: string; fr: string; fe: string; jid: string;
};
```

Field mapping from JSON:
- `name` → job name (strip `"perf "` prefix and ` (hardware)` suffix)
- `hw` → `hardware`
- `ys` / `ts` → `y_status` / `t_status`
- `y` / `t` → `y_sps` / `t_sps`
- `d` / `p` → `delta` / `pct`
- `r` → `result`
- `pd` → `t_perf_drop`
- `tag` → `"new"` or `"old"` for failures, `""` otherwise
- `fr` → `t_fail_step`
- `fe` → `t_first_error`
- `jid` → `t_job_id`

## Canvas constants (update on every new run)

```typescript
const YESTERDAY_RUN      = "<yesterday pipeline URL>";
const TODAY_RUN          = "<today pipeline URL>";
const TODAY_COMMIT_XLA   = "<run_info.today.sha>";
const TODAY_COMMIT_MLIR  = "<run_info.today.mlir_sha>";
const TODAY_COMMIT_METAL = "<run_info.today.metal_sha>";
```

## Canvas sections (same order as text report)

1. **Failed** — Job | Hardware | Tag  (warning/error row tones)
2. **Perf Regression** — Job | Drop  (error tone)
3. **Failed → Passed** — Job | Hardware | Today sps  (success tone)
4. **Perf Improvement** — Job | Hardware | Delta | Change%  (success tone)
5. **In Progress / Queued** — Job | Hardware | Status  (warning tone)
6. **Failures In Detail** — Job | Tag | Failure Reason (first error) | Step | Link  (warning/error tones)
7. **Full Comparison** — Job | Hardware | Yesterday | Today | Delta | Change% | Result
8. **By Hardware Platform** — stats per hw group

All section headings: `<H2><span style={{ textDecoration: "underline", fontWeight: 700 }}>Title</span></H2>`

Canvas imports: `import { Divider, Grid, H1, H2, H3, Row, Stack, Stat, Table, Text, useHostTheme } from 'cursor/canvas';`
