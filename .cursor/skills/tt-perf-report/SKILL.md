---
name: tt-perf-report
description: Run tt-perf-report on Tracy ops CSV slices, generate report CSV and summary PNG artifacts, and build colored HTML previews of the terminal-style performance report. Use when the user mentions tt-perf-report, performance report HTML, summary PNG, bar chart, stacked report, or report generation from a sliced ops CSV.
---

# TT Perf Report

Use this skill when the task is specifically about `tt-perf-report` outputs:
- running `tt-perf-report` on an existing slice CSV
- regenerating `tt_perf_report.csv`, `tt_perf_summary.csv`, or `tt_perf_summary.png`
- saving the terminal-style `Performance Report`
- creating or updating a previewable HTML rendering of that report

## Environment

Always run this workflow inside the TT-XLA Docker container, not from the host shell.

Use:

```bash
docker exec --user <CONTAINER_USER> <CONTAINER_NAME> /bin/bash -lc '
cd <REPO_PATH_IN_CONTAINER>
source venv/activate
<COMMAND>
'
```

Important:
- `venv/activate` in this repo is `pwd`-sensitive and uses `$(pwd)` for repo-relative paths
- always `cd <REPO_PATH_IN_CONTAINER>` before `source venv/activate`
- do not run `source /absolute/path/to/venv/activate` from `/` or any non-repo directory

After activating the repo venv, verify that `tt-perf-report` is available on `PATH`.
If it is missing, install `tt-perf-report` into that active venv before continuing.

## Command Defaults

Run `tt-perf-report` on the isolated or pre-sliced CSV, not on the full trace CSV.

Use `--no-advice` by default so reports contain the table plus stacked/summary outputs without the advice section.

Do not include `.csv` in the `--summary-file` argument. Use a basename only, because `tt-perf-report` adds its own suffixes.

Default command:

```bash
tt-perf-report <SLICE_CSV> \
  --no-advice \
  --summary-file <SUMMARY_BASENAME> \
  --csv <REPORT_CSV>
```

Example:

```bash
tt-perf-report odd_attention.csv \
  --no-advice \
  --summary-file odd_attention_summary \
  --csv odd_attention_report.csv
```

This produces:
- `tt_perf_report.csv` or `<name>_report.csv`
- `tt_perf_summary.csv` or `<name>_summary.csv`
- `tt_perf_summary.png` or `<name>_summary.png`

The generated `tt_perf_summary.png` is the default chart artifact.
If the user asks for:
- the image
- the chart
- the bar chart

interpret that as the generated summary PNG unless they explicitly ask for something else.

## Terminal Report Capture

If the user wants the terminal-style `Performance Report` text, prefer saving the real `tt-perf-report` terminal output instead of rewriting it by hand.

When available, preserve:
- the chaptered `Performance Report`
- the `Stacked report` section

If you need both CSV artifacts and terminal output, it is acceptable to run `tt-perf-report` twice:
- once with `--csv` and `--summary-file` to generate the files
- once with `--summary-file` and stdout redirection to capture the terminal report text

## HTML Preview

If the user wants a previewable HTML rendering of the terminal report:
- keep the terminal text content unchanged
- use the shared CSS from `tt-perf-report-preview.css` in this skill directory
- do not leave the report body as raw unclassified `<pre>` text; wrap rendered lines in semantic spans/classes so the preview actually shows colors
- classify report lines into the preview classes such as `title`, `rule-title`, `header`, `warning`, `bullet`, `cat-tm`, `cat-matmul`, `cat-ccl`, `cat-eltwise`, `cat-reduce`, `cat-special`, `stacked-row`, and `hi-us-*`
- render the report name above the terminal block as a dedicated title element using the `report-name` CSS class
- use a single flat dark gray page background, not a navy page with a separate gray card/container
- keep `SLOW` rows highlighted with a red background
- keep TM-style ops muted gray
- use subtle category tinting for non-TM ops
- color `FastReduceNCDeviceOperation` with the reduce bucket, not the CCL bucket
- if the user wants `Device Time` emphasis, highlight whole rows with `Device Time > 50 us`

Use severity buckets for `Device Time` row highlighting:
- `50-100 us` -> gray left border and light gray tint
- `100-200 us` -> yellow left border and light yellow tint
- `200+ us` -> red left border and light red tint

Append the summary PNG below the text report when the user wants the chart embedded in the page.
Render the embedded summary image at a fixed width of `600px` with `max-width: 100%` so it stays consistent while still shrinking on narrower previews.

Keep a single canonical HTML report by default.
Do not create a duplicate preview copy in `generated/live_preview/` unless the user explicitly asks for it or the editor cannot preview the canonical path.

If there are multiple copies because the user explicitly requested them, regenerate them all from the same source or copy the canonical file so the styling stays identical.

Use `html_preview_example.html` in this skill directory as the visual reference example for preferred HTML preview formatting.

## Recommended Workflow

1. Verify the container, repo path, and `tt-perf-report` availability.
2. Run `tt-perf-report --no-advice` on the slice CSV with `--summary-file` and `--csv`.
3. If needed, rerun without `--csv` and capture stdout to save the terminal-style report text.
4. Treat the summary PNG as the default visualization artifact.
5. If requested, render the saved terminal report as a colored HTML preview using `tt-perf-report-preview.css`.

## References

- `tt-perf-report-preview.css`
- `html_preview_example.html`
- `.cursor/skills/layer-profiling/SKILL.md`
