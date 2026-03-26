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

Default command when the user wants the terminal-style report and summary artifacts:

```bash
tt-perf-report <SLICE_CSV> \
  --no-advice \
  --summary-file <SUMMARY_BASENAME>
```

Example:

```bash
tt-perf-report odd_attention.csv \
  --no-advice \
  --summary-file odd_attention_summary
```

This produces:
- `tt_perf_summary.csv` or `<name>_summary.csv`
- `tt_perf_summary.png` or `<name>_summary.png`
- the terminal-style report on stdout

If the user also wants the per-op CSV artifact, run a second command:

```bash
tt-perf-report <SLICE_CSV> \
  --no-advice \
  --summary-file <SUMMARY_BASENAME> \
  --csv <REPORT_CSV>
```

This additional run produces:
- `tt_perf_report.csv` or `<name>_report.csv`

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

If you need both CSV artifacts and terminal output, run `tt-perf-report` twice:
- once with `--csv` and `--summary-file` to generate the files
- once with `--summary-file` and stdout redirection to capture the terminal report text
- use `--no-advice` on both runs unless the user explicitly wants the advice section
- do not do the second run if the user only needs terminal output plus summary CSV/PNG; a single `--summary-file` run already produces those

## HTML Preview

If the user wants a previewable HTML rendering of the terminal report:
- keep the terminal text content unchanged
- use the shared CSS from `tt-perf-report-preview.css` in this skill directory
- do not leave the report body as raw unclassified `<pre>` text; wrap rendered lines in semantic spans/classes so the preview actually shows colors
- by default, keep only the `Performance Report` section in the HTML preview
- strip warning lines and omit later sections such as `Stacked report` unless the user explicitly asks to keep them
- classify report lines into the preview classes such as `title`, `rule-title`, `header`, `cat-tm`, `cat-matmul`, `cat-ccl`, `cat-eltwise`, `cat-reduce`, `cat-special`, and `hi-us-*`
- render the report name above the terminal block as a dedicated title element using the `report-name` CSS class
- use a dark olive/forest page background with deeper mossy panels, code-like monospace headings, and soft pink accents in the page chrome
- keep `SLOW` rows highlighted with a red background
- keep TM-style ops muted gray
- use subtle olive/pink-forward category tinting for non-TM ops while preserving strong text contrast
- keep highlighted rows using their original op-category text color; use background/border emphasis instead of recoloring the text
- color `FastReduceNCDeviceOperation` with the reduce bucket, not the CCL bucket
- color `SoftmaxDeviceOperation` with the special bucket to match the real generated preview output
- if the user wants `Device Time` emphasis, highlight whole rows with `Device Time > 50 us` in the `Performance Report`
- preserve a distinct medium-severity highlight (`100-200 us`) when such rows exist; do not flatten everything to gray/red only
- if the user keeps the `High Op-to-Op Gap` advice section, re-highlight only that section by `Op-to-Op Gap`, not by `Device Time`
- when re-highlighting `High Op-to-Op Gap`, only apply the highlight to actual op rows, not plain text, notes, or URLs

Use severity buckets for `Device Time` row highlighting:
- `50-100 us` -> gray left border and light gray tint
- `100-200 us` -> pink left border and light pink tint
- `200+ us` -> stronger rose/red left border and light rose tint

Use severity buckets for `High Op-to-Op Gap` row highlighting when that advice section is present:
- `100-200 us` -> gray left border and light gray tint
- `200-400 us` -> amber left border and light amber tint
- `400+ us` -> rose left border and light rose tint

Append the summary PNG below the text report when the user wants the chart embedded in the page.
Render the embedded summary image at a fixed width of `600px` with `max-width: 100%` so it stays consistent while still shrinking on narrower previews.

Keep a single canonical HTML report by default.
Do not create a duplicate preview copy in `generated/live_preview/` unless the user explicitly asks for it or the editor cannot preview the canonical path.

If there are multiple copies because the user explicitly requested them, regenerate them all from the same source or copy the canonical file so the styling stays identical.

Use `report_preview_example.html` in this skill directory as the visual reference example for preferred HTML preview formatting.

## Index Page Template

If the user wants an overview page for multiple slices, create a compact `index.html` that combines the different slice types into one summary:

- start from `index_template.html` in this skill directory and replace the placeholders
- prefer the same code-like monospace visual direction as the published GitHub Pages index
- top header: model name, platform, and batch size
- short lead: explain that the run was first traced as a smaller model/decode window, then representative full layers and sublayers were sliced from that same run
- summary cards: use only the highest-signal numbers
- use forest-toned boxes with deeper mossy surfaces and visible pink accents on headings, pills, and card chrome

Preferred summary-card pattern:
- trace scope: model, platform, decode window, batch size
- representative even full-layer device perf
- representative odd full-layer device perf
- estimated total device perf, if the user asked for extrapolation
- device-perf `tokens/s/user`, if derived from estimated device time
- end-to-end `tokens/s/user`, if the user provides or asks for a benchmark/CI number

When mixing numbers from different sources, label them explicitly:
- `Device-Perf ...` for values derived from Tracy/`tt-perf-report`
- `End-to-End ...` for values from pytest, CI, or benchmark logs

Recommended data sources for the index:
- full traced `decode_2` slice: baseline decode device time
- representative `even_full` and `odd_full` slices: layer-level device times
- representative `even_attention`, `odd_attention`, `even_moe`, `odd_moe` slices: sublayer drill-down pages
- optional CI/pytest benchmark result: end-to-end throughput

If the index includes an estimated total GPT-OSS decode device time, calculate it from the traced reduced-layer decode run plus the missing alternating layers:

```text
gpt_oss_20b_total_decode_device_time ~= device_time_at_num_layers + (12 - measured_even_layers) * even_layer_device_time + (12 - measured_odd_layers) * odd_layer_device_time
```

Use the traced reduced-layer decode window to determine `measured_even_layers` and `measured_odd_layers`; do not hard-code a specific reduced-layer count unless the user explicitly gives one.

When showing both rate types on the index:
- derive `Device-Perf Tokens/s/User` from the estimated total decode device time
- label CI or benchmark throughput separately as `End-to-End Tokens/s/User`

Recommended layout below the summary:
- one section for even layers
- one section for odd layers
- in each section, show the full-layer report as the primary card
- place attention and MoE reports as sub-cards underneath the matching full layer

Keep the index concise:
- prefer one short lead paragraph and one short summary paragraph
- avoid long prose blocks if the cards already communicate the key numbers
- use plain wording like `batch size 64` instead of shorthand like `bs64` unless the user explicitly wants the shorthand

The template is intentionally generic for any model that:
- is organized into repeated layers or blocks
- has one or more drill-down part pages per layer
- needs a top-level summary that mixes traced device-perf numbers and optional end-to-end benchmark numbers

## Recommended Workflow

1. Verify the container, repo path, and `tt-perf-report` availability.
2. If the user wants the terminal-style report or HTML preview, run `tt-perf-report --no-advice` on the slice CSV with `--summary-file` and capture stdout.
3. Only if the user also wants the per-op CSV artifact, run a second `tt-perf-report --no-advice` command with `--summary-file` and `--csv`.
4. Treat the summary PNG as the default visualization artifact.
5. If requested, render the saved terminal report as a colored HTML preview using `tt-perf-report-preview.css`.

## References

- `tt-perf-report-preview.css`
- `report_preview_example.html`
- `index_template.html`
- `.cursor/skills/layer-profiling/SKILL.md`
