---
name: perf-hub-compare
description: Publish or update GitHub Pages perf hubs under github_pages/perf_reports, wire the side-by-side compare_reports.html comparator, and maintain the fusion-count table on run index pages. Use when adding a new traced run next to an existing one, comparing 03-25 vs 03-27 style layouts, updating compare_reports segments, or editing perf hub index.html fusion tables and summary cards.
---

# Perf hub & run comparison (GitHub Pages)

Use this skill when the work is about **static perf hubs** in the repo (`github_pages/perf_reports/`), not about running Tracy or `tt-perf-report` (those stay in `.cursor/skills/layer-profiling/SKILL.md` and `.cursor/skills/tt-perf-report/SKILL.md`).

**Intended workflow:** assume **at least one published run** already exists (overview + HTML reports). The task is usually to **add a new test run** and **compare it to a chosen baseline** via the hub list, per-run `index.html`, and `compare_reports.html`.

## Repository layout

```text
github_pages/perf_reports/
  index.html                    # Lists runs; links to comparator
  compare_reports.html          # Side-by-side iframes; segment picker
  <YYYY_MM_DD_HH_MM_SS>/        # One folder per capture / publish
    tt_perf/
      index.html                # Run overview: mesh, fusion table, summary cards, report links
      *_performance_report.html # Colored tt-perf-report HTML previews
```

Paths in links are **relative to `perf_reports/`** (e.g. `2026_03_27_12_21_16/tt_perf/index.html`).

## Adding a new run (baseline hub already exists)

1. **Artifacts:** Place the new run’s colored HTML reports under `github_pages/perf_reports/<NEW_STAMP>/tt_perf/`. Filenames should match segments you will wire in the comparator (see below).
2. **Run overview:** Copy the **closest prior** `tt_perf/index.html` as a template. Adjust:
   - **Lead** line: test name, mesh / sharding story, batch.
   - **Mesh box** (`mesh-axes`): Galaxy **4×8**, `batch` axis **4** (rows), `model` axis **8** (columns); `<dd>` lines describe how each axis is used for **this** run (dual-axis weight shard vs model-axis TP + DP `input_ids`, etc.).
   - **Fusion table** (see next section): counts for LayerNorm, SDPA decode, RoPE; **full (NL)** header where **N** = number of transformer layers in the **traced decode window** for this publish (e.g. 4 vs 6).
   - **Summary cards:** even/odd full-layer ms, estimated total device time, device-perf tok/s/user, end-to-end card if applicable. If the benchmark **does not complete** (hang), say so in the end-to-end detail (e.g. **full model hangs**) and keep value **—** if no number.
   - **Report cards:** ensure `href` / `meta` filenames match real files; add **Traced decode window** card + `decode_2_performance_report.html` only when that HTML exists.
3. **Perf hub root:** Add a `<a class="run-card">` in `github_pages/perf_reports/index.html` for `<NEW_STAMP>` with date, title, one-line pairing hint (which older run to open in the comparator).
4. **Comparator:** Update `github_pages/perf_reports/compare_reports.html`:
   - Set `RUN_A` / `RUN_B` to the **relative** `.../tt_perf/` paths for **baseline** vs **new** run (order is arbitrary but keep pane headers consistent).
   - Update the **pane `<header>`** text to name each run and test type.
   - Update **`SEGMENTS`:** each entry either has `file` (both sides) or `fileLeft` / `fileRight` when a segment exists on only one publish (e.g. full decode window HTML only on the newer run).
   - Toolbar quick links (`<a href=".../tt_perf/index.html">`) should point at the same two runs.
   - Optional deep link: `compare_reports.html?s=<segment_id>` (also `segment` query param or hash supported by the page script).

Do **not** require regenerating baseline HTML when only adding a new run unless the user wants parity fixes on the old hub.

## Fusion table (run `index.html`)

Use a **tabular** `<table class="fusion-table">` inside `.fusion-stats`, not prose paragraphs.

| Column | Content |
|--------|---------|
| Fusion | Short label: LayerNorm, SDPA decode, RoPE |
| Operation | `tt-perf-report` device op name in `<code>` (e.g. `LayerNormDeviceOperation`, `SdpaDecodeDeviceOperation`, `TernaryDeviceOperation` for RoPE in captures that map RoPE that way) |
| attn | Count in the **attention** slice HTML / table for this op |
| MoE | Count in the **MoE** slice |
| full **(NL)** | Total count in the **full traced decode** report for this op; **N** in the header is the **layer count** in that decode window for this publish (same **N** for every row — **no separate layers column**) |

Style: `thead` uses small caps except the last column: use class `fusion-col-window` and CSS `text-transform: none` so the header reads **`full (4L)`** or **`full (6L)`** (not `FULL (4L)`).

Counts are **not** required to satisfy attn + MoE = full; **full** is the multi-layer window total.

## Comparator segment list (`compare_reports.html`)

The `SEGMENTS` array drives the `<select>`. Typical `id` values:

- `index` → `index.html` on both sides (run overview: mesh, fusion table, summary).
- `even_full`, `odd_full` → `even_full_performance_report.html`, `odd_full_performance_report.html`.
- `decode_2` → often **asymmetric**: `fileLeft: null`, `fileRight: "decode_2_performance_report.html"` when the older publish has no full-decode HTML; left iframe uses inline `srcdoc` explaining the gap.
- `even_attention`, `odd_attention` → attention sublayers.
- `even_moe`, `odd_moe` → omit or use `fileLeft: null` if a file is missing on one side; otherwise same `file` on both.

After edits, smoke-check: open `compare_reports.html`, switch each segment, confirm **left/right** src and blank pane messages.

## Reference pairing (Galaxy GPT-OSS examples)

Concrete published pair in-repo:

| Run folder | Test focus |
|------------|------------|
| `2026_03_25_12_13_39` | Dual-axis weight-sharded layout; **6** layers in traced decode window; fusion table **full (6L)**; has end-to-end number from CI in summary |
| `2026_03_27_12_21_16` | Model-axis TP + DP `input_ids` shard; **4** layers in traced decode window; fusion table **full (4L)**; `decode_2_performance_report.html` present; end-to-end card documents **full model hangs** |

Mesh captions are **4×8** with batch **4** / model **8** on both; the **`<dd>` explanations** differ by run.

## Cross-skills

- Generate slice CSVs, Tracy workflow, extrapolation math: `.cursor/skills/layer-profiling/SKILL.md`
- Run `tt-perf-report`, build colored HTML report files: `.cursor/skills/tt-perf-report/SKILL.md`
- GPT-OSS layer boundaries: `.cursor/skills/gpt-oss-layer-parsing/SKILL.md`
