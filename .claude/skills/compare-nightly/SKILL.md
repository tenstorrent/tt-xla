---
name: compare-nightly
description: Compare two nightly GitHub Actions runs and classify failures as new, persisting, or fixed
disable-model-invocation: true
allowed-tools: Read, Read(/tmp/**), Write(/tmp/**), Glob, Grep, Skill, Bash(gh api *), Bash(gh api * > /tmp/**), Bash(jq *), Bash(mkdir -p /tmp/**), Bash(rm -rf /tmp/**), Bash(python3 *), Bash(for *)
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

## Step 1 — Analyze each run using analyze-nightly

Invoke the analyze-nightly skill twice using the Skill tool, once for each run, passing the
"save" argument so results are stored to disk:

1. Invoke skill `analyze-nightly` with args `$0 save`
   — waits for completion, then reads `/tmp/nightly-analysis-$0.md` (baseline analysis)
2. Invoke skill `analyze-nightly` with args `$1 save`
   — waits for completion, then reads `/tmp/nightly-analysis-$1.md` (comparison analysis)

Do not proceed to Step 2 until both analyses are complete and both files exist.

## Step 2 — Extract failures and compare using analyse_failures.py

A reference Python script `analyse_failures.py` lives alongside this skill file.
Copy it to /tmp/ and run it to parse both analysis files and produce the comparison:

```bash
cp .claude/skills/compare-nightly/analyse_failures.py /tmp/analyse_failures.py
python3 /tmp/analyse_failures.py /tmp/nightly-analysis-$0.md /tmp/nightly-analysis-$1.md
```

The script outputs a Markdown report covering New / Persisting / Fixed sections directly.
Pass `--json` to get structured JSON output instead if you need to post-process results.

The script handles:
- Parsing `# ownership-area`, `## root-cause`, and `- test_id (arch) -> [link](url)` lines
  (any link label is accepted, e.g. `[job-link]`, `[baseline job]`, `[comparison]`)
- Normalising test IDs to ignore hardware arch suffixes when matching across runs
  (strips *all* trailing arch tokens, longest-first)
- Merging multiple arch variants of the same test into a single bullet line
- Rendering the final Markdown sections (or JSON)

**Matching across runs is two-pass.** Pass 1 matches on the arch-normalised test
ID. Pass 2 then reconciles the leftovers by token signature, to absorb the case
where the two analyze-nightly passes labelled the SAME underlying failure with
different naming conventions (e.g. a `perf vllm_bge_m3` label vs a
`test_vllm_benchmarks.py::...[bge_m3]` pytest nodeid). Without this, such a
failure was double-counted as both New *and* Fixed — the known cause of inflated
new/fixed counts. Reconciliation requires ≥2 shared discriminating tokens and
≥0.8 containment (so e.g. `qwen3_4b` is never merged with `qwen3_8b`, nor
`batch1` with `batch32`), and uses greedy best-first assignment.

**Trust the script's counts** — do not hand-correct them as "unreliable." When
the two files use inconsistent naming, the script now reconciles automatically:
- Reconciled pairs are reported as **persisting** and tagged
  `_(matched across naming drift)_`, with a `> **Note:**` block at the top of the
  report and `WARNING:` lines on stderr listing each reconciled pair.
- A test that **persists but changes failure mode** (e.g. `pcc=nan` ->
  `std::bad_alloc`) is tagged `_(root cause changed: "..." -> "...")_`.
- The stderr summary line reports `N new, N persisting (N reconciled across
  naming drift), N fixed`.

In `--json` mode the object additionally contains a `reconciled` array (the
persisting items that were matched via pass 2) and a `warnings` array (one string
per reconciled pair). Each persisting item carries `reconciled`,
`root_cause_changed`, and `root_cause_baseline` fields.

If the script is unavailable or the output needs manual correction, fall back to parsing
the files by hand: for each bullet line extract `test_id` (text before the first ` (`),
`arch` (text between `(` and `)`), `root_cause` (most recent `##` heading), and
`job_url` (URL inside `[job-link](...)`). Build two lists — BASELINE_FAILURES and
COMPARISON_FAILURES — then proceed to Step 4.

## Step 3 — Collect commits merged between the two runs

Fetch the head SHA that each run was triggered on:
`gh api "repos/tenstorrent/tt-xla/actions/runs/{run-id}" --jq '.head_sha'`

Do this for both $0 (baseline SHA) and $1 (comparison SHA).

Then fetch the full list of commits on the main branch between those two SHAs (exclusive
of the baseline SHA, inclusive of the comparison SHA):
`gh api "repos/tenstorrent/tt-xla/compare/{baseline-sha}...{comparison-sha}" --jq '.commits[] | {sha: .sha, short_sha: .sha[0:8], author: .commit.author.name, message: (.commit.message | split("\n")[0])}'`

If the API returns an empty list or an error, note it and continue.

For each commit, produce a structured per-commit summary using a for loop.

### Per-commit summary

For each commit SHA, fetch the full commit details in a single call:
`gh api "repos/tenstorrent/tt-xla/commits/{sha}" --jq '{message: .commit.message, files: ([.files[].filename | split("/")[0]] | unique)}'`

This returns:
- `message`: the full commit message (title + body, which often includes the PR description)
- `files`: top-level directories of changed files

From the commit body, derive:
- **Why:** one sentence explaining the motivation (use the PR description body if present; otherwise infer from the commit title)
- **Benefit:** one sentence on what this improves for users or the project

Produce a structured summary:
```
**`{short_sha}`** — {commit title} (by {author})
- **Why:** {motivation}
- **Files:** {comma-separated list of top-level directories or key files changed}
- **Benefit:** {what this improves}
```

Separate each commit summary with a horizontal rule (`---`).

### Special handling for tt_forge_models uplift commits

If a commit title contains "Uplift third_party/tt_forge_models" or a changed file path
is `third_party/tt_forge_models`, apply the following extra steps instead of the generic
"Files" line:

1. Extract the old and new submodule SHAs from the commit's patch. Fetch the commit and
   find the file entry for `third_party/tt_forge_models`:
   `gh api "repos/tenstorrent/tt-xla/commits/{sha}" --jq '.files[] | select(.filename == "third_party/tt_forge_models") | .patch'`
   Parse the patch for lines `-Subproject commit {OLD_SUB_SHA}` and `+Subproject commit {NEW_SUB_SHA}`.

2. List commits in the tt-forge-models submodule between those two pointers:
   `gh api "repos/tenstorrent/tt-forge-models/compare/{OLD_SUB_SHA}...{NEW_SUB_SHA}" --jq '.commits[] | {sha: .sha[0:8], full_sha: .sha, message: (.commit.message | split("\n")[0])}'`

3. For each submodule commit, fetch the top-level model directories changed:
   `gh api "repos/tenstorrent/tt-forge-models/commits/{full_sha}" --jq '[.files[].filename | split("/")[0]] | unique | .[]'`

4. Replace the **Files** line with a **Models affected** table:

   | Submodule commit | Change | Model(s) |
   |---|---|---|
   | `{short_sha}` | {commit title} | `model_a`, `model_b` |

## Step 4 — Match failures across runs

If `analyse_failures.py` was used in Step 2 it already computed and rendered the
three sets, including any naming-drift reconciliation. Incorporate that output
into the Step 5 report verbatim — keep its `> **Note:**` block and the
`_(matched across naming drift)_` / `_(root cause changed: ...)_` tags. Do **not**
second-guess or hand-correct the counts; the reconciliation logic exists
precisely to handle the naming-convention mismatch that previously made manual
correction necessary.

If falling back to manual comparison: a failure in COMPARISON_FAILURES matches a
failure in BASELINE_FAILURES when the test_id strings are identical, or differ only
in a trailing hardware arch token (e.g. `-n150`, `-p150`, `-n300-llmbox`). Compute:
- **New**: in COMPARISON_FAILURES but not in BASELINE_FAILURES
- **Persisting**: in both
- **Fixed**: in BASELINE_FAILURES but not in COMPARISON_FAILURES

When matching by hand, also reconcile entries that name the **same underlying
test under different conventions** — most commonly a perf label such as
`perf vllm_bge_m3` in one file vs a pytest nodeid such as
`test_vllm_benchmarks.py::...[bge_m3]` in the other. These are the SAME test;
classify them as **persisting**, not as both new and fixed. Match on the shared
discriminating tokens (model name, size, batch), and keep the size/batch distinct
(`qwen3_4b` ≠ `qwen3_8b`, `batch1` ≠ `batch32`). If a persisting test's root
cause differs between the two files, note the change rather than treating it as
two separate failures.

## Step 5 — Produce the report

Output the following Markdown report. Do not emit a section if it is empty.

```
# Commits Merged Between Runs

_(baseline: {baseline-sha}, comparison: {comparison-sha})_

**`{short_sha}`** — {commit title} (by {author})
- **Why:** {motivation}
- **Files:** {top-level directories changed}
- **Benefit:** {what this improves}

---

**`{short_sha}`** — Uplift third_party/tt_forge_models (by {author})

| Submodule commit | Change | Model(s) |
|---|---|---|
| `{sub_sha}` | {submodule commit title} | `model_a`, `model_b` |

---

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
- Clean up temporary files written by this skill in /tmp/ when done, but do NOT
  delete the `/tmp/nightly-analysis-*.md` files until after comparison is complete.
- If an analyze-nightly invocation fails, note it and continue — do not abort.
- If a run has already been analyzed and `/tmp/nightly-analysis-{run-id}.md` exists,
  you may skip re-invoking analyze-nightly for that run and read the cached file directly.
