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
- Parsing `# ownership-area`, `## root-cause`, and `- test_id (arch) -> [job-link](url)` lines
- Normalising test IDs to ignore hardware arch suffixes when matching across runs
- Merging multiple arch variants of the same test into a single bullet line
- Rendering the final Markdown sections (or JSON)

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
three sets. Incorporate that output into the Step 5 report.

If falling back to manual comparison: a failure in COMPARISON_FAILURES matches a
failure in BASELINE_FAILURES when the test_id strings are identical, or differ only
in a trailing hardware arch token (e.g. `-n150`, `-p150`, `-n300-llmbox`). Compute:
- **New**: in COMPARISON_FAILURES but not in BASELINE_FAILURES
- **Persisting**: in both
- **Fixed**: in BASELINE_FAILURES but not in COMPARISON_FAILURES

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
