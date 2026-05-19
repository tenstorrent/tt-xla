---
name: update-test-config-from-ci-runs
description: Automate updates to test_config YAMLs under tests/runner/test_config/ based on a nightly/weekly CI run. Use when the user wants to sync test configs with CI results.
disable-model-invocation: false
allowed-tools: Read, Edit, Write, AskUserQuestion, Bash

context: fork
model: opus
---

## Step 1 — Gather required inputs from the user

Before doing anything else, ask the user for the pieces of information this
skill cannot proceed without. Use the `AskUserQuestion` tool to collect them
(one question per item, so the user can answer each independently):

1. **GitHub token** — needed to authenticate `gh` so the skill can
   download nightly/weekly run artifacts and logs from
   `github.com/tenstorrent/tt-xla`. Ask the user to paste a Personal Access
   Token (classic or fine-grained) with at least `repo` and `actions:read`
   scope. Do not echo the token back in plain text in summaries; only use it
   to authenticate `gh` for the current session.

2. **Run cadence** — ask whether the run to sync is from the **nightly** or
   **weekly** CI pipeline. Present this as a single-select question with the
   two options `nightly` and `weekly`. This determines which workflow file
   (`schedule-nightly.yml` vs `schedule-weekly*.yml`) the skill will consult
   later to map jobs → architectures.

3. **Run ID** — the GitHub Actions run id whose results the skill should sync
   into the YAMLs. Accept either a bare numeric id (e.g. `12345678901`) or a
   full run URL
   (`https://github.com/tenstorrent/tt-xla/actions/runs/12345678901`); if a
   URL is given, extract the trailing numeric id.

4. **Type of model** *(optional)* — ask the user which model category to sync.
   Present as a single-select question with options like `language` or `vision`
   (model families grouped by domain, not individual model names). If the
   user leaves this blank or says "all", the skill should consider every
   model in the run. Use this only as a filter when matching model-ids in
   later steps.

Do not assume defaults for any required value (token, cadence, run ID). The
"type of model" question is the only optional one. If the user declines to
provide a required answer, stop and tell them the skill cannot continue
without it.

## Step 2 — Export the GitHub token to the environment

Once the user has provided their GitHub token in Step 1, export it so `gh`
and any subsequent commands in this session can authenticate against
`github.com/tenstorrent/tt-xla`:

```bash
export GITHUB_TOKEN=<token-from-step-1>
```

Notes:
- Substitute `<token-from-step-1>` with the exact value the user provided.
- Do **not** echo, print, or otherwise display the token value back to the
  user or in any summary output — only use it to set the env var.
- Run this `export` inside the Bash tool so it applies to subsequent `gh`
  calls in the same shell session.

## Step 3 — Download the run's report artifacts

Once the user has provided the run ID and run cadence in Step 1, download the
report artifacts for that run using the project's helper script:

```bash
python .github/scripts/download_artifacts.py --run-id <run-id> -o <cadence>_<monthday> --filter report
```

Substitutions:
- `<run-id>` — replace with the numeric run ID the user provided in Step 1
  (question 3). If the user gave a URL, use only the trailing numeric id.
- `<cadence>` — replace with `nightly` or `weekly`, based on the user's
  answer to the run-cadence question in Step 1 (question 2).
- `<monthday>` — replace with the lowercase 3-letter month abbreviation
  followed by the day-of-month of the run, with no separator (e.g.
  `jan14`, `may19`, `dec03` — pad the day only if the user has been
  using a 2-digit form; otherwise use the bare day number as in the
  example `weekly_jan14`). Determine the date from the run itself
  (`gh run view <run-id> --json createdAt`), not from "today's" date,
  so the directory name matches the run's actual day.

Example invocations:
- Nightly run from 14 Jan: `-o nightly_jan14`
- Weekly run from 19 May: `-o weekly_may19`

Run the command from the repository root (`/home/tt-xla`) via the Bash tool.
After it finishes, confirm the output directory exists and contains the
downloaded report files before moving on.

## Step 4 — Summarize the downloaded JUnit XMLs

Run the project's JUnit summarizer against the directory downloaded in
Step 3. The `--xml` argument must use the **same** directory name produced
in Step 3 (e.g. `weekly_jan14`, `nightly_may19`) — do not hard-code
`weekly_jan14`; carry the value forward from Step 3.

The output log file name embeds **both** the run cadence (from Step 1,
question 2) and the run's month-day (the same `<monthday>` derived from
the run's `createdAt` and used in Step 3):
- If cadence is `nightly` → redirect to `../nightly_<monthday>.log`
  (e.g. `../nightly_may19.log`)
- If cadence is `weekly`  → redirect to `../weekly_<monthday>.log`
  (e.g. `../weekly_may19.log`)

In other words, the log file name mirrors the artifact directory name from
Step 3, just with a `.log` suffix instead of being a directory.

Branch on whether the user provided an answer to the optional "type of
model" question (Step 1, question 4):

- **If the user did NOT provide a type of model** (left blank / answered
  "all"):

  ```bash
  python .github/scripts/summarize_junit_xmls.py --xml <cadence>_<monthday> &> ../<cadence>_<monthday>.log
  ```

- **If the user DID provide a type of model** (e.g. `language`, `vision`):

  ```bash
  python .github/scripts/summarize_junit_xmls.py --xml <cadence>_<monthday> --model-type <model-type> &> ../<cadence>_<monthday>.log
  ```

  Where `<model-type>` is the exact value the user selected
  (e.g. `language` or `vision`).

Notes:
- `<cadence>_<monthday>` must match the `-o` value used in Step 3
  (e.g. if Step 3 used `-o weekly_jan14`, then Step 4 must pass
  `--xml weekly_jan14` and write to `../weekly_jan14.log`).
- A nightly run from 14 Jan writes to `../nightly_jan14.log`; a weekly
  run from 19 May writes to `../weekly_may19.log`. Do not mix the
  cadence/date with another run's name.
- Run from the repository root via the Bash tool. After it completes,
  verify the chosen log file (`../<cadence>_<monthday>.log`) exists
  before moving on.

## Step 5 — Filter the summary log by the `guidance` column

Read the log file written in Step 4 (`../<cadence>_<monthday>.log`, e.g.
`../nightly_may18.log` or `../weekly_may16.log`) and produce a filtered
version that contains only the rows where the **last column** (`guidance`)
is **not** `N/A`.

### Log format

The Step 4 log is whitespace-aligned columnar text. The first non-blank
line is the header; example columns (in order):

```
specific_test_case  model_type  group  weights_dtype  arch  bringup_status  model_status  pcc  pcc_thres  pcc_en  parallelism  time  guidance
```

Each subsequent line is one test row, with values padded by spaces so the
columns line up. The `guidance` value is the **final whitespace-separated
token** on each row. Examples from `nightly_may18.log`:

- Header line — keep it.
- `... time 930.792 N/A` → `guidance == N/A` → **drop**.
- `... 22.852 N/A` → `guidance == N/A` → **drop**.
- `... 22.852 PROMOTE_TO_EXPECTED_PASSING` (or any non-`N/A` token)
  → **keep**.

### Output file

Write the filtered output next to the input file (same directory — one
level above the repo root). Name it by prefixing `changes_` to the input
file's base name:

- `../nightly_may18.log` → `../changes_nightly_may18.log`
- `../weekly_may16.log` → `../changes_weekly_may16.log`

### Required behavior

- **Always preserve the header row** (the line that names the columns)
  so the filtered file is still a readable table.
- **Drop** any data row whose final whitespace-separated token equals
  exactly `N/A`.
- **Keep** any data row whose final whitespace-separated token is
  anything else (any non-empty value that is not the literal `N/A`).
- Preserve the original column alignment / spacing for kept rows — do
  **not** reformat or re-tabulate; output the line verbatim.
- Skip fully blank lines silently (do not write them to the output).
- If no rows pass the filter (only the header would remain), still
  write the header to the output file and mention this in the
  conversation summary.

### Implementation hint

Use a single `awk` invocation from the Bash tool so we don't need a
temporary Python script:

```bash
awk 'NR==1 {print; next} NF>0 && $NF != "N/A" {print}' ../<cadence>_<monthday>.log > ../changes_<cadence>_<monthday>.log
```

After writing, report the line count of the new file
(`wc -l ../changes_<cadence>_<monthday>.log`) so the user knows how many
rows survived the filter.

## Step 6 — Route each changes row to the correct test_config directory

For every row in `../changes_<cadence>_<monthday>.log` (the file produced in
Step 5), determine **which test_config directory** the corresponding YAML
lives in by looking at the test-function key prefix of the
`specific_test_case` column (the first column of the row):

| Row's test-function key   | Target directory                          |
|---------------------------|-------------------------------------------|
| `test_all_models_jax[...]`   | `tests/runner/test_config/jax`        |
| `test_all_models_torch[...]` | `tests/runner/test_config/torch`      |
| `test_llms_torch[...]`       | `tests/runner/test_config/torch_llm`  |

Notes:
- Paths are relative to the repository root (`tt-xla/`). Do **not** prefix
  them with `/home` — use repo-relative paths so the skill works regardless
  of where the repo is checked out.
- The key is the identifier **before** the `[` in `specific_test_case`
  (e.g. `test_all_models_torch[bert/...]` → key is `test_all_models_torch`).
- If a row's key does not match any of the entries in the table above,
  do not guess — flag the row to the user and skip it.

### Step 6a — Pick the YAML file within the target directory

Each `test_config/<framework>/` directory contains several YAML files, one
per `(run_mode, parallelism)` combination. Pick the file by reading the
`<parallelism>-<run_mode>` segment of the `specific_test_case` (the tokens
that appear just before the closing `]`).

The filename pattern is:

```
test_config_<run_mode>_<parallelism>.yaml
```

Concretely, for the `jax` directory the four files are:

| `<parallelism>-<run_mode>` segment in test name | Target YAML                                                                 |
|-------------------------------------------------|-----------------------------------------------------------------------------|
| `single_device-inference`                       | `tests/runner/test_config/jax/test_config_inference_single_device.yaml`     |
| `tensor_parallel-inference`                     | `tests/runner/test_config/jax/test_config_inference_tensor_parallel.yaml`   |
| `data_parallel-inference`                       | `tests/runner/test_config/jax/test_config_inference_data_parallel.yaml`     |
| `single_device-training`                        | `tests/runner/test_config/jax/test_config_training_single_device.yaml`      |

Worked example:

- Row: `test_all_models_jax[blenderbot/summarization/jax-3B-single_device-inference]`
- Step 6 directory: `tests/runner/test_config/jax`
- `<parallelism>-<run_mode>` segment: `single_device-inference`
- → Target YAML:
  `tests/runner/test_config/jax/test_config_inference_single_device.yaml`

The `torch` directory follows the **same** filename pattern as `jax`, plus
one additional file for tensor-parallel training:

| `<parallelism>-<run_mode>` segment in test name | Target YAML                                                                   |
|-------------------------------------------------|-------------------------------------------------------------------------------|
| `single_device-inference`                       | `tests/runner/test_config/torch/test_config_inference_single_device.yaml`     |
| `tensor_parallel-inference`                     | `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`   |
| `data_parallel-inference`                       | `tests/runner/test_config/torch/test_config_inference_data_parallel.yaml`     |
| `single_device-training`                        | `tests/runner/test_config/torch/test_config_training_single_device.yaml`      |
| `tensor_parallel-training`                      | `tests/runner/test_config/torch/test_config_training_tensor_parallel.yaml`    |

The `torch_llm` directory only contains the two inference YAMLs:

| `<parallelism>-<run_mode>` segment in test name | Target YAML                                                                       |
|-------------------------------------------------|-----------------------------------------------------------------------------------|
| `single_device-inference`                       | `tests/runner/test_config/torch_llm/test_config_inference_single_device.yaml`     |
| `tensor_parallel-inference`                     | `tests/runner/test_config/torch_llm/test_config_inference_tensor_parallel.yaml`   |

Notes:
- The same `<run_mode>` / `<parallelism>` extraction rule applies across all
  three directories — only the set of filenames present in each directory
  differs.
- If the test name's `<parallelism>-<run_mode>` segment does not map to any
  YAML in the target directory, do not guess — flag the row to the user and
  skip it.
- Ignore `test_config_placeholders.yaml` in the `torch` directory — it is
  not a destination for changes coming from CI rows.

(More sub-steps to be added by the user — wait for further instructions
before proceeding past this YAML-file selection.)

## Step 7 — Build `final_report.log` with a per-row summary

Produce a human-readable report that combines every row from the Step 5
filtered log with a one-line summary describing the cross-arch status of
that test case.

### 7a — Output location and shape

Create a new file (alongside the other logs — same directory as
`../changes_<cadence>_<monthday>.log`):

```
../final_report.log
```

For every data row in `../changes_<cadence>_<monthday>.log`:
1. Copy the row **verbatim** into `final_report.log` (preserve spacing
   and column alignment).
2. Immediately after the row, write **one summary line** describing the
   row, then a blank line for readability.

The Step 5 header row is copied once at the top; no summary is written
under it.

### 7b — Skip training rows (for now)

If the row's run-mode segment is `training` (i.e. the last token inside
`[...]` after the final `-` is `training`, e.g. `...-single_device-training`,
`...-tensor_parallel-training`):

- Still copy the row into `final_report.log`.
- **Do not** compute or write a summary line for it. Either leave the
  summary slot empty or write a single explicit marker line such as
  `summary: (training case — skipped for now)`.

Training cases are intentionally out of scope for Step 7 until the user
extends the skill.

### 7c — Inference rows: cross-arch lookup

The cross-arch lookup (n150 ↔ p150) **only applies to single-device
inference rows**. For any other parallelism mode (e.g.
`tensor_parallel-inference`, `data_parallel-inference`), do not look up
a counterpart — these rows typically run on different hardware
(`n300-llmbox`, `galaxy-wh-6u`, etc.) and have no n150/p150 pairing.
Summarize them in place by describing their own status (bringup,
pcc vs. threshold) and skip Steps 7c.2 – 7c.4.

Detect single-device inference by either:
- the row's `parallelism` column being `single_device`, **and**
- the test name's `<parallelism>-<run_mode>` segment being
  `single_device-inference`.

For rows that pass this single-device-inference check, build the summary
by inspecting the row and looking up the same test case under the
**other** arch.

Columns of interest in the row (from Step 4's summarizer):
- `arch` — usually `n150` or `p150` (the row's own arch).
- `bringup_status` — e.g. `PASSED`, `INCORRECT_RESULT`, `FAILED_RUNTIME`.
- `pcc` — measured PCC for this row.
- `pcc_thres` — configured PCC threshold for this row.

#### Step 7c.1 — Decide whether *this* row is passing

Compare `pcc` against `pcc_thres`. If `pcc` is a numeric value `>= 0.99`
(and `bringup_status` is not a hard failure like `FAILED_RUNTIME`), treat
this row as **passing** in its arch. Otherwise treat it as **not passing**
and let the summary reflect that directly (e.g. report `bringup_status` and
the measured `pcc`).

#### Step 7c.2 — Identify the "other arch"

If the row's `arch` is `n150`, the other arch is `p150` (and vice versa).
Other arch values (e.g. `n300-llmbox`, `galaxy-wh-6u`, `p150b`) are not
part of the n150 ↔ p150 pair and should be summarized without a cross-arch
lookup — just describe the row itself.

#### Step 7c.3 — Look up the same test case under the other arch

For the same `specific_test_case` (the first column) but with the other
arch:

1. First, look inside `../changes_<cadence>_<monthday>.log` itself —
   the counterpart may already be in the filtered list.
2. If not found there, look inside the full Step 4 log
   (`../<cadence>_<monthday>.log`) — the counterpart may exist but had
   `guidance == N/A` and was filtered out at Step 5.

Match strictly on the `specific_test_case` string. The arch column is
what distinguishes the two rows.

#### Step 7c.4 — Compose the summary line

Use the lookup result to pick one of the following summary shapes
(adapt wording but keep it one line):

- **Counterpart not found in either log:**
  `summary: passing in <this_arch>; no entry found for <other_arch>`
- **Counterpart found with `bringup_status == FAILED_RUNTIME`:**
  `summary: passing in <this_arch>; model not supported in <other_arch> (FAILED_RUNTIME)`
- **Counterpart found with `bringup_status == PASSED` and `pcc >= 0.99`:**
  `summary: already passing in <other_arch>; now also passing in <this_arch>`
- **Counterpart found with another status** (e.g. `INCORRECT_RESULT`,
  `XFAIL` with low pcc): describe the status and measured pcc, e.g.
  `summary: passing in <this_arch>; <other_arch> is <bringup_status> with pcc=<value>`

Worked examples (taken from the user's notes):

- `bi_lstm_crf/pytorch-Default-single_device-inference` is in the
  changes log at `n150` with `pcc=1.0`. If no `p150` counterpart is
  found anywhere, the summary should be:
  `summary: passing in n150; no entry found for p150`
- `allam/causal_lm/pytorch-7B_Instruct-single_device-inference` is in
  the changes log at `p150` (passing). Looking up `n150` in the
  Step 4 log shows `bringup_status=FAILED_RUNTIME`, so the summary
  should be:
  `summary: passing in p150; model not supported in n150 (FAILED_RUNTIME)`

### 7c.5 — Append a `status` keyword to every summary line

Each summary line must end with a `status: true|false` token. The value is
determined by the row's run-mode and parallelism:

- **Training rows** (any parallelism): `status: false` — training cases are
  not yet treated as passing by this skill.
- **`single_device-inference` rows**: `status: true` if this row passes
  (`bringup == PASSED` and `pcc >= 0.99`) **and** the counterpart row in
  the other arch satisfies **any** of the following:
    - the counterpart is also `PASSED` with `pcc >= 0.99` (model passes
      on both archs), **or**
    - the counterpart's `bringup_status` indicates the model is **not
      supported** on that arch (e.g. `FAILED_RUNTIME`,
      `FAILED_FE_COMPILATION`) — model is passing on every arch where it
      is supposed to run, **or**
    - the counterpart is **missing** from both the changes log and the
      full Step 4 log (the test is simply not run on that arch). Treat
      "no entry" the same as "not applicable on that arch".

  Otherwise → `status: false`. In particular: if either pcc is below
  0.99, or the counterpart is `INCORRECT_RESULT` / any other functional
  failure → `false`.
- **`tensor_parallel-inference` rows** (e.g. on `n300-llmbox`,
  `galaxy-wh-6u`): `status: true` if this row's `pcc >= 0.99`, else
  `status: false`. No cross-arch lookup.
- **`data_parallel-inference` rows**: same rule as tensor parallel —
  `status: true` if `pcc >= 0.99`, else `false`.
- **Rows whose bringup status is a hard failure** (e.g. `FAILED_RUNTIME`,
  `FAILED_FE_COMPILATION`, `INCORRECT_RESULT`): `status: false`, regardless
  of pcc.

Write the `status` token on its **own line** directly under the summary
line (not appended inline with a separator).

Also: at the end of the `summary:` line itself, append the **target YAML
path** resolved in Step 6 / 6a (the YAML this row would update), using
the form `. Target YAML: <repo-relative-path>`.

The two-line block looks like:

```
summary: passing in p150; model not supported in n150 (FAILED_RUNTIME). Target YAML: tests/runner/test_config/jax/test_config_inference_single_device.yaml
status: false
```

More examples:

```
summary: already passing in n150; now also passing in p150. Target YAML: tests/runner/test_config/torch/test_config_inference_single_device.yaml
status: true

summary: passing in n300-llmbox (pcc=0.9999, thres=0.99). Target YAML: tests/runner/test_config/jax/test_config_inference_tensor_parallel.yaml
status: true

summary: (training case - skipped for now). Target YAML: tests/runner/test_config/torch/test_config_training_single_device.yaml
status: false
```

### 7d — Final file structure

`../final_report.log` should be readable as the same table from
Step 5 with summary + status lines interleaved, e.g.:

```
<header row from changes log>

<data row 1 verbatim>
summary: <one-line cross-arch summary>. Target YAML: <repo-relative path>
status: true|false

<data row 2 verbatim>
summary: (training case - skipped for now). Target YAML: <repo-relative path>
status: false

<data row 3 verbatim>
summary: <one-line cross-arch summary>. Target YAML: <repo-relative path>
status: true|false
...
```

After writing the file, report its path and the count of rows that
received a real summary vs. were skipped (training).

## Step 8 — Apply the updates to the test_config YAMLs

For every block in `../final_report.log` (each block is: the data row, the
`summary:` line, and the `status:` line):

1. **If `status: false`** — skip the block. No YAML edits.
2. **If `status: true`** — open the YAML named in the block's
   `Target YAML:` path (computed in Step 6 / 6a) and update the entry whose
   key matches the part inside `[...]` of the row's first column.

### How to find the entry to edit

The YAML key is exactly the test name **without the function-key prefix**
and **without the surrounding `[ ]`**. Examples:

| Row first column                                                                        | YAML key                                                                          |
|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `test_all_models_torch[bi_lstm_crf/pytorch-Default-single_device-inference]`            | `bi_lstm_crf/pytorch-Default-single_device-inference`                             |
| `test_llms_torch[qwen_2_5/causal_lm/pytorch-1.5B_Instruct-llm_decode-...-inference]`    | `qwen_2_5/causal_lm/pytorch-1.5B_Instruct-llm_decode-...-inference`               |

Look for that key as a YAML map key inside the target file. If the key is
not present in the file, flag the row to the user and skip it — do not
guess or invent a new entry.

### What to change

When the entry is found, normalize it to the **passing form**:

```yaml
<key>:
  status: EXPECTED_PASSING
```

Apply these transformations to whatever the current entry contains:

- **Set `status` to `EXPECTED_PASSING`.** If `status` was
  `KNOWN_FAILURE_XFAIL` (or any other non-passing value), overwrite it.
- **Remove the `reason:` field**, if present. (It was only relevant while
  the test was failing.)
- **Remove the `required_pcc:` override**, if present. Once the test passes
  at the default threshold (0.99), the per-test override is no longer
  needed.
- **Remove the `assert_pcc: false` line** (and any trailing inline comment
  on the same line) if the row's `guidance` column is `ENABLE_PCC_099` and
  the row is passing with `pcc >= 0.99`. This corresponds to the case where
  PCC assertion was previously disabled (often with a comment linking a
  compute-config / math-fidelity issue) but the test now passes the default
  0.99 threshold, so the disable is no longer warranted. Worked example —
  before:
  ```yaml
  gemma/pytorch-1.1_7B_IT-single_device-inference:
    supported_archs: ["p150"]
    status: EXPECTED_PASSING
    assert_pcc: false # ComputeConfig math_fidelity/fp32_dest_acc_en - https://github.com/tenstorrent/tt-xla/issues/2861
    optimization_level: 2
  ```
  After:
  ```yaml
  gemma/pytorch-1.1_7B_IT-single_device-inference:
    supported_archs: ["p150"]
    status: EXPECTED_PASSING
    optimization_level: 2
  ```
- Preserve any unrelated fields on the entry (e.g. `arch`-specific
  overrides, `supported_archs`, `optimization_level`) verbatim. Do not
  reorder or reformat other entries in the file.

### Worked examples

Before:
```yaml
bi_lstm_crf/pytorch-Default-single_device-inference:
  status: KNOWN_FAILURE_XFAIL
  reason: "RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet."
```

After:
```yaml
bi_lstm_crf/pytorch-Default-single_device-inference:
  status: EXPECTED_PASSING
```

Before:
```yaml
qwen_2_5/causal_lm/pytorch-1.5B_Instruct-llm_decode-seq_1-batch_1-single_device-mesh_default-inference:
  status: EXPECTED_PASSING
  required_pcc: 0.98
```

After:
```yaml
qwen_2_5/causal_lm/pytorch-1.5B_Instruct-llm_decode-seq_1-batch_1-single_device-mesh_default-inference:
  status: EXPECTED_PASSING
```

Before (guidance was `ENABLE_PCC_099`, pcc 0.9966 ≥ 0.99):
```yaml
gemma/pytorch-1.1_7B_IT-single_device-inference:
  supported_archs: ["p150"]
  status: EXPECTED_PASSING
  assert_pcc: false # ComputeConfig math_fidelity/fp32_dest_acc_en - https://github.com/tenstorrent/tt-xla/issues/2861
  optimization_level: 2
```

After (drop `assert_pcc: false` and its trailing comment; keep `supported_archs` and `optimization_level`):
```yaml
gemma/pytorch-1.1_7B_IT-single_device-inference:
  supported_archs: ["p150"]
  status: EXPECTED_PASSING
  optimization_level: 2
```

### Reporting

After applying all `status: true` blocks, report to the user:
- Total `status: true` blocks processed.
- The exact set of YAMLs modified (one line each).
- Any rows skipped because the YAML key was not found.
- Confirm that no `status: false` blocks were touched.

## Step 9 — Open a PR with the test_config updates

After Step 8 has modified the YAMLs, package the changes into a pull
request against `main`.

### No-op short-circuit (required)

Before doing anything in Step 9, check whether Step 8 actually modified
any files. Run this from the repo root:

```bash
git diff --quiet -- tests/runner/test_config/ && echo NO_CHANGES || echo HAS_CHANGES
```

If the output is `NO_CHANGES` (Step 8 made zero edits — every
`status: true` entry was already at the target state on `main`), do
**not** attempt to create a branch, commit, push, or open a PR. GitHub
rejects empty PRs, and an empty branch is just noise.

Instead, report clearly to the user:

> No test_config changes to commit — every `status: true` entry from
> `final_report.log` is already at the target state on `main` (e.g. a
> prior maintenance PR already promoted them). No PR opened.

Include the list of `status: true` keys that were already at target so
the user can verify the no-op was correct. Then exit Step 9.

Only proceed with the rest of Step 9 when `git diff` shows real changes
under `tests/runner/test_config/`.

### Automatic push and PR creation (no checkpoint)

The user has pre-authorized this skill to push the branch and open the PR
automatically. Do **not** stop to ask for confirmation before
`git push` or `gh pr create` — run the full sequence end-to-end and
report the resulting PR URL.

Sequence:

1. Create the branch off `main` (see "Branch name" below). If the branch
   already exists for the same run, reuse it.
2. Stage and commit only the test_config YAMLs that Step 8 modified
   (see "Commit" below).
3. `git push -u origin <branch>` to publish the branch.
4. `gh pr create --draft` against `main` with the title/body described
   in "PR" below (draft PR — not ready-for-review).
5. Print the PR URL to the user.

Only include the test_config YAMLs in the commit — do not stage unrelated
working-tree changes (e.g. local edits to `.claude/skills/...`,
downloaded artifact dirs under `nightly_<monthday>/` / `weekly_<monthday>/`,
`*.log` files in `/home`). Stash or otherwise exclude them before
committing.

This section supersedes any prior confirmation-checkpoint behavior — do
not pause and wait for explicit user go-ahead. The Step 9 no-op
short-circuit above is the only condition under which Step 9 stops
without opening a PR.

### Branch name

Branch name template:

```
<cadence>_maintance_<monthday>
```

- `<cadence>` — `nightly` or `weekly` (from Step 1).
- `<monthday>` — same lowercase `<mon><day>` token used in Steps 3–5
  (e.g. `may19`, `jan14`). Derived from the run's `createdAt`, not
  today's date.

Examples:
- Nightly run from 19 May → `nightly_maintance_may19`
- Weekly run from 14 Jan → `weekly_maintance_jan14`

Create the branch off the current `main` (or the project's default
branch) before staging the edits. If the branch already exists for the
same run, reuse it; do not append timestamps.

### Commit

- Commit only the test_config YAMLs that Step 8 modified. Do **not**
  include unrelated changes (the skill files under `.claude/`, the
  downloaded `nightly_may19/` artifact dir, the `*.log` files in `/home`,
  etc.) — those are workspace artifacts, not part of the PR.
- Commit message:

  ```
  <cadence> maintance <monthday>
  ```

  e.g. `nightly maintance may19`, `weekly maintance jan14`.

### PR

Open the PR as a **draft** via `gh pr create --draft` against `main`.
Draft PRs let the user review the diff in the GitHub UI and mark it
ready-for-review when satisfied, without immediately notifying
reviewers or running merge gates.

- **Title**: title-case form of the commit message —
  `Nightly Maintance <monthday>` or `Weekly Maintance <monthday>`
  (matches existing PRs in the project, e.g. PR #4739
  "Nightly Maintance may16").
- **Body**: a bulleted list of the `status: true` rows from
  `../final_report.log`. Each bullet should at minimum identify:
  - the test key (the part inside `[...]` of the row's first column),
  - the target YAML path, and
  - the action taken (e.g. "removed XFAIL", "removed required_pcc
    override").

Do not include `status: false` rows in the PR body — they were not
touched.

### Reporting

After the PR is opened, return the PR URL to the user.
