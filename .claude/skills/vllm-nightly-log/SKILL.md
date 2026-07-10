---
name: vllm-nightly-log
description: Extract the generated text from the vLLM jobs of a tt-xla GitHub Actions run (nightly schedule or manual Run Test with vllm-model-tests*.json). Takes a run ID provided by the user or by CI (numeric ID or run URL), finds every vLLM job (run_vllm_n150_tests, run_vllm_n300_tests, run_vllm_llmbox_tests, run_vllm_galaxy_wh_6u_tests, run_vllm_bhqb_tests, run_vllm_p150_tests), pulls each job log, and writes a plain-text .log file grouped by test file. For each captured generation it records the full pytest node id (test), the model id when one is detected, the device, and the generated output text the test printed (the `output:` / `prompt: ..., output: ...` lines), filtering out progress-bar noise. Use whenever the user or CI gives a run ID and wants a log of vLLM model outputs. Never default to the latest run; if no run ID is in the message, ask once.
allowed-tools: Read, Write, Glob, Grep, Bash(gh api *), Bash(gh run view *), Bash(gh run list *), Bash(python3 *), Bash(python *), Bash(mkdir -p *)
argument-hint: run-id-or-url [output-path]
---

# vLLM Nightly Model-Output Log

Given a tt-xla GitHub Actions run that includes vLLM jobs (scheduled nightly
**or** a manual **Run Test** with `vllm-model-tests-nightly.json` /
`vllm-model-tests.json`), produce a plain-text log file that lists
**every vLLM generation that ran**, grouped by test file, with these fields per block:

- **test** — the full pytest node id, e.g.
  `tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py::test_llama3_3b_generation`.
  This is the primary identifier for every output.
- **model** — the HF model id (e.g. `meta-llama/Llama-3.2-3B`), shown **only when
  one is actually detected** in the log (a real `org/Model` id in the node-id
  param or a vLLM `model='...'` config line). It is omitted when the test does
  not name a model — the node id already identifies the test, so we never guess.
- **device** — e.g. `n300`, `n300-llmbox`, `galaxy-wh-6u`
- **output** — the generated text the test printed (`output: ...`), plus the prompt when present

Blocks are **grouped by test `file_path`**: one `####`-banner section per file,
with an `outputs:` count, then all of that file's blocks underneath (across every
device). This lets you scan the run one test file at a time.

This skill is **vLLM-only**. It ignores every non-vLLM job in the run. It is a
sibling of `ci-benchmark-analyzer` but does a different job: it does not analyze
perf or diagnose failures — it captures what the models generated.

## Input handling

**The run ID must come from the caller** (interactive user **or** CI prompt).
Do not guess it and do not default to "latest".

| Input | Resolve to run ID |
|-------|-------------------|
| Numeric run ID | use directly |
| GitHub Actions URL | extract the `actions/runs/<ID>` segment |

An optional second argument is the output path (default `vllm-nightly-<RUN_ID>.log`
in the current directory). If the user/CI names a path (e.g.
`vllm-nightly-logs/vllm-nightly-<RUN_ID>.log`), write there. **Never ask for the
output path** — if it was not supplied, silently use the default.

**CI / non-interactive:** When the invoking message already includes a run ID
(e.g. GitHub Actions passes `${{ github.run_id }}`), use that ID immediately.
Do **not** ask for the run ID. Do **not** wait for confirmation. Proceed to the
extractor.

## What counts as a vLLM job

vLLM jobs come from
`.github/workflows/test-matrix-presets/vllm-model-tests-nightly.json` (and the
push variant `vllm-model-tests.json`). Their job names all match
`run_vllm_<device>_tests`, and the device token is embedded in the name:

| Job name | Device |
|----------|--------|
| `run_vllm_n150_tests` | n150 |
| `run_vllm_p150_tests` | p150 |
| `run_vllm_n300_tests` | n300 |
| `run_vllm_llmbox_tests` | n300-llmbox |
| `run_vllm_galaxy_wh_6u_tests` | galaxy-wh-6u |
| `run_vllm_bhqb_tests` | qb2-blackhole |

Match on the `run_vllm_..._tests` name prefix; parse the device token from the
name. Unknown tokens fall back to the raw token.

## How the generated text appears in logs

The tests print the generated text with one of these forms (see
`tests/integrations/vllm_plugin/generative/`):

```python
print(f"prompt: {prompts[0]}, output: {output_text}")   # -> prompt: ..., output: ...
print("output: ", output_text)                           # -> output:  ...
print(f"output: {output_text}")                          # -> output: ...
```

The **test** identity is taken from the pytest node id printed in the log
(`tests/.../test_file.py::test_name[params]`) and is tracked as the run walks the
log, so each captured output is attributed to the test that was running. The
**model** is filled in only when a genuine `org/Model` id appears — either in the
node-id parameter (e.g. `test_...[meta-llama/Llama-3.2-3B]`) or in a vLLM engine
config line (`model='...'`); bare params like `[n300]` or `[single_device]` are
device/config tokens, not models, so they are left as node-id-only (no `model:`
line). The device comes from the job name.

**Noise filtering.** vLLM's tqdm progress bars and throughput summaries also
contain the substring `output:` (e.g. `est. speed ... output: 6.80 toks/s`).
These are **not** generated text and are dropped — the script filters out any
`output:` line matching `toks/s`, `est. speed`, `it/s`, `Processed prompts`,
`Adding requests`, `EngineCore pid=`, or log-level markers. Only real generated
text survives.

## Workflow

1. **Preflight: confirm `gh` is authenticated FIRST.** As the very first action
   when this skill starts — before asking anything — run `gh auth status`. If it
   fails (not logged in / no token), stop and tell the user to authenticate. Do
   NOT ask for the run ID and do NOT run the extractor yet, since both need `gh`.
   In CI, `GH_TOKEN` is usually already set — if `gh auth status` succeeds,
   continue. Interactively, show these options (the `!` prefix runs the command
   in-session so the auth persists):

   - `! gh auth login` — interactive login, or
   - `! export GH_TOKEN=<your-token>` — if they have a token handy

   Once `gh` can reach the repo (default `tenstorrent/tt-xla`), continue to
   step 2.

2. **Resolve the run ID.**
   - If the invoking message already includes a numeric run ID or Actions URL
     (including CI prompts that pass `${{ github.run_id }}`), use it and skip
     asking. This is the normal path in GitHub Actions.
   - Otherwise, ask *exactly one* question as a **plain-text message** (do NOT
     use the `AskUserQuestion` tool — it cannot take a free-text run ID). Write:
     *"Please provide the run ID (numeric ID or the GitHub Actions run URL) you
     want the vLLM model-output log for."*
     Then stop and wait for the user's reply.
   - Ask nothing else. Do **not** ask about the output path, device filtering,
     format, or anything beyond the run ID — use the defaults for all of those.
   - Resolve the run ID: numeric ID used directly; a GitHub Actions URL →
     extract the `actions/runs/<ID>` segment.
   - Once you have the run ID, proceed through the remaining steps without any
     further questions or confirmations.
   - Never guess or default to the latest run.

3. Run the bundled extractor — it lists the vLLM jobs, downloads each job log,
   parses the test/model/device/output, filters progress-bar noise, groups by
   test file, and writes the `.log` at the requested path (the `.json` sidecar
   is written to `/tmp/<log-basename>.json`, not next to the log):

   ```bash
   python3 <skill-path>/scripts/extract_vllm_outputs.py <RUN_ID> --output <OUTPUT_PATH>
   ```

   The script prints the jobs it found to stderr and the entry count on
   completion. A job whose log cannot be fetched (HTTP 404 — logs expired or
   never uploaded) is **skipped, not fatal**: the script logs
   `(log unavailable — skipping this job)` and continues with the rest, then
   reports how many jobs were skipped.

4. Read the resulting `.log` and give the user a short summary: how many vLLM
   jobs were found, how many outputs were captured, how many jobs (if any) were
   skipped for unavailable logs, and call out any job whose `conclusion` was not
   `success` (those may have produced no `output:` line because they crashed
   before generation).

5. If **zero** outputs were captured but jobs exist, say so plainly — the tests
   may run under pytest stdout capture so `output:` lines only surface on
   failure or with `-s`. Point the user at the job URLs the script listed rather
   than inventing output.

## Output format

The `.log` file is grouped by test file. Each file gets a `####` banner with an
`outputs:` count, followed by one `====` block per captured output:

```
################################################################################
file_path: tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py
outputs:   6
################################################################################
================================================================================
test:   tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py::test_llama3_3b_generation
model:  meta-llama/Llama-3.2-3B
device: n300
prompt: I like taking walks in the
output: park near my house, especially in autumn when ...
job:    run_vllm_n300_tests  (https://github.com/tenstorrent/tt-xla/actions/runs/.../job/...)
================================================================================
...
```

- `test:` is always the **full** node id (path + `::test_name` + `[params]`) —
  never abbreviated.
- `model:` is present only when a real model id was detected; otherwise the
  block is identified by its node id alone.
- Files appear in first-seen order; blocks keep their in-file order, so the same
  test across n150/p150/n300 sits together within its file section.

The `.json` sidecar — written to `/tmp/<log-basename>.json` — holds the same
entries as a **flat list** (one object per output with `test`, `model`,
`device`, `prompt`, `output`, `job`, `job_url`), so downstream tools can regroup
or filter however they like.

## Notes

- `gh` must be authenticated with access to `tenstorrent/tt-xla`.
- In scheduled nightlies, `.github/workflows/schedule-nightly.yml` invokes this
  skill via Claude after `nightly_tests` and publishes the log to the
  **vLLM nightly model outputs** job summary (run ID = `${{ github.run_id }}`).
- Do not fabricate model output. Only report `output:` text actually present in
  the logs; if a model has no captured output, omit it (the script already does).
- Progress-bar / throughput lines (`toks/s`, `est. speed`, `it/s`, …) are not
  generated text and are filtered out — do not report them as model output.
- A test with no detectable model id is still captured; it is identified by its
  full node id with no `model:` line — this is expected, not a bug.
- A job whose log 404s is skipped (not fatal); mention any skipped jobs in the
  summary and point the user at those job URLs.
- The script never modifies the repo — it only reads run logs and writes the
  requested `.log`/`.json` files.
