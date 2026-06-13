---
name: automate-ci-runs
description: Automate CI runs for tt-xla. Currently supports dispatching the "Run Test Single" workflow (`.github/workflows/manual-test-single.yml`) on `tenstorrent/tt-xla`. Use when the user wants to trigger a CI run, kick off a manual test workflow, or run pytest against a specific runner via GitHub Actions.
disable-model-invocation: false
allowed-tools: Read, Edit, Write, AskUserQuestion, Bash
context: fork
model: opus
---

# Automate CI Runs

Dispatches GitHub Actions workflows on `tenstorrent/tt-xla` from the conversation. Reuses the user's existing `gh` credentials — never asks for a PAT.

Supported workflows:
- **Run Test Single** (`.github/workflows/manual-test-single.yml`) — run pytest against a chosen path/runner.

## Step 1 — Verify gh auth

Before doing anything else, confirm the user is authenticated with `gh` against `github.com`. Run:

```bash
gh auth status -h github.com
```

- If it exits **zero** → auth is good, continue to the next step.
- If it exits **non-zero** → **stop the skill here**. Tell the user:

  > You're not logged in to `gh`. Run `gh auth login` in your terminal (use the `!` prefix in Claude Code to run it in this session), then re-invoke this skill.

  Do not prompt for a PAT and do not attempt to authenticate on the user's behalf — this skill reuses existing `gh` credentials only. Do not proceed to any later step until the user re-runs the skill with auth working.

## Step 2 — Collect run parameters from the user

Once auth is verified, ask the user for all three required parameters **in a single plain-text message** — a numbered list, no `AskUserQuestion` tool. The user replies with all three values in one message. (This mirrors the prompt style of the `/update-test-config-from-ci-runs` skill: list the items, explain each briefly, then wait for the user's combined reply.)

Print the message in this shape (adapt the wording, keep the structure):

> To dispatch the **Run Test Single** workflow, I need the following from you:
>
> 1. **`branch_name`** — the branch on `tenstorrent/tt-xla` to dispatch against (this is the `--ref` for `gh workflow run`). Your current local branch is `<output of git rev-parse --abbrev-ref HEAD>` — say `same` to use it, or type any branch that exists on the remote.
> 2. **`full_command_to_test`** — the full test identifier in the form `<prefix>-<pytest_nodeid>` (or just a bare pytest nodeid if no prefix applies). It is passed through to the workflow's `dir` input as-is. Examples:
>    - `torch-tests/runner/test_models.py::test_all_models_torch[coherlabs/pytorch-Coherelabs_aya_expanse_32b-tensor_parallel-inference]`
>    - `llm_torch-tests/runner/test_models.py::test_llms_torch[llama/causal_lm/pytorch-3.2_3B-llm_decode-seq_1-batch_1-single_device-mesh_default-inference]`
> 3. **`machine_name`** — the runner to execute on (`runs_on` input). Must be one of: `n150`, `n300`, `p150`, `n300-llmbox`, `llmbox-1`, `llmbox-2`, `galaxy-wh-6u`, `qb2-blackhole`.
>
> Please reply with all three values so I can continue. All three are required — I cannot proceed without them.

Then wait for the user's reply. Parse out the three values from whatever format they send (comma-separated, line-by-line, labeled, etc. — be flexible). Validate:

- `branch_name` is non-empty. If the user wrote `same` or `current`, substitute the current local branch.
- `full_command_to_test` is non-empty.
- `machine_name` is exactly one of the 8 allowed runners listed above. If not, re-ask just that item.

If any of the three are missing or invalid, re-ask only the missing/invalid item — do not re-prompt for values the user already gave correctly. Store the final answers as `branch_name`, `full_command_to_test`, and `machine_name` for use in later steps.

## Step 3 — Map collected parameters to workflow form fields

The skill collects three values in Step 2; each maps to exactly one field on the `Run Test Single` dispatch form (the same form shown at `github.com/tenstorrent/tt-xla/actions/workflows/manual-test-single.yml` → "Run workflow"):

| Step 2 variable        | Workflow UI field                 | `gh workflow run` argument          |
| ---------------------- | --------------------------------- | ----------------------------------- |
| `branch_name`          | **Use workflow from → Branch**    | `--ref <branch_name>`               |
| `full_command_to_test` | **Path passed to pytest** (`dir`) | `-f dir=<full_command_to_test>`     |
| `machine_name`         | **Choose runners for running test** (`runs_on`) | `-f runs_on=<machine_name>` |

Notes:
- All other form fields (`mark`, `contains`, `args`, `shared_runners`, `parallel_groups`, `mlir_override`, `forge_models_override`, `torchxla_build_run_id`, `artifact_sha`) are **not** collected in this version of the skill. The workflow requires `parallel_groups`, so the dispatch will pass `-f parallel_groups=1` as a fixed default. The rest are optional and omitted.
- `full_command_to_test` is passed through to `dir` as-is — no prefix stripping or splitting in this step. (If a prefix split is needed later, it will be added as a separate step.)
- Do not invent values for unmapped fields. If the user later wants to set `mark` / `contains` / etc., that's a future extension.

## Step 4 — Trigger the workflow run

Dispatch the workflow with `gh workflow run`, using the mapping from Step 3. Run exactly this command (substituting the Step 2 values, with proper shell quoting since `full_command_to_test` and `branch_name` may contain `[`, `]`, `::`, `/`, etc.):

```bash
gh workflow run manual-test-single.yml \
  --repo tenstorrent/tt-xla \
  --ref "<branch_name>" \
  -f dir="<full_command_to_test>" \
  -f runs_on="<machine_name>" \
  -f parallel_groups=1
```

Then:

1. **Check the exit code.** If `gh workflow run` exits non-zero, print the stderr verbatim and stop. Common causes: the branch doesn't exist on the remote, the user lacks `actions: write` permission, or the workflow file isn't on that branch. Do not retry automatically — surface the error to the user.

2. **Find the dispatched run.** `gh workflow run` does not return the run URL. After it succeeds, wait a couple of seconds for GitHub to register the run, then query:

   ```bash
   gh run list \
     --repo tenstorrent/tt-xla \
     --workflow=manual-test-single.yml \
     --branch "<branch_name>" \
     --user "@me" \
     --limit 1 \
     --json databaseId,url,status,createdAt
   ```

   Take the top entry — it should be the run you just dispatched (the `--user @me` filter scopes it to the current authenticated user, so concurrent dispatches by others won't collide). If the list is empty, wait a few more seconds and retry once; if still empty, tell the user the dispatch succeeded but the run hasn't appeared yet and point them to `https://github.com/tenstorrent/tt-xla/actions/workflows/manual-test-single.yml`.

3. **Report back to the user.** Print, in a short message:
   - The run URL (clickable)
   - The branch, `dir`, and `runs_on` that were dispatched (so the user can confirm the values are right)
   - That the run is now queued / in-progress on GitHub Actions

   Do not poll the run to completion in this version of the skill — dispatch + URL is the end of the flow. The user can watch it in the browser or via `gh run watch <id>` themselves.
