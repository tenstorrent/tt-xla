# Analyze a nightly and auto-bisect regressions (our categories only)

One command that, for a nightly CI run: collects failures, keeps only **our** test
categories, classifies each failure as **NEW** or **EXISTING**, **auto-bisects NEW**
regressions (locally for archs you have, or via CI for archs you don't), records
**since when** EXISTING ones have been failing, and flags **hard-to-bisect** cases
(timeouts; tests that pass standalone but fail only in the nightly group).

It is a thin wrapper: it drives the existing `/collect-failures`,
`/find-regression-boundaries`, and `/bisect-commit` skills plus the helper scripts
under `.claude/scripts/`, adding the category filter, new/existing branching, the
persistent ledger, and CI-based bisect for remote archs.

## Usage

```
/bisect-nightly <run_url_or_id> [local_archs="n150,n300"] [remote_strategy="emit"] [auto_dispatch=false]
```

**Arguments** (from `$ARGUMENTS`):
- `run_url_or_id` — nightly run URL or numeric run id.
- `local_archs` — comma list of archs available on THIS host (bisected locally). Default `n150,n300`.
- `remote_strategy` — for archs not in `local_archs`: `emit` (print the commit window + ready-to-run
  CI dispatch commands, default) | `fanout` (dispatch every commit in the window in parallel) |
  `bisect` (binary-search dispatch).
- `auto_dispatch` — `true` to actually run `remote_strategy` against CI; `false` (default) forces `emit`
  for remote archs even if a dispatch strategy was named (safe default — dispatching spins up many CI jobs).

## Categories

Scope is **exclude-based**: keep every failing pytest test **except** two areas.
OUT (dropped): vLLM/integration (`tests/integrations/**`) and perf benchmarks (`tests/benchmark/**`).
IN (everything else, labelled `model` / `op_by_op` / `op` / `graph` / `example` / `other`): PyTorch & JAX
model tests (runner `test_all_models_torch|_jax|test_llms_torch` + hand-written `tests/**/models/**`),
op tests (`.../ops/`), op-by-op (`test_all_models_op_by_op`), graph tests (`.../graphs/`), and examples.
PJRT C++ unit tests never appear (they are ctest/gtest, not in these pytest logs).

---

## Instructions for Claude

Determine the repo root and set `GITHUB_REPO=tenstorrent/tt-xla`:
```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
```
All artifacts live under `$REPO_ROOT/bisection/` (shared with the sub-skills). The ledger is
`$REPO_ROOT/bisection/bisected.json`.

### Phase 1 — Collect failures
Invoke skill `collect-failures` with the run URL/id (Skill tool). It writes
`bisection/run_<run_id>_failures.json`. Read that file; note `run_id`, `sha` (this nightly's head),
`workflow_id`, `run_date`.

### Phase 2 — Filter to our categories
```bash
python3 .claude/scripts/filter_categories.py "bisection/run_<run_id>_failures.json"
```
Produces `bisection/run_<run_id>_failures_filtered.json` (only in-scope `failed_tests`, each tagged with
`category`; timeouts left in `timed_out_jobs`; dropped ones under `out_of_scope_tests`). Report the
category counts to the user.

### Phase 2b — FIXED failures (regressions resolved since the previous nightly)

The current-run failure list alone can't show what got **fixed**. Diff the **previous** nightly's
in-scope failures against the current run's, so the analysis is complete (NEW / EXISTING / **FIXED**):

1. Collect + filter the previous nightly the same way (reuse cached logs where present):
   ```bash
   # <prev_run_id> = immediately-preceding "On nightly" run (Phase 3 already fetches this list)
   # (run /collect-failures on <prev_run_id> if not already collected, then:)
   python3 .claude/scripts/filter_categories.py "bisection/run_<prev_run_id>_failures.json"
   ```
2. **FIXED** = in-scope (test_id, machine_type) pairs failing in `<prev_run_id>` but **absent** from the
   current run's in-scope failures. **PERSISTING** = present in both (these are the EXISTING set).
   Report FIXED in the summary and, in the ledger, close out any matching `pending`/`bisected` entry
   (e.g. `--status` set to a note or delete) so a resolved regression stops being tracked.

Report the FIXED list to the user (test_id, arch, prev error).

### Phase 3 — Regression boundaries (new vs existing / since-when)

Two steps — pre-download logs, then run the boundary finder **directly** (do NOT shell out to
`claude -p`: the auto-mode permission classifier blocks a nested
`claude --dangerously-skip-permissions` subprocess, so `run_regression_batches.py`'s spawner won't run
inside a Claude session).

1. **Pre-download** the previous-nightly job logs into `bisection/logs/` (reuses the tested downloader):
   ```bash
   python3 .claude/scripts/run_regression_batches.py "bisection/run_<run_id>_failures_filtered.json" --download-only
   ```
2. **Invoke `find-regression-boundaries` directly via the Skill tool** on the filtered file
   (it launches its own parallel subagents — no nested Claude process):
   invoke skill `find-regression-boundaries` with args `bisection/run_<run_id>_failures_filtered.json`.

Then read the resulting `bisection/regression_report_<stem>.json`. Each result has `first_bad_*`,
`last_good_*`, and `last_good_status ∈ {PASSED, DIFFERENT_ERROR, NOT_FOUND}`.

> The `run_regression_batches.py` **full** mode (parallel `claude -p` batches) remains usable when YOU run
> it in a terminal outside auto-mode — it's faster for large (many-test) runs. Inside this skill, use the
> two-step direct path above.

### Phase 4 — Classify and act on each (test_id, machine_type)

For the **timeout bucket** (`timed_out_jobs` in the filtered file): for each, record in the ledger and
report as unbisectable — do NOT bisect (no reliable good/bad signal):
```bash
python3 .claude/scripts/ledger.py set --test "<last_test_executing-or-job>" --machine "<arch>" \
  --status timeout --first-bad-run-id <run_id> --notes "timed out; last executing: <last_test_executing>"
```

For each entry in the regression report:

1. **Ledger check** — skip work already done:
   ```bash
   python3 .claude/scripts/ledger.py get --test "<test_id>" --machine "<machine_type>"
   ```
   If it returns an entry with `status == "bisected"`, report the known `blame_sha`/`blame_pr` and
   **skip**. If `status == "pending"` and this is still EXISTING, just refresh it. Otherwise proceed.

2. **NEW regression** — `last_good_status == "PASSED"`. The last-good commit is the previous nightly's
   head: `last_good_sha`. Bad = `first_bad_sha`. Auto-bisect:
   - **Local arch** (`machine_type` ∈ `local_archs`): invoke skill `bisect-commit` with
     `test_id`, `first_bad_sha`, `last_good_sha`, `known_error` (from the report/failures).
     - If it reports **"REPRODUCED: NO (test passed on bad commit)"** → this is a **group-only** failure
       (passes standalone, fails only in the nightly group; likely test pollution/ordering/memory).
       Record `--status group_only` and stop bisecting it (optionally note that a whole-group repro is
       needed). Do not treat as a real code-bisect.
     - On success → record blame:
       ```bash
       python3 .claude/scripts/ledger.py set --test "<test_id>" --machine "<machine_type>" \
         --status bisected --blame-sha <BLAME> --blame-pr <PR_URL> \
         --first-bad-sha <first_bad_sha> --last-good-sha <last_good_sha> --known-error "<err>"
       ```
   - **Remote arch** (`machine_type` ∉ `local_archs`, e.g. p150 / n300-llmbox / galaxy-wh-6u /
     qb2-blackhole): use the CI dispatcher. If `auto_dispatch != true`, use `emit` (hand the user the
     window + commands); else use the chosen `remote_strategy`:
     ```bash
     python3 .claude/scripts/ci_bisect_dispatch.py <emit|fanout|bisect> \
       --test "<test_id>" --arch "<machine_type>" \
       --good "<last_good_sha>" --bad "<first_bad_sha>" --timeout 600
     ```
     For `fanout`/`bisect` it prints the BLAME commit + PR → record `--status bisected` as above.
     For `emit` → record `--status pending` with a note "remote arch: CI window emitted".
     - **PCC checkpoint-drift caveat:** if `known_error` is a PCC drop, first check whether the model's
       HF checkpoint changed around `run_date` (model "regressions" are often checkpoint drift, not a
       commit). If so, record `--status checkpoint_drift` and skip bisecting.
     - **tt-mlir drill-down (optional):** if the blame commit is a `TT_MLIR_VERSION` uplift, the tt-xla
       result names the uplift; drill into tt-mlir with `ci_bisect_dispatch.py` using
       `ensure_mlir_branch` (branches `bisect/mlir_<sha6>_<DDMM>`) — or note the tt-mlir SHA range for a
       follow-up.

3. **EXISTING regression** — `last_good_status == "DIFFERENT_ERROR"` or boundary not found
   (predates history). Do **not** auto-bisect. Report "failing since `first_bad_run_date`
   (`first_bad_sha`, run `first_bad_run_id`)" and mark pending if not already tracked:
   ```bash
   python3 .claude/scripts/ledger.py set --test "<test_id>" --machine "<machine_type>" \
     --status pending --first-bad-run-id <first_bad_run_id> --first-bad-sha <first_bad_sha> \
     --known-error "<err>" --notes "failing since <first_bad_run_date>; bisection not yet done"
   ```

### Phase 5 — Report

Print a Markdown summary. Group by outcome; one row per (test_id, machine_type):

```
# Nightly regression bisect — <workflow_name> run <run_id> (<run_date>)

In scope: <N> (model=<a>, op=<b>, op_by_op=<c>, graph=<d>, example=<e>, other=<f>) · dropped (vLLM+perf): <M>

## NEW → bisected
| test_id | arch | blame commit | PR |
|---|---|---|---|

## NEW → remote arch (CI window emitted / dispatched)
| test_id | arch | window (good..bad) | status |
|---|---|---|---|

## EXISTING → failing since (bisection pending)
| test_id | arch | failing since (date / run / sha) | error |
|---|---|---|---|

## FIXED since previous nightly (was failing, now passing)
| test_id | arch | prev error (now resolved) |
|---|---|---|

## Hard cases (not bisected)
| test_id / job | arch | reason (timeout / group_only / checkpoint_drift) |
|---|---|---|
```

The ledger `bisection/bisected.json` is updated in place, so re-running on the next nightly skips
already-bisected regressions and preserves pending ones.

---

## Constraints
- Read-only `gh` for analysis; the only state-changing actions are: pushing `bisect/*` branches +
  dispatching `manual-test-single.yml` (remote bisect, gated by `auto_dispatch`), and updating the ledger.
- `/bisect-commit` runs in the sibling `tt-xla_bisect` clone — the working repo is never checked out/reset.
- Remote CI dispatch spins up jobs on shared runners — default to `emit` unless the user opts into
  `auto_dispatch=true`.
- Honor the memories: generic `Error code: 13` is unwrapped by `/collect-failures`; PCC drops may be HF
  checkpoint drift (`checkpoint_drift`); tt-mlir is pinned via `TT_MLIR_VERSION` in `third_party/CMakeLists.txt`.

## Error handling
| Situation | Action |
|---|---|
| `/collect-failures` finds no in-scope failures after filtering | Report "no in-scope regressions"; stop |
| Regression report has boundary_found=false for a test | Treat as EXISTING/pending; note history limit |
| Local bisect can't reproduce on bad commit | Reclassify `group_only`; do not bisect |
| Remote dispatch: artifact/wheel expired for a commit | that probe SKIPs (125); narrower window may be needed |
| Ledger already has `bisected` for a test | Skip; report existing blame |
