# Automated `transformers` Uplift

The pipeline added in [#5412](https://github.com/tenstorrent/tt-xla/pull/5412) (commit `86efe02c`) automates the boring parts: it notices a new release, opens a work-in-progress branch, runs our test suites against it, and asks Claude to fix what broke. A human still reviews everything and opens the PR.

> **This is version 1.** It works end to end, but there is room to improve. If you spot a better heuristic, a missing suite, or a rough edge while using it, please fix it in the workflow rather than working around it locally, that's how this gets good.

---

## tl;dr

1. A nightly job (04:00 UTC) checks whether `transformers` has a newer stable release than the pin in `venv/requirements-dev.txt`.
2. If so, it creates `transformers-uplift/<version>`, bumps the pin, and dispatches the orchestrator on that branch.
3. The orchestrator runs tests in increasing order of cost. After each stage it hands the failures to Claude, which edits source in tt-xla and `tt_forge_models` and commits the result back to the branch.
4. When the cheap stages are green (or the iteration budget runs out) it runs the full model suite and the perf sweep.
5. You review the branch, run whatever extra suites you want by hand, and open the PR.

The pipeline never touches `main` — it refuses to commit to a protected branch. For more details the workflows and scripts that were added check [the PR](https://github.com/tenstorrent/tt-xla/pull/5412)

---

## Worked example: `transformers-uplift/5.5.2`

The scheduled job fired on the night of 2026-07-31 and produced [`transformers-uplift/5.5.2`](https://github.com/tenstorrent/tt-xla/tree/transformers-uplift/5.5.2) (run `30605994179`). Three commits, all by `github-actions[bot]`:

| Commit | Stage | Outcome |
| --- | --- | --- |
| `f86f78d03` | schedule | Pin bumped 5.5.1 → 5.5.2, transient artifacts gitignored |
| `564defadb` | full model suite | 7 failures, **0 fixed, 7 skipped** — all classified as tt-metal JIT build-cache races and a `Bad StatusOr` error, all matching the main baseline |
| `dfddc37d1` | perf sweep | 13 failures, **0 fixed, 13 skipped** — artifact-download timeouts, a device hang, a hugepages failure, a host OOM, two fabric topology-mapping errors, and three galaxy perf regressions (5.9%, 8.3%, 12.3%) |

`api-check` was green on the first attempt and the 32-model baseline passed completely, so `decide` went straight to finalize on iteration 1.

---

## Running the rest of the tests by hand

The automated pipeline covers api-check, the nightly-equivalent passing suite, and the perf sweep. That is not everything we need before landing an uplift. Two manual workflows fill the gap.

**Important:** dispatch them *from the uplift branch* (Actions → workflow → "Use workflow from branch" → `transformers-uplift/x.y.z`). Fixes are committed to whatever branch you dispatch from, and both workflows refuse to run against `main`. I forgot to add a check to those workflows to ensure the it's running on a uplift branch, please add if you have time.

### `Manual - Test + Transformers Uplift Fix`

| Input | Notes |
| --- | --- |
| `target_version` / `current_version` | e.g. `5.5.2` / `5.5.1`. Only used to give Claude the right changelog range — get them right or the diagnosis suffers. |
| `test_suite` | Pick a preset, or `Custom` and supply the matrix JSON in `test_suite_custom`. |
| `baseline_run_id` | A completed `schedule-nightly` run on `main`. Strongly recommended: without it, Claude sees *all* failures, including our pre-existing ones. Find one with `gh run list --workflow schedule-nightly.yml --branch main --limit 5`. |
| `tag` | Artifact and commit-message discriminator. Change it per run so artifacts don't collide. |
| `mlir_override` | Optional tt-mlir SHA, if you also need to move that pointer. |

### `Manual - Perf + Transformers Uplift Fix`

Same versions and `tag`, plus `runs-on-filter` (one runner pool, or `All`) and `test_filter` (case-insensitive substring, comma-separated — e.g. `resn,yol`). Use it to re-run a single benchmark after a fix instead of paying for the whole sweep.

### Coverage target: match the nightly

The bar for an uplift is simple — everything `schedule-nightly.yml` ("On nightly") runs should have run on the uplift branch. That's four test suites plus the perf benchmark. The pipeline covers two of them; you run the other three by hand.

**Automatic — the pipeline already did these**

| Nightly job | Suite | Notes |
| --- | --- | --- |
| `test_forge_models_passing` | `model-test-passing.json` | `test-and-fix-full` |
| `perf-benchmark` | perf benchmark sweep | `test-and-fix-perf`, same four runner pools (n150-perf, p150-perf, galaxy-wh-6u, qb2-blackhole). One difference: the uplift sweep excludes vLLM benchmarks, the nightly includes them. |

Plus the api-check collection sweep and the 32-model `baseline-uplift.json` smoke, which the nightly doesn't run at all.

**Manual — run these three yourself**

| Nightly job | Suite | What it adds |
| --- | --- | --- |
| `nightly_tests` | `basic-test-nightly.json` | Everything outside the forge-models runner: JAX single- and multi-chip tests, torch graph/op tests, the vLLM plugin integration tests, and examples — across wormhole_b0, n150, p150, n300, n300-llmbox, qb2-blackhole, galaxy-wh-6u and galaxy-bh. This is the widest hardware spread of the three. |
| `test_full_model` | `model-test-full.json` | The `model_test`-marked JAX and torch model tests that live outside `tt_forge_models`, including the `large` ones, on n150 / p150 / n300. |
| `test_forge_models_xfail` | `model-test-xfail.json` | The xfail, skipped and placeholder forge-models entries. Worth running because an uplift can change a test's *failure mode* — a new error hiding behind an expected failure — and because tests that start passing should be promoted. |

You don't need all of these green; many have known failures. What you need is confidence that nothing got *worse* because of the version bump, which is exactly what the `baseline_run_id` comparison gives you — so pass a nightly run id on each of these three dispatches.

---

## What to look at in a run

**Read the commit bodies first.** They are the fix summaries, and they are structured: `## Fixed`, `## Skipped (left for human review)`, `## Stats`. Five minutes there tells you more than the workflow logs.

**Then check for these specifically:**

- **Weakened checks.** Grep the diff for `required_pcc`, `assert_pcc` and `KNOWN_FAILURE_XFAIL`. The skill is allowed to loosen these, but giving up accuracy or coverage is always a human call.
- **The `Skipped` list.** Failures nobody fixed. Each needs a verdict from you: known flake, real bug worth an issue, or something Claude got wrong.
- **Fixes that spread.** Root causes get fixed everywhere they appear, so one insight can touch a dozen loaders — a bad pattern applied twelve times is twelve reviews.
- **`tt_forge_models` commits.** They land on a same-named branch there and are **force-pushed**; that branch has to merge before the submodule pointer can move here.
- **A red X is expected.** The run's conclusion tracks the underlying test jobs, so pre-existing flakes still show as failed. Judge by the fix summaries, not the badge. Several stacked runs means the loop iterated; stopping at iteration 5 with failures open means it hit the cap and never converged.
- **Warnings worth noticing.** A missing baseline (Claude then treats every failure as new) and `api.anthropic.com is NOT routed via tailscale0`. Neither fails the run.

---

## Landing the uplift

1. Review the branch as above; drop or rewrite anything you disagree with.
2. If `tt_forge_models` was patched, open and merge that PR first, then update the submodule pointer on the uplift branch.
3. Open the tt-xla PR. Because the diff touches the `transformers` pin, `call-inspect-changes.yml` automatically runs the `transformers-uplift-qualification` matrix (the fast baseline suite plus a couple of graph tests).
4. In the PR description, carry over the `Skipped` entries and file issues for the real ones — those are the known gaps you're landing with.
5. Make sure you work with perf team before we merge to ensure there are no regression in important tests.
