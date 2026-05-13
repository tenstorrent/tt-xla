# tt-xla SharedCLStaticCache override task

Started: 2026-05-13. Owner: ssalice (via Claude assistant).

## Goal

Validate whether a Python-level fix in tt-xla (sharing `cumulative_length` across all `StaticLayer`s and only incrementing it once per step) eliminates the per-layer redundant updates that caused the transformers 5.5 decode-throughput regression on LLM benchmarks. The compiler-side fix (tt-mlir PR #8277) is intentionally NOT included on this branch — we pin tt-mlir to **main** to isolate the override's effect.

## Branch plan

- **tt-xla branch**: `ssalice/cl-override-test` (branched off `ssalice/transformers-5.5-draft`)
- **tt-mlir version**: `085eaa8c3` — most recent tt-mlir main commit whose pinned tt-metal (`78197e564f3`) uses SFPI 7.45.0 (matches local system). Earlier attempts: `73f2e685d54f` needed SFPI 7.47.0, `f8d3bf0e9` needed 7.44.0. This is a tt-mlir main commit, predates user's PR, so does NOT contain the consolidate pass — isolates override effect as intended.

## Step status

| # | Step | Status |
|---|---|---|
| 1 | Save memory + create this task.md | ✅ done |
| 2 | Create branch `ssalice/cl-override-test` off `ssalice/transformers-5.5-draft` | ✅ done |
| 3 | Update `third_party/CMakeLists.txt` tt-mlir pin to `73f2e685d54f` (latest main) | ✅ done |
| 4 | Add `TTSharedCLStaticLayer` + `override_cache_cumulative_length` to `python_package/tt_torch/transformers_overrides.py` | ✅ done |
| 5 | Wire override into `tests/benchmark/llm_utils/decode_utils.py` `init_static_cache` | ✅ done |
| 6 | Local eager parity check (override produces 1 shared CL, K/V outputs match baseline bit-for-bit over prefill + 3 decode steps) | ✅ done |
| 7 | Build tt-xla (rebuilds tt-mlir from new pin transitively) | in progress (background) |
| 8 | Commit + push branch to remote | pending (after build) |
| 9 | Trigger initial Performance Benchmark run via `gh workflow run manual-benchmark.yml` with `test_filter=llama_3_1_8b_instruct_tp`, `runs-on-filter=n150`, `sh-runner=false` | pending |
| 10 | Poll run every ~10 min via wakeup | pending |
| 11 | If passes: trigger n150 + n300-llmbox runs in parallel (same filter, sh-runner=false) | pending |
| 12 | Final report back to user with results / regression numbers | pending |

## CI workflow info

- Workflow file: `.github/workflows/manual-benchmark.yml` (named "Performance Benchmark")
- Trigger via: `gh workflow run manual-benchmark.yml --ref ssalice/cl-override-test -f test_filter=llama_3_1_8b_instruct_tp -f runs-on-filter=<arch> -f sh-runner=false`
- Architectures to test: `n150`, `n300-llmbox`

## Live run IDs

(will be filled in as runs are triggered)

| Arch | Run ID | Status | Result |
|---|---|---|---|
| n150 (sanity) | — | not started | — |
| n150 (final) | — | not started | — |
| n300-llmbox | — | not started | — |

## Notes / decisions log

- tt-mlir locally is on branch `ssalice/cl-cache-fix-attr-gated` (the PR #8277 branch). For this experiment we want tt-mlir's `main`, so we'll temporarily check out main in the local tt-mlir clone, build it, and reset tt-xla's pin to that SHA. After the experiment we can switch back without losing the PR branch (still safe on remote).
- "Verify the override works" locally = at minimum: (1) `python -c "from tt_torch.transformers_overrides import override_cache_cumulative_length"` succeeds, (2) running override against a freshly-created `StaticCache` produces a cache where all layers' `cumulative_length` are the same tensor object (`is` check), and (3) a few decode steps in eager mode (no tracing) produce outputs identical to baseline `StaticCache`.
