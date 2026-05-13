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
| 7 | Build tt-xla (rebuilds tt-mlir from new pin transitively) | ✅ done (after 2 SFPI-mismatch attempts, settled on pin `085eaa8c3`) |
| 8 | Commit + push branch to remote | ✅ done (commit `bdb739a0b`, pushed to `origin/ssalice/cl-override-test`) |
| 9 | Trigger initial Performance Benchmark run | ✅ done (run `25771757720` on n150, queued 01:07 UTC) |
| 10 | Poll run every ~10 min via wakeup | in progress |
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
| n150 (sanity, WRONG ARCH) | [25771757720](https://github.com/tenstorrent/tt-xla/actions/runs/25771757720) | failed (no matching tests) | filter mismatch — not a real failure |
| n300-llmbox (sanity) | [25773764770](https://github.com/tenstorrent/tt-xla/actions/runs/25773764770) | **failed** (PCC -0.194 vs required 0.94) | **correctness regression — see analysis below** |
| n300-llmbox (final) | — | NOT triggered | blocked on fix |

**Note:** the `_tp` variant only runs on `n300-llmbox` in the matrix (user confirmed). n150 will be skipped entirely.

## Failure analysis (run 25773764770) — UPDATED 2026-05-13 after investigation

**Real bug found:** the aliasing is broken by `transfer_to_device` in `tests/benchmark/benchmarks/llm_benchmark.py:176-208`, not by JAX pytree as I first hypothesized.

```python
def transfer_to_device(input_args, device):
    for layer in input_args["past_key_values"].layers:
        ...
        layer.cumulative_length.zero_()
        layer.cumulative_length = layer.cumulative_length.to(device)  # ← per-layer .to(), no dedup
        layer.device = device
```

For 32 layers all aliased to a single CPU `shared_cl`, this calls `shared_cl.to(device)` 32 times. Each call creates a **distinct** device tensor. After the loop, the 32 layers each point to their own separate device tensor — aliasing destroyed.

Compare to the runner's `to_device` in `tests/infra/runners/torch_device_runner.py` (added by commit `bd26f9baa`) which **does** preserve aliasing via a `moved={id(x): result}` dict. That runner uses dedup specifically to handle StaticCache shared state correctly. But the benchmark path uses its own `transfer_to_device`, which never got the same treatment.

## Fix

Add dedup to `transfer_to_device` (small change to `llm_benchmark.py`). For each layer's `cumulative_length`, key on `id(...)`; only zero and `.to(device)` the first time, then re-alias all subsequent layers to the same already-moved tensor. No-op behavior for non-overridden caches (each layer has a distinct CL → no dedup hits).

## Original (incorrect) analysis below — kept for context

Test assertion: `AssertionError: First decode PCC failed. PCC=-0.194434, Required=0.94`

PCC of −0.19 means the decode output is essentially **uncorrelated** with the reference (i.e. wildly wrong, not just numerically slightly off). The override produces **wrong device output even though eager-mode parity passed locally**. The most likely root cause is **alias breakage across the JAX/torch-xla pytree boundary**:

- All 32 layers' `cumulative_length` start as aliases of one `shared_cl` tensor.
- During tracing/compilation, each `StaticLayer.cumulative_length` becomes its own pytree leaf → the compiled graph has **32 distinct cumulative_length inputs** and **32 distinct cumulative_length outputs**, even though at call time they all bind to the same Python tensor.
- On the way back out, the runtime writes each layer's returned `cumulative_length` to that layer's attribute slot. Layers 0–30 had `should_increment=False`, so they return their input unchanged (still 0). Layer 31 returns the incremented value (1).
- After the first decode step the aliasing is **broken**: layer 0 has cl=0, layer 31 has cl=1, the rest are 0. On the next step, each layer's reads see its own stale value → wrong `cache_position` → wrong attention KV slot → catastrophic divergence → PCC ≈ 0.

This is a fundamental mismatch between Python-side aliasing and how JAX/XLA-style tracers flatten the cache state. The eager-mode parity test missed it because eager simply mutates the in-memory tensor — there's no pytree round-trip to break the alias.

## Why my eager parity test missed it

Eager iterates layers in Python directly. The `mark_static_address` mechanism and the function-boundary input/output flattening that breaks aliasing only kick in under `torch.compile` / torch_xla tracing. My test would have caught this only if I'd run it through `torch.compile(...)` or actually exercised the compiled path — I didn't.

## Why this can't be patched with a small tweak

Possible "small fix" ideas all fail:
- **Have all 32 layers increment** by the same amount → defeats the whole point (32 redundant ops in graph).
- **Add a no-op `add_(0)` on layers 0–30** so all operations look identical → in-place mutation by `kv_length` on layer 31 still only happens once, the 31 no-ops still don't keep the per-output values in sync, so the alias-after-writeback issue persists.
- **`mark_static_address(shared_cl)`** would help if the issue were just dynamo not caching the address, but the deeper issue is pytree-decomposition of the cache state at the JAX boundary, which mark_static_address doesn't fix.

## What would actually fix it

Move `cumulative_length` ownership **out of `StaticLayer` and onto `StaticCache`** — so the cache has one `cumulative_length` tensor (not 32). Each `update()` reads from `cache.cumulative_length` (via a back-reference) instead of `self.cumulative_length`. Then the cache's pytree has exactly one CL leaf, one input, one output — no aliasing to break.

This is a deeper code change than the simple in-place layer swap. It would require:
1. Adding a `cumulative_length` tensor on `SharedCLStaticCache` (subclassed from `StaticCache`).
2. Subclassing `StaticLayer` so each layer holds a reference back to its parent cache and proxies `cumulative_length` access through that reference.
3. Modifying `override_cache_cumulative_length` to attach the back-reference.

Pytree flattening of the new cache would emit one CL input/output total, not 32 — and the alias-after-writeback bug disappears because there's nothing to alias.

## Status

Halted further CI triggers. Branch `ssalice/cl-override-test` still pushed at commit `bdb739a0b` with the broken override. Next step requires either:
- (a) Implementing the architectural fix above and re-pushing; or
- (b) User input on whether to proceed with the deeper refactor or shelve this approach in favor of the tt-mlir compiler pass alone.


## Notes / decisions log

- tt-mlir locally is on branch `ssalice/cl-cache-fix-attr-gated` (the PR #8277 branch). For this experiment we want tt-mlir's `main`, so we'll temporarily check out main in the local tt-mlir clone, build it, and reset tt-xla's pin to that SHA. After the experiment we can switch back without losing the PR branch (still safe on remote).
- "Verify the override works" locally = at minimum: (1) `python -c "from tt_torch.transformers_overrides import override_cache_cumulative_length"` succeeds, (2) running override against a freshly-created `StaticCache` produces a cache where all layers' `cumulative_length` are the same tensor object (`is` check), and (3) a few decode steps in eager mode (no tracing) produce outputs identical to baseline `StaticCache`.
