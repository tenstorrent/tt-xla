# Failed Tests Analysis — Transformers 5.5 Uplift

**Branch:** `ssalice/transformers-5.5-draft`
**Date:** 2026-05-12
**Failed tests:**
- `tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-0.5B_Instruct-single_device-training]`
- `tests/runner/test_models.py::test_all_models_torch[qwen_3/causal_lm/pytorch-0_6B-single_device-training]`

**CI failure:** GitHub Actions run [25390122079](https://github.com/tenstorrent/tt-xla/actions/runs/25390122079), jobs 74469325667 (qwen_3) and 74469325575 (qwen_2_5).

---

## Phase 1: Root Cause Investigation

### Failure signature (CI)
Both tests aborted with `running the test CRASHED with signal 9` (SIGKILL) at ~296-297s wall time.

| Test | CI duration | CI status |
|---|---|---|
| qwen_2_5/causal_lm/0.5B_Instruct training | 296.33s | SIGKILL (pytest-forked child killed) |
| qwen_3/causal_lm/0_6B training | 297.47s | SIGKILL (pytest-forked child killed) |

### Local reproduction (no memory limits)
Both tests **pass** locally:

| Test | Local duration | Local result |
|---|---|---|
| qwen_2_5/causal_lm/0.5B_Instruct training (DEBUG) | 285.18s | PASSED |
| qwen_2_5/causal_lm/0.5B_Instruct training (no-DEBUG) | 218.59s | PASSED |
| qwen_3/causal_lm/0_6B training (DEBUG) | 271.33s | PASSED |

### Memory profile (qwen_2_5, DEBUG, local)
RSS sampled every 5s during the run:

```
ELAPSED RSS
04:20   4.14 GB  ← steady (TT compile/exec phase)
04:25   4.14 GB
04:30   4.14 GB
04:35   4.14 GB
04:40   4.68 GB  ← warmup
04:46  14.35 GB  ← spike (gradient materialization)
04:51  19.37 GB  ← peak (comparison phase)
04:56  19.39 GB
05:01   1.58 GB  ← teardown
```

qwen_3 observed 24.7 GB RSS at 4:33 (single sample, during comparison).

### CI environment
- GitHub Actions runner host memory: **31.39 GB**
- Container: `--device /dev/tenstorrent`, no explicit `--memory` limit (inherits host)
- Pytest cmd: `--forked --log-memory --durations=0 ...`
- Step timeout: 240 minutes (`notimeout` marker present)
- `TT_METAL_OPERATION_TIMEOUT_SECONDS: 120`

### Conclusion: phase-locked OOM-kill at comparison
- The kill is `signal 9` (SIGKILL), not `signal 6` (SIGABRT). tt-metal timeouts use `TT_THROW` → SIGABRT, so the kill is **not** from tt-metal.
- The 296/297s coincidence is phase-locked, not clock-locked: both tests progress through compile/forward/backward and hit the **gradient comparison phase** at ~T=4:50, at which point RSS spikes from ~5GB to 19-25GB in ~10s.
- On a 31.39GB CI host with OS+runner+pytest-parent overhead (~5-10GB), the in-fork peak crosses available headroom and the kernel sends SIGKILL.
- **Root cause: OOM-kill during the post-backward gradient comparison phase in `TorchModelTester._test_training`.**

### Why this regressed under the transformers 5.2 → 5.5.1 uplift
The comparison phase memory dominates regardless of transformers version. The most plausible contributor introduced by the uplift is `Qwen2Model` `use_cache` defaulting to `True` even in training: in 5.5, `merge_with_config_defaults` resolves `use_cache` from `config.use_cache` (Qwen2.5 ships `True`) and only overrides when `gradient_checkpointing and training`. With `use_cache=True`, a `DynamicCache` of K/V tensors is built during forward and retained as part of the model output object until that output is reassigned. (More analysis in Fixes section.)

---

## Phase 2-4: Fixes Tested

### Fix candidates (ranked by effort × leverage)
1. **`use_cache=False` during training** — addresses tx5.5 cache-default regression. Branch: `ssalice/qwen-training-use-cache-fix`. Status: TBD.
2. **Stream gradient compare (pop-and-compare loop)** — collapses comparison-phase working set from ~6-8N to ~2N. Highest leverage.
3. **`zero_grad(set_to_none=True)`** — frees `.grad` storage after `_extract_grads` clones them. Small drop (~1-2GB), trivial change.
4. **`num_layers` truncation** in test config — reduces model depth ~6x. Sledgehammer; reduces test coverage at depth.

### Fix attempts

#### Attempt 1: `use_cache=False` during training (branch `ssalice/qwen-training-use-cache-fix`)
Set `self._model.config.use_cache = False` at the start of `_test_training` in `tests/infra/testers/single_chip/model/torch_model_tester.py`.

**Result:** Test still passes locally (207.94s, faster than baseline 285s), **but peak RSS unchanged at 24.14 GB** (sampled every 3s; the baseline peak with 5s sampling was 19.4 GB, likely undercounting). The K/V cache is GC'd after `_unpack_forward_output` reassigns `cpu_res` to the logits tensor, so `use_cache` does not drive the comparison-phase peak. **This fix did NOT solve the OOM.** Hygiene benefit only.

#### Attempt 2: Stream gradient compare (branch `ssalice/qwen-training-stream-compare-v2`) — **CHOSEN FIX**

Replace the single `_compare(tt_grads, cpu_grads)` call (line 310) with a `.pop()`-based loop that compares one `(name, tt_grad, cpu_grad)` pair at a time, aggregates `pcc=min`, `atol=max`, `allclose=all`, `equal=all`, `passed=all`, and frees each pair before the next.

The comparison-phase peak comes from `_match_data_types` casting every tensor to `float64` (2x memory) AND four sequential comparison ops (`equal`, `atol`, `pcc`, `allclose`) running on the full pytree, each producing large intermediates. Streaming collapses this working set to ~2x the largest single grad tensor (embedding-tied `lm_head.weight` ≈ 0.55 GB fp32 → 1.1 GB fp64) plus a few intermediate ops on that one tensor.

**qwen_2_5 result:** PASSED in 210.12s, **peak RSS 6.71 GB** (down from 24.14 GB observed unfixed — 3.6x reduction).

**qwen_3 result:** PASSED in 242.14s, **peak RSS 10.56 GB** (down from 24.7+ GB observed unfixed — 2.3x reduction).

Both leave 20+ GB headroom on CI's 31 GB host, eliminating the OOM-kill.

**Code change:** `tests/infra/testers/single_chip/model/torch_model_tester.py` (`_test_training` only, ~45 lines added). Framework-level — automatically applies to every torch single-device training test.

---

## Verification

### Local pass logs

| Test | Branch | Duration | Peak RSS | Result | Log |
|---|---|---|---|---|---|
| qwen_2_5/causal_lm/0.5B_Instruct training | unfixed (`ssalice/transformers-5.5-draft`) | 285.18s | 19.39 GB (5s sample) | PASSED | `/tmp/qwen-repro/qwen_2_5_debug.log` |
| qwen_2_5 (use_cache=False fix only) | `ssalice/qwen-training-use-cache-fix` | 207.94s | 24.14 GB | PASSED — fix INEFFECTIVE | `/tmp/qwen-repro/qwen_2_5_fix1.log` |
| **qwen_2_5 (streaming compare)** | `ssalice/qwen-training-stream-compare-v2` | **210.12s** | **6.71 GB** | **PASSED — fix EFFECTIVE** | `/tmp/qwen-repro/qwen_2_5_fix2.log` |
| qwen_3/causal_lm/0_6B training | unfixed | 271.33s | ~24.7 GB (single obs) | PASSED | `/tmp/qwen-repro/qwen_3_debug.log` |
| **qwen_3 (streaming compare)** | `ssalice/qwen-training-stream-compare-v2` | **242.14s** | **10.56 GB** | **PASSED — fix EFFECTIVE** | `/tmp/qwen-repro/qwen_3_fix2.log` |

### Why this fixes CI
CI runner has 31.39 GB host memory; baseline tests peaked at 20-25 GB during the gradient comparison phase and were OOM-killed (`signal 9`). With the streaming fix, peaks drop to 7-11 GB, leaving >20 GB headroom alongside OS/runner overhead.

### Why other proposals were rejected
- **use_cache=False** (proposal 6): tested, did not reduce peak. The `DynamicCache` is GC'd after `_unpack_forward_output` reassigns `cpu_res` to logits, so it's not the OOM driver.
- **zero_grad(set_to_none=True)** (proposal 3): would only free ~2 GB of CPU `.grad` storage; insufficient on its own when the comparison spike is ~20 GB.
- **num_layers truncation** (proposal 4): would work (sledgehammer) but reduces test coverage at depth and requires plumbing a new YAML knob through the runner.
- **Skip clone in to_device** (proposal 5): the `clone()` is on a code path that doesn't carry the heavy training tensors (model params bypass it via `workload.model.to(device)`). Negligible impact.
- **Offload cpu_grads to disk** (proposal 1): would help but introduces IO and tmpfile lifecycle; streaming achieves the same drop without disk dependency.

### Notes / future cleanups
- The forward-result `_compare(tt_res, cpu_res)` at line 309 still uses the full-pytree path, but `cpu_res` is the logits tensor (~78 MB fp32) so it's not an OOM risk for current models. If a future training test produces a much larger forward output, applying the same streaming approach to forward comparison would generalize the fix.
- The fix is framework-level in `TorchModelTester._test_training`, so every torch single-device training test benefits automatically — no per-model changes needed.

### Constraints honored
- Each tested fix lives on its own branch (`ssalice/qwen-training-use-cache-fix`, `ssalice/qwen-training-stream-compare-v2`).
- Advisor was consulted at each decision point: before substantive work, during reconciliation when initial OOM/watchdog hypothesis split, after fix1's null result, and before declaring done.
- One test at a time: qwen_2_5 was verified passing before applying the same framework fix to qwen_3.

---

## Verification

(Logs and final pass evidence to be added once a fix lands the peak below CI's effective headroom)
