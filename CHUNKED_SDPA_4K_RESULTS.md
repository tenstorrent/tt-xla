# Chunked SDPA users-factor: 4k mixed-batch validation

Status: **IN PROGRESS** (started 2026-08-06 ~04:25 UTC)

## What is under test

The tt-mlir chunked-SDPA users factor (tt-mlir PR #9142), exercised end to end
through vLLM at 4096 context with batches that mix multi-chunk and single-chunk
prompts.

## Exact build under test

| Component | Value |
|---|---|
| tt-mlir branch | `ssalice/chunked-sdpa-users-factor-ttxla-pin` |
| tt-mlir SHA | `1bbc2eca5ddf24b821f3ff1fb33d08ee7f335284` |
| tt-mlir base | `c91bfb2d45` (tt-xla's pin as of 2026-08-05) |
| tt-metal pin | `f1f4ff75579ebd7a69c7da52d45368f273026d85` |
| tt-xla branch | `ssalice/devstral-qwen-5893` |
| tt-xla base | `origin/main` @ `1864e78fa` + PR #5893 (5 commits) + #5799 |

### KNOWN SKEW — read before interpreting failures

tt-xla main pins tt-mlir `724d2ff7b4`; this run uses `1bbc2eca5d`, based on the
older `c91bfb2d45`. 14 tt-mlir commits separate them, **2 of which touch paths
that matter here**:

- `724d2ff7b4` #9078 — rewrites `RegisterCustomShardingRule.cpp`, the exact file
  the users factor lives in.
- `6c43aaa38f` #9091 — models TTIR cache ops as in-place operations, i.e. the
  paged-cache path chunked prefill runs through.

So this is a tt-mlir/tt-xla pairing that exists nowhere else. A **pass** is
meaningful (the users factor works end to end). A **failure** is ambiguous and
must not be attributed to the users factor without re-testing.

Cheap fix if needed: both revisions pin the **same tt-metal**
(`f1f4ff7557`), so rebasing the 6 fix commits onto `724d2ff7b4` costs a tt-mlir
recompile only, no tt-metal rebuild.

## Pre-flight checks (done)

- [x] tt-mlir lit suite on the freshly built binary: `ttmlir-opt` rc=0,
      `FileCheck` rc=0, 10/10 modules.
- [x] 2D-mesh module `ChunkedSDPAUserAndHeadSharding` produces
      `query 4x2x64x16 -> result 4x2x64x16`, both factors composing, no
      resharding collective. Confirms the fix is live in *this* build.
- [x] Mixed-batch stride math unit-checked at batch 2/4/8/16/32/64/128/256 —
      never degenerate (the batch-256-only `stride` in `test_dptp_qwen` divides
      to 0 at batch<32 and would have raised `ValueError`).

## Runs

Driver: `run_4k_mixed.sh` (runs in the container). Started 04:42:18 UTC.

| # | Model | Mesh | Batch | Ctx | Result | Wall | Log |
|---|---|---|---|---|---|---|---|
| 1 | Qwen3-32B | (8,4) | 32 | 4096 | **PASS** | 48 min | `logs/4k_mixed/qwen3-32b_4k_b32.log` |
| 2 | Devstral-2-123B | (4,8) | 8 | 4096 | **FAIL** (3/8 rows garbage) | 80 min | `logs/4k_mixed/devstral_4k_b8.log` |

## Headline

**The users factor works.** Both models compiled and ran chunked prefill on a 2D
DP+TP mesh with **zero** `Result shape must match query shape` errors — that is
the exact verifier failure tt-mlir #9142 fixes, and it is the thing that made
this configuration impossible before. Chunked prefill genuinely engaged in both
(`per-seq chunk 128`; qwen budget 4096, devstral 1024), so this is not a case
where chunking was compiled at warm-up and never executed.

### Run 1 — Qwen3-32B (8,4) b32 @ 4096: PASS

0 empty of 32, all rows coherent, `assert_output_coherent` passed on every row.
Multi-chunk rows correctly attend their ~356-token prefix:

```
[  0] MULTI  | ' that the city has always been careful to document its history...'
[  4] MULTI  | ' that the city has been building bridges for a long time...'
[  1] single | " quite nice. It's a bit chilly, but the sun is shining..."
[  3] single | ' "The Alchemist" by Paulo Coelho...'
```

The MULTI rows are on-topic for the archive prefix, which means the
cached-prefix chunked-SDPA path returned correct content, not just correct
shapes.

### Run 2 — Devstral-2-123B (4,8) b8 @ 4096: FAIL, but not where you'd expect

```
[  0] MULTI  | ' that the city has been building bridges for a very long time...'   OK
[  1] single | ' sunny, ", ", "1", "1", " "1", " " " "1 "1", ...'                    GARBAGE
[  2] single | ' summer 11 (11.111 |11 (1 |1 (1 (11 |11 |1 |1 |1'                    GARBAGE
[  3] single | ' the (1) the "s" and "s "1" "s1"1 "s " " "1" " " "1 "1'              GARBAGE
[  4] MULTI  | ' that the city has been building bridges for a very long time...'   OK
[  5] single | ' pizza. I like pizza because it is delicious...'                     OK
[  6] single | ' that I can sleep in. I usually wake up at 6:30 on weekdays...'      OK
[  7] single | ' be shaped by the development of new technologies...'                OK
```

Failed on `assert_output_coherent` (word ratio 0.086 < 0.35) at row 1.

**Both multi-chunk rows are correct; the corruption is in three consecutive
single-chunk rows (1,2,3).** That is the inverse of a chunked-SDPA defect — the
chunked path is the part that works. Rows 5,6,7 are also single-chunk and fine,
so it is not "all short rows".

## Devstral failure: what it is NOT, and the two confounds

Ruled out from the log:
- No `Result shape must match query shape` (0 occurrences).
- PR #5893's active-row offset guard never fired (no "active rows disagree").
- Not empty output — 0 empty of 8. The rows generate, they generate wrong.

Two confounds mean this **cannot yet be called a real bug**:

1. **Devices were never reset between the runs.** `tt-smi -glx_reset` exited
   **127 — `tt-smi: No such file or directory`** inside the container (tt-smi
   lives on the host, not in the image). So Devstral ran on a mesh left in
   whatever state a 48-minute Qwen run at a *different mesh shape* ((8,4) vs
   (4,8)) left it.
2. **The tt-mlir pin skew** documented above.

Next step is a re-run of Devstral alone with a device reset actually performed
from the host. That distinguishes stale device state from a genuine
mixed-batch row-corruption bug.

### Leading hypothesis if the retry reproduces: b1-prefill, not chunked SDPA

Both tests set `prefill_batch_threshold: 16` and `min_num_seqs: 1`. From
`scheduler/ascend_scheduler.py:128`:

```python
if self.prefill_batch_threshold > 0 and self.b1_min_num_seqs > 0:
    num_pending = len(self.waiting) + len(self.skipped_waiting)
    if num_pending <= self.prefill_batch_threshold:
        fresh_prefill_cap = self.b1_min_num_seqs      # = 1 -> serial prefill
```

The gate is on **pending request count**, not batch size. That splits the runs:

| Run | Prompts | Initial `num_pending` vs 16 | Fresh-prefill path |
|---|---|---|---|
| Qwen3-32B | 32 | 32 > 16 | batched prefill (until the queue drains below 16) |
| Devstral | 8 | **8 <= 16** | **b1 serial prefill from the first step** |

So the run that passed and the run that failed took *different prefill paths*,
and the failing one is the path chunked prefill interacts with least. Combined
with the corruption landing on single-chunk rows while both multi-chunk rows are
correct, the suspicion points at b1-prefill row handling rather than the chunked
SDPA op.

Cheap discriminator if the retry still fails: re-run Devstral at batch 32 (above
the threshold, same path as the passing Qwen run) or drop
`prefill_batch_threshold` to 4 at batch 8. If either comes back clean, the
chunked-SDPA users factor is exonerated and the bug is in b1-prefill.

Also ruled out as a cause: the `TT_FATAL: Chip N logical eth core ... connects to
a remote mmio device` lines appear identically in the **passing** Qwen log, so
they are benign discovery noise.

Both use `prefill_chunk_size=128`, greedy sampling, full depth (no
`num_hidden_layers` override), `experimental_kv_cache_dtype=bfp_bf8`,
`enable_trace=True`, `cpu_sampling=False`.

Batch composition: 25% multi-chunk (~356-token prefix, 3 prefill chunks each)
spread evenly so every DP replica gets some, 75% single-chunk. Both classes
asserted non-empty — a run where nothing chunked would look like a pass and
test nothing.

## Analysis

_Pending run completion._

## Environment fixes needed to get here

- **vllm was missing entirely** from the regenerated venv (CI installs it as a
  prebuilt `vllm_tt` wheel artifact). Installed `vllm==0.26.0` +
  `transformers==5.14.1` per `integrations/vllm_plugin/requirements-vllm-plugin.txt`.
  Dry-ran first: `torch` / `torch-xla` / `torchvision` / `torchaudio` are
  untouched; only unused nvidia wheels get added, plus transformers 5.5.1 ->
  5.14.1.
- **The host cannot run any of this.** `/usr/bin/python3.12` and
  `libpython3.12.so.1.0` do not exist on the host, so the venv and `ttmlir-opt`
  only work inside the `tt-xla-ird-ssalice` container. Everything runs via
  `docker exec`.
- `venv/activate` dereferences an unbound `LD_LIBRARY_PATH` on line 10, so any
  driver script using `set -u` dies silently the moment it sources it. The
  driver exports `LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"` and does not set
  `-u`.

## Notes / carried decisions

- Dropped from the old WIP branch, deliberately:
  - `_pin_users_to_batch_axis_hook` (tt-xla-side workaround) — this is exactly
    what the tt-mlir users factor replaces; carrying it would mask the thing
    being tested.
  - `_dp_inert_slots` DP-condense fix (our `#5778`) — superseded by `#5799` on
    the new base, which is the reviewed implementation.
  - Uncommitted `start_index` offset WIP — the new base has 33 `start_index`
    occurrences and PR #5893's `60283d725` decides the chunk offset from active
    rows, which covers it. **Preserved in `stash@{0}`** if it turns out not to be.
- `test_dptp_qwen` / `test_dptp_devstral` were left untouched: both pin
  `num_hidden_layers=2` and qwen has its coherence assert commented out, so they
  answer "does it compile" not "is the output right". The new
  `*_mixed_4k` tests run the real model and check the text.
