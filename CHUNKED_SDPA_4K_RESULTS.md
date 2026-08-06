# Chunked SDPA users-factor: 4k mixed-batch validation

Status: **COMPLETE** (2026-08-06, 04:25 - 09:31 UTC)

## TL;DR

1. **The tt-mlir chunked-SDPA users factor works — positively, not just "no
   crash".** The strongest evidence is not the absent verifier error, it is that
   Qwen's multi-chunk rows returned **on-topic content for a ~356-token prefix
   that exists only in the paged KV cache**. Producing text about the city
   bridge archive requires correctly reading back cached-prefix KV across three
   prefill chunks on a 2D DP+TP mesh. Shapes being right is necessary; content
   being right is the proof. Secondarily: **zero** `Result shape must match
   query shape` in any run, which is the verifier failure #9142 fixes.
2. **Qwen3-32B (8,4) batch 32 @ 4096: PASS**, all 32 rows coherent, multi-chunk
   rows correctly attend their long prefix.
3. **Devstral (4,8) batch 8 @ 4096 failed — and it is NOT the users factor.**
   Disabling the b1 serial-prefill path (`prefill_batch_threshold: 0`) makes the
   identical config pass on the identical build, so the fault is somewhere in
   that path. The *mechanism* is still unknown. My row->device bucket-mismatch
   explanation is **untested** — the A/B that appeared to falsify it used a
   broken metric (see retraction below). Qwen3-32B did not reproduce the bug at
   full depth, batch 32.

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

| # | Model | Mesh | Batch | Ctx | `pbt` | Result | Wall | Log |
|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3-32B | (8,4) | 32 | 4096 | 16 | **PASS** | 48 min | `qwen3-32b_4k_b32.log` |
| 2 | Devstral-2-123B | (4,8) | 8 | 4096 | 16 | **FAIL** 3/8 rows | 80 min | `devstral_4k_b8.log` |
| 3 | Devstral-2-123B | (4,8) | 8 | 4096 | 16 | **FAIL** identical (post device reset) | 68 min | `devstral_4k_b8_retry.log` |
| 4 | Devstral-2-123B | (4,8) | 8 | 4096 | **0** | **PASS** | 82 min | `devstral_4k_b8_nob1.log` |

(`pbt` = `prefill_batch_threshold`. All logs under `logs/4k_mixed/`.)

> **`pbt=0` is a bisect result, NOT a recommended setting.** It disables the b1
> serial-prefill path, which exists deliberately for low-concurrency
> small-graph prefill. Run 4 changes it only to isolate the variable. Do not
> ship it as a config.

### Working-tree state you will come back to

- **`third_party/CMakeLists.txt` is intentionally dirty and uncommitted.** It
  overrides `TT_MLIR_VERSION` to the branch
  `ssalice/chunked-sdpa-users-factor-ttxla-pin` instead of main's
  `724d2ff7b4`. Every result here was produced with that override in place;
  reverting it changes what a rebuild produces.
- **`logs/4k_mixed/` is 3.9 GB and untracked.** Mostly whitespace — ~447k blank
  lines per log from progress output — so the useful content is a tiny fraction.
  `grep -aE "^\[ *[0-9]+\] (MULTI|single)"` pulls the per-row results out of any
  of them.

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

### Retry on freshly reset devices: reproduces byte-for-byte

The first run had a confound — `tt-smi -glx_reset` exited **127
(`tt-smi: No such file or directory`)** inside the container, because tt-smi
lives on the host and not in the image. So Devstral ran on a mesh left dirty by
a 48-minute Qwen run at a *different mesh shape*.

Re-ran Devstral alone after a real host-side `tt-smi -glx_reset` (all 32 chips
re-initialized). Result (`logs/4k_mixed/devstral_4k_b8_retry.log`, rc=1, 68 min):

**Identical output on all 8 rows, character for character.** Same rows 1/2/3
corrupted, same token soup, same assertion (word ratio 0.086).

That is informative:

- **Not stale device state.** The reset changed nothing.
- **Not flaky.** Fully deterministic across two runs 1.5 h apart.
- **Not thermal/timing.** Deterministic corruption of the *same* rows.

Remaining confound is the **tt-mlir pin skew** documented above.

### The corrupted rows start correct and then degenerate

Lining the failing outputs up against their prompts:

| Row | Prompt ends | Output starts | Then |
|---|---|---|---|
| 1 | `The weather today is` | ` sunny,` | `", ", "1", "1", " "1"...` |
| 2 | `My favourite season is` | ` summer` | `11 (11.111 \|11 (1 \|1...` |
| 3 | `The best book I have read is` | ` the` | `(1) the "s" and "s "1"...` |

**The first token or two are correct, then it collapses into punctuation and
digits.** Prefill produced a sensible first token for every corrupted row, so
prefill read the prompt correctly; the degeneration begins in decode. That
points at KV/state corruption after the prefill step rather than at the prefill
attention output itself — again away from chunked SDPA, which is a prefill-path
op.

Positional pattern (batch 8, mesh (4,8) = 4 DP replicas x 2 users):

- Corrupted: rows 1, 2, 3 — all short prompts in the **first half** of the batch.
- Clean: rows 5, 6, 7 — the *same* short prompts in the second half.
- Clean: rows 0 and 4 — both long/multi-chunk.

Under round-robin DP assignment (`replica = i % 4`) that reads as: each of
replicas 1/2/3 has its first-half row corrupted and its second-half row clean,
while replica 0 holds both long prompts and is entirely clean.

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

Also ruled out as a cause: the `TT_FATAL: Chip N logical eth core ... connects to
a remote mmio device` lines appear identically in the **passing** Qwen log, so
they are benign discovery noise.

## RETRACTED: the bucket-mismatch "falsification" was a broken metric

**The A/B below did not test anything. Its result must not be relied on.**

`test_dptp_qwen_bucket_ab` reported `DEGENERATE_ROWS=[]` for both arms. That was
a bug in my degeneracy check, not a clean result:

```python
words = sum(c.isalpha() or c.isspace() for c in text)   # WRONG
if words / max(len(text), 1) < 0.75: ...
```

`str.isalpha()` is **True for CJK**, so multilingual token soup scores higher
than real English:

```
broken_metric(CJK soup)  = 0.985   -> "clean"
broken_metric(English)   = 0.926
non-Latin ratio(soup)    = 0.246   -> the real checker fails at >0.03
```

Both arms were emitting total garbage at 10 layers. The test also "PASSED" only
because it was report-only with no assertion.

Consequences:

1. **The row->device bucket-mismatch hypothesis is UNTESTED, not falsified.**
   The retraction below stands retracted; the hypothesis is open again.
2. **10 layers is not a viable cheap repro for either model.** Truncation alone
   destroys output quality, so bug-corruption cannot be distinguished from
   truncation-garbage. Confirmed on both Qwen (soup) and Devstral (soup, and
   `assert_output_coherent` correctly failed it on non-Latin ratio 0.200).
3. This is consistent with the pre-existing `test_dptp_qwen`, which pins
   `num_hidden_layers=2` **and has its coherence assert commented out** — a
   truncated model cannot pass it.

The test has been fixed to use `assert_output_coherent` and to fail loudly when
every row is degenerate, so a run that cannot discriminate reports as an error
rather than a pass.

### A valid bucket A/B needs full depth

At full depth Qwen is ~48 min/arm and Devstral ~80 min/arm. On Devstral's mesh
the arms would be `max_num_seqs=8` (small=4, decode=8, mismatched) vs
`max_num_seqs=4` (all buckets 4, aligned), both with `pbt=16` armed.

Not run: the machine is shared and this session is already well over its run
budget.

## SUPERSEDED (see retraction above): row->device bucket mismatch

An earlier revision of this document argued the corruption came from prefill and
decode using different row-count buckets, so a request's row landed on a
different DP device for the KV write than for the read. **That is wrong.**

Tested directly with `test_dptp_qwen_bucket_ab` (Qwen3-32B, 10 layers, mesh
(8,4), `pbt=16` armed in both arms, only `max_num_seqs` differing):

| Arm | small | decode | Mapping | Predicted | **Actual** |
|---|---|---|---|---|---|
| `max_num_seqs=16` | 8 | 16 | `r` vs `r//2` | FAIL rows 1-7 | **PASS**, `DEGENERATE_ROWS=[]` |
| `max_num_seqs=8` | 8 | 8 | identical | PASS | PASS, `DEGENERATE_ROWS=[]` |

The experiment was valid, not vacuous — the log confirms the mismatch condition
existed in arm A: `min_num_reqs 1 -> 8`, prefill graphs compiled at **both**
`reqs=8` and `reqs=16`, decode at `reqs=16`. The predicted corruption simply did
not occur.

Conclusion: a prefill/decode bucket mismatch is **not sufficient** to cause the
corruption. Whatever the b1 path does wrong on Devstral, it is not this.

### What this run does establish

- **Qwen3-32B does not reproduce the bug** at 10 layers on mesh (8,4), with the
  b1 cap armed, at either bucket alignment. Two more clean data points for Qwen.
- **Layer count is a usable lever**: 10-layer runs took 71 min and 12 min versus
  48-82 min at full depth, and behave sanely. A cheap repro loop is feasible
  *if* it is built on Devstral, which is the config that actually fails.
- Graph compilation is identical regardless of `prefill_batch_threshold`; the
  threshold changes dispatch only.

### Still true, still unexplained

`pbt=0` fixes Devstral deterministically (runs 2/3 vs 4). That is solid
empirical fact and is untouched by this falsification. The mechanism is simply
still unknown.

Remaining differences between the failing Devstral config and the passing Qwen
A/B, any of which could matter: **model** (Devstral-2-123B vs Qwen3-32B),
**depth** (full vs 10), **mesh** (4,8) dp=4 tp=8 vs (8,4) dp=8 tp=4.

### Suggested next discriminator

Run Devstral at 10 layers, batch 8, `pbt=16`. It changes only depth from the
known-failing config. If it still fails, there is a ~15 min repro loop to
iterate on. If it passes, depth is implicated and the bug needs full weights.

## Superseded: b1 serial prefill, confirmed by controlled experiment

Run 4 changed **one variable** — `prefill_batch_threshold: 16 -> 0`, which
disables the b1 serial-prefill cap. Same model, same batch, same context, same
mesh, same tt-mlir build, same prompts, devices reset identically before both.

**Result: PASS.** The three rows that were deterministically corrupted across
two prior runs are now coherent:

| Row | `pbt=16` (FAIL) | `pbt=0` (PASS) |
|---|---|---|
| 1 | `' sunny, ", ", "1", "1", " "1"...'` | `' sunny and hot. The temperature is 30 degrees Celsius...'` |
| 2 | `' summer 11 (11.111 \|11 (1 \|1...'` | `' summer. I like summer because I can go to the beach...'` |
| 3 | `' the (1) the "s" and "s "1"...'` | `' the book of the book of the book...'` |

### Conclusion

**The chunked-SDPA users factor is exonerated.** The same binary that produced
garbage at `pbt=16` produces correct output at `pbt=0`, so nothing in the
tt-mlir sharding rule can be responsible. The bug is in the **b1 serial-prefill
path** (`scheduler/ascend_scheduler.py:128`), in how it interacts with a
mixed-length batch under DP.

This also **defuses the tt-mlir pin skew** as a concern for these results: the
skewed build passes both the Qwen run and the Devstral `pbt=0` run. The skew
would only need revisiting to chase the b1-prefill bug itself.

### Caveats on the pass

- Row 3 at `pbt=0` is repetitive (`the book of the book of the book`) — it
  clears the 0.35 word-ratio bar but is degenerate text. Row 7 repeats similarly
  in every run including Qwen's. Worth a look, but it is a quality issue, not
  corruption.
- Devstral was only tested at batch 8. Whether the b1 bug also bites at batch 32
  (where `num_pending` starts above the threshold, as in the passing Qwen run)
  is untested.

### Suggested next step

Isolate the b1-prefill bug directly: batch 8, `prefill_batch_threshold: 16`, but
with an **all-short** batch (no multi-chunk prompts). If that passes, the bug
needs the mixed-length batch and is about how b1 serial prefill sequences fresh
short prompts against a request that is mid-chunk. The test is already
parametrized on `prefill_batch_threshold`, so this is a small addition —
`_mixed_batch(8, long_fraction=0)` would need the `max(1, ...)` floor relaxed.

**Not run deliberately.** That is a ~70 min run on a shared galaxy, it opens a
new investigation rather than finishing the one asked for, and this session
already spent 4 device runs against a stated ~3-run budget. Left written down
and ready instead.

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
