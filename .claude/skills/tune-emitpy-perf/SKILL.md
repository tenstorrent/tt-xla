---
name: tune-emitpy-perf
description: Iteratively reduce device time of a tt-xla codegen_py (EmitPy) decode `main.py` by profiling with tracy + tt-perf-report, patching the top op, re-checking PCC, and re-profiling. Use when the user wants to hand-tune the emitted TTNN python based on device perf evidence.
argument-hint: <export-dir> [--mesh-shape ROWSxCOLS]
---

# Tune emitpy decode perf — tt-xla

Combines [`setup-emitpy-from-benchmark`](../setup-emitpy-from-benchmark/SKILL.md)
(emit + parity + PCC harness) with [`capture-decode-step-device-perf`](../capture-decode-step-device-perf/SKILL.md)
(tracy + tt-perf-report). Read both first.

Goal: make `main.py` **faster** while holding PCC.

## Setup (once)

Run [`setup-emitpy-from-benchmark`](../setup-emitpy-from-benchmark/SKILL.md)
to get a parity-restored `main.py` with a PCC baseline and a device-time
baseline `summary.txt` (top ops + inline SLOW/BW/FLOPs hints). The ledger
and every subsequent iteration compare against that `summary.txt`. 

In `third_party/tt-mlir/.../third_party/tt-metal/` look for ops that 
could replace (fuse) groups of ops in the model, and in `tt-mlir` check for existing
fusing patterns that are expected to be matched in the model but failed.

Start `TUNING_LOG.md` where you will document these potential fusings and all future steps. 

Use full lines when logging, no need to break them to short lines.

Log initial pcc and device time. Every iteration must have a commit and a log entry. 

Make a branch where you will add each iteration as one commit. 
Revert with clean `git revert`.

Const eval functions are constant evaluation, they don't affect perf.

## Loop

One op, one knob per iteration. Each iteration starts from the previous
`summary.txt` (setup's for iteration 1, the prior iteration's re-profile
output otherwise).

1. **Patch.** Pick one target and apply one change. Ideas to try:
   - Fusing group of ops from the tt-metal table at the top of `TUNING_LOG.md`.
   - Commuting layout ops (move `permute`/`reshape` up or down to cancel
     an inverse pair).
   - Lowering data format: input/weight dtype, op `dtype=` (output only),
     or `fp32_dest_acc_en` (accumulator).
   - `tt-perf-report` advice columns.
   - Changing op knobs (block sizes, sharded variant, compute kernel
     config, math fidelity) - focus on ops that show higher impact in the perf report.
   - Leaving tensors in l1 memory
1. **Commit.** `git commit` the patch on the tuning branch before testing,
   so `git revert` is the clean undo path if PCC or DT regresses.
2. **PCC.** Revert immediately if it drops below baseline.
3. **Profile.** tracy → `tt-perf-report` scoped to the signposts. Keep if
   `Device Time Sum` dropped, revert otherwise. The new `summary.txt` is
   the input for the next iteration's step 1.
4. **Log.** Append one row to the `TUNING_LOG.md` ledger:
   `| # | patch | scope | tracy DT delta | PCC delta | decision | why |`.
   If there were some important discoveries, log them too. Commit log changes and summary png.

## Hang discipline

Bad knobs can wedge the model at 99% CPU with no log progress. Every run
must be kill-able:

- Background it (`block_until_ms: 0`).
- Budget ≈ 3× the last good run; on timeout `kill -KILL`.
- After **any** kill, `tt-smi -glx_reset_auto` (or `-r` for non-galaxy)
  before the next attempt.
- Rerun once after reset, and if it hangs again mark the experiment **failed**,
  revert the patch, move on. 

## Multi-step inputs + honest tracy measurement
     
Trace replays on whatever sits at the captured buffer addresses, so reusing capture-data as replay-data leaks artificially
warm caches into the number. For honest decode-step-K device time:

1. **Capture per-step inputs once.** Dump CPU snapshots from the benchmark for steps 2..N and materialize them as
   `tensors_step{K}/` (direct `ttnn.from_torch` + `ttnn.dump_tensor`). The codegen+dry_run path can't capture step-K naturally.
2. **Replay with disjoint data.** Warmup + trace-capture from step-1 tensors, replay from `tensors_step{K}`. Keep activation
   buffers alive across `end_trace_capture` so `ttnn.copy_host_to_device_tensor` can refill them in place. Wrap
   `ttnn.execute_trace` in dedicated `REPLAY_START`/`REPLAY_END` signposts.
3. **Scope the report to the replay.** A single run emits the inner decode signposts 3× (warmup, capture, replay); use the
   `REPLAY_*` markers to pick only the last one.
4. **Validate against multiple steps.** Re-check PCC against step-2 and step-3 goldens, not just step-1, so the patch isn't
   overfit to the capture data.

## Stop

When no patch moves the number without breaking PCC.

## Output

Table of top ops with `Device Time Sum` baseline → final, one line per
applied patch, list of codegen bugs filed during setup.
