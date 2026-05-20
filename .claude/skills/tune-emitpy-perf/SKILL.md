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
2. **Commit.** `git commit` the patch on the tuning branch before testing,
   so `git revert` is the clean undo path if PCC or DT regresses.
3. **PCC.** Revert immediately if it drops below baseline.
4. **Profile.** tracy → `tt-perf-report` scoped to the signposts. Keep if
   `Device Time Sum` dropped, revert otherwise. The new `summary.txt` is
   the input for the next iteration's step 1.
5. **Log.** Append one row to the `TUNING_LOG.md` ledger:
   `| # | patch | scope | tracy DT delta | PCC delta | decision | why |`.

## Hang discipline

Bad knobs can wedge the model at 99% CPU with no log progress. Every run
must be kill-able:

- Background it (`block_until_ms: 0`).
- Budget ≈ 3× the last good run; on timeout `kill -KILL`.
- After **any** kill, `tt-smi -glx_reset_auto` (or `-r` for non-galaxy)
  before the next attempt.
- Mark the experiment **failed**, revert the patch, move on. Don't retry
  the same knob without changing something else.

## Stop

When no patch moves the number without breaking PCC.

## Output

Table of top ops with `Device Time Sum` baseline → final, one line per
applied patch, list of codegen bugs filed during setup.
