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
Run pcc and perf inside docker container, and reset on host when needed. 

Make a branch where you will add each iteration as one commit. 
Revert with clean `git revert`.

Const eval functions are constant evaluation, they don't affect perf.

## Loop

One op, one knob per iteration. Each iteration starts from the previous
`summary.txt` (setup's for iteration 1, the prior iteration's re-profile
output otherwise).

## Loop

Each iteration starts from the previous profile (`summary.txt`):

1. **Patch.** Ideas to try:
   - Fuse group of ops into one op found in tt-metal.
     Start from most beneficial ones, like `moe_compute` and `sdpa` ops.
     Take your time trying to make them work in more iterations if needed,
     abandon only when you are absolutely sure it can't be done. 
   - Minimize TM (tensor manipulation) ops - move to cancel an inverse pair, 
     fold same ops, change chains of tm ops...
   - Check type conversions - remove redundant ones, use lower type where 
     it is not affecting global percision.
   - Check tile-alignment - prefer input shapes to be closer to full tiles.
   - Minimize CCL ops - minimize number of ccl ops, modify tm ops
     around it or change gather dim to minimize ccl op traffic. 
     Note that ccls always work with full tiles.
     Use ops that fuse ccls with other ops.
   - Check `tt-perf-report` advice columns.
   - Change op knobs (block sizes, sharded variant, compute kernel
     config, math fidelity) - focus on ops that show higher impact in the perf report.
   - Use l1 sharding - choose good core grid and make l1 chains of ops.
     Start with this only after you finish all other optimizations.

   For smaller changes that affect perf locally, you can add more at once
   and see from device perf how much each contributed.

   After larger changes, check for dead code, duplicated code, and 
   const eval.

   When you think you are finished, check latest perf report again
   and think of further optimizations.

2. **PCC.** Check pcc against golden and against baseline. If run fails, 
   try fixing it. If the change lowers pcc or it can't be fixed to run,
   revert it.

3. **Commit.** Once PCC holds, `git commit` the patch (the best working
   version you have) on the tuning branch. 

4. **Profile.**  Get ops perf csv with `run -t` and then run 
  `tt-perf-report` scoped to the signposts.
 
   Keep if `Device Time Sum` dropped; otherwise `git revert` the commit
   (don't discard — keep the attempt + revert pair in history). 
   The new summary files are inputs for the next iteration's step 1. 

   Extrapolate full model device time based on device time for labeled parts
   to understand how much each change actually affects full model perf.

5. **Log.** Append one row to the `TUNING_LOG.md` ledger:
   `| # | patch | scope | tracy DT delta | PCC delta | decision | why |`.

## Hang discipline

Bad knobs can wedge the model at 99% CPU with no log progress. Every run
must be kill-able:

- Background it (`block_until_ms: 0`).
- Budget ≈ 3× the last good run; on timeout `kill -KILL`.
- After **any** kill, `tt-smi -glx_reset_auto` (or `-r` for non-galaxy)
  on host before the next attempt. 
- Rerun once after reset, and if it hangs again mark the experiment **failed**,
  revert the patch, move on. 

## Stop

After 30 iterations, or if no patch moves the number without breaking PCC.

## Output

Optimized py code. As articafts add summary of most significant changes, 
per op perf improvement and list of blockers for failed fusings.
