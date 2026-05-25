# `ttnn` trace API + tracy + per-op perf measurement on codegen `_main`

Reference for the workflow we settled on after the May 25 deep-dive. Three
companion documents:

* **[`MULTI_TOKEN_CAPTURE.md`](MULTI_TOKEN_CAPTURE.md)** — how to capture
  multi-step decode inputs via direct `ttnn.from_torch` + `ttnn.dump_tensor`
  (Path B). Read that first if you don't yet have `tensors_step{2,3,...}/`.
* **[`MULTI_TOKEN_CAPTURE.md`](MULTI_TOKEN_CAPTURE.md) "What a proper
  upstream fix would look like" section** — the issue we'd file against
  tt-xla / tt-mlir to make multi-step capture a first-class codegen feature.
* **[`TUNING_LOG.md`](TUNING_LOG.md)** — per-iteration ledger using the
  device-time numbers produced by the methodology below.

## Two things, easy to conflate

| | "trace" | "tracy" |
|--|--|--|
| What | tt-metal device-side replay buffer of kernel launches. APIs: `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.execute_trace` / `ttnn.release_trace`. | Tenstorrent's perf profiler (host-side recording of device kernel start/end cycles into per-op CSV rows). Invoked via `python -m tracy ... script`. |
| Purpose | Eliminate host-side launch overhead by batch-issuing recorded ops from a device buffer. | Measure where device time goes per op (Device FW / Kernel Duration, FLOPs, BW, kernel hints). |
| Default in `_main`? | No. `main.py:main()` runs `_main` once and exits unless `TTXLA_USE_TRACE=1`. | Off. Enabled by `./run -t` which invokes `python -m tracy ... main`. |

They compose, but the interaction is non-trivial — see "Tracy + trace
interaction" below.

## The three device passes in the trace pattern

`pcc.py --trace` and `main.py`'s `TTXLA_USE_TRACE=1` branch (and tt-metal's
own `tt_transformers/tt/generator.py:240-273`) all share this structure:

```
1. WARMUP        — single _main call. Cold-compile + populate ce_cache.
                   Per-op device events are emitted to tracy.
2. CAPTURE       — single _main call inside begin_trace_capture / end_trace_capture.
                   Ops execute AND are recorded into a device-side replay buffer.
                   Per-op events ARE emitted to tracy here too (once the buffer
                   issue below is fixed).
3. REPLAY        — ttnn.execute_trace(blocking=True). The device-side buffer
                   replays the captured ops as one batched dispatch from the
                   host's POV. Per-op tracy events are NOT emitted in this pass.
```

Between (2) and (3), if you want the replay to compute on *different* data
than the capture saw (the tt-metal "honest measurement" pattern), you refill
the captured input buffers in place — see "In-place refill" below.

## Profiler buffer overflow — the silent gotcha

Default `DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT = 1000`
(`tt_metal/impl/profiler/profiler_state_manager.cpp:19`). Per-op events are
written into a DRAM ring buffer per device that's sized off this constant.
For a single decode step of this graph that's ~370 unique ops; one decode
pass + one capture pass blows past 1000 and the second pass's events are
dropped silently.

Symptom (before the fix): warmup decode_1 had populated `DEVICE FW DURATION`
columns; trace capture's decode_1 had empty columns; `tt-perf-report` summed
the second pass to **0 μs**.

Fix:

```bash
python -m tracy ... --op-support-count 10000 ...
```

Sets `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000`. 10× headroom is more
than enough for a 3-pass run; bump higher if you're tracing models with
≥3k unique ops per pass.

Verified directly: with the bump, both warmup and capture passes have full
per-op rows populated (`E49_trace_bigbuf` artifact, 23,758 rows in the
capture window with 11,872 of them carrying populated FW/Kernel durations
across 32 devices).

## Tracy + trace interaction — what gets captured where

Even with `--op-support-count` bumped, **`execute_trace` does not emit
per-op rows** to `ops_perf_results_*.csv`. Verified by counting rows with
HOST_START_TS between the `REPLAY_START` / `REPLAY_END` signposts: **zero**.

The signposts themselves DO appear (tracy.signpost is a host-side write),
so you get the replay's wall-clock window for free. The per-kernel device
counters from the replay end up in `profile_log_device.csv` (raw cycles, no
op-level aggregation) — they're not joined back to ops in the
`ops_perf_results_*.csv` pipeline.

To get per-op device times for the replay specifically you have three
practical options:

| Option | Mechanism | Trade-off |
|--|--|--|
| **A** | Use the warmup pass's per-op times as a proxy. The replay reissues the same kernels with identical inputs to the device, so per-op kernel duration is essentially constant. | Doesn't reflect any host-launch elimination benefit; same number as a non-trace `./run -t`. Adequate for tuning comparisons since you're comparing per-op kernel time. |
| **B** | `REPLAY_START` / `REPLAY_END` host-timestamp signposts → wall-clock replay duration. | One number (e.g. ~27 ms for this graph), no per-op breakdown. Cleanest "how long did the trace replay take" measurement. |
| **C** | Programmatic API. After `execute_trace`: `ttnn.ReadDeviceProfiler(device)` + `ttnn.get_latest_programs_perf_data()`. Requires env vars `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1` (and incompatible with `--device-trace-profiler`). See `models/demos/deepseek_v3_d_p/tests/didt/sweep_deepseek_v3_matmul_tune.py:_get_tracy_timing_and_cores` for the canonical usage. | Returns a dict of `ProgramAnalysisData` per chip; you bypass `tt-perf-report` and process the dict directly. Best when iterating over many configs and producing a CSV row per config. |

Of these, **(A)** is what the tuning loop uses today; **(B)** is appended
as a wall-clock sanity check; **(C)** is unused so far in this branch but is
the right path if we ever want per-op replay numbers for a comparison table.

## `_main` deallocate-removal patch

For the in-place refill pattern (next section) to work, the trace's
input buffers must outlive `end_trace_capture` so `ttnn.copy_host_to_device_tensor`
has somewhere to write into. Originally `_main` did:

```python
ttnn.deallocate(args_0, False)   # main.py:3625 — input_ids
ttnn.deallocate(args_1, False)   # main.py:3928 — cache_position
```

Both are now commented out (see the "Path B" notes in `main.py`). The 6
KV-cache args (`activations[2..7]`) and the small `arg49 / arg50` activations
already survived `_main`; only the two `INT32` scalar-ish inputs needed
unlocking. Memory impact is negligible — each is < 1 KB per chip.

## In-place refill (the tt-metal pattern)

Mirrors `models/tt_transformers/tt/common.py:565-570`:

```python
ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
```

writes new host data INTO the existing device tensor (in place, at the
pinned buffer address). After capture, before `execute_trace`, you refill
each of the 10 activation buffers from a different step's tensorbins. The
replay's recorded ops read from those same addresses → they compute on the
refilled data.

`pcc.py --trace` does this for you with `--replay-from <dir>`. The trace
records ops on whatever `--trace-from` pointed at; `--replay-from`
(defaulting to `--trace-from`) controls what gets refilled before
`execute_trace`. Honest measurement pattern:

```
pcc.py --trace \
    --warmup-from ./tensors           # step 1 — populate kernel cache
    --trace-from  ./tensors           # step 1 — records ops with these addresses
    --replay-from ./tensors_step2     # step 2 — buffers refilled before replay
```

The replay then computes step-2 logits, PCC validates them against
`golden_logits_step2.pt`. Verified at 0.92 PCC.

## End-to-end recipes

### Tracy perf measurement (warmup-pass numbers, comparable to E47 baseline)

```bash
tt-smi -glx_reset_auto
docker exec tt-xla-ird-mvasiljevic bash -lc "
  cd /home/mvasiljevic/tt-xla && source venv/activate &&
  cd deepseek_codegen/graph_0 && export TT_METAL_RUNTIME_ROOT=\$TT_MLIR_HOME/third_party/tt-metal/src/tt-metal &&
  export PYTHONPATH=\"\$TT_METAL_RUNTIME_ROOT/ttnn:\$TT_METAL_RUNTIME_ROOT/tools/:\$PYTHONPATH\" &&
  export FAKE_DEVICE=TG TTXLA_USE_TRACE=1 TTXLA_TENSORS_DIR=./tensors &&
  python3 -m tracy -r -m -v -p \
      --op-support-count 10000 \
      --tracy-tools-folder \$TT_METAL_RUNTIME_ROOT/build/tools/profiler/bin/ \
      -n E50_<patch_name> 'main'
"

# Scope to the warmup pass (first decode_1_start..decode_1_end):
RD=deepseek_codegen/graph_0/generated/profiler/reports/E50_<patch_name>/<TS>
tt-perf-report $RD/ops_perf_results_E50_<patch_name>_<TS>.csv \
    --start-signpost decode_1_start --end-signpost decode_1_end \
    --summary-file $RD/summary
# read summary.csv / summary.txt, sum Device Time Sum column
```

`./run -t` is the canonical wrapper but doesn't pass `--op-support-count`;
expand it to the explicit `python3 -m tracy` form above when you need the
bigger profiler buffer.

### PCC validation (with trace API + step-2 replay)

```bash
tt-smi -glx_reset_auto
docker exec tt-xla-ird-mvasiljevic bash -lc "
  cd /home/mvasiljevic/tt-xla && source venv/activate &&
  cd deepseek_codegen && export FAKE_DEVICE=TG &&
  python pcc.py --trace \
      --warmup-from ./tensors \
      --trace-from  ./tensors \
      --replay-from ./tensors_step2
"
# logits PCC ≥ 0.9 vs golden_logits_step2.pt confirms the patch handles
# step-2 inputs correctly via the in-place-refill replay.
```

### Replay wall-clock (option B)

Already wrapped: `pcc.py` emits `REPLAY_START` / `REPLAY_END` signposts
around `ttnn.execute_trace`. After a run, the diff of their `HOST START TS`
in `ops_perf_results_*.csv` is the replay wall-clock in nanoseconds. For
this graph it's been ~26-27 ms across runs.

`pcc.py` also prints the Python `time.perf_counter_ns` around the same
call. Both numbers should agree to within a few μs.

## Verified numbers (this branch, May 25 2026)

| | Total | Method |
|--|--|--|
| E47 baseline (step-1) | 22,522 μs | single `_main`, no trace, no `--op-support-count` needed |
| E49 step-1 (`./run -t` no trace) | 22,663 μs | same as E47, +141 μs (rebase noise) |
| E49 step-1 (`./run -t` + `TTXLA_USE_TRACE=1` + `--op-support-count 10000`, warmup pass) | 22,689 μs | matches non-trace within 26 μs |
| E49 step-2 replay wall-clock | ~27 ms | `REPLAY_START..REPLAY_END` signposts |
| E49 step-2 standalone `_main` | 27,565 μs | step-2 has heavier MoE routing → SparseMatmul ~doubled vs step-1 (expected, data-dependent) |
| `pcc.py --trace --warmup-from tensors --trace-from tensors --replay-from tensors_step2` | 0.921465 PCC vs `golden_logits_step2.pt` | confirms emit is position-agnostic |
| `pcc.py --verify-set-b ./tensors_step3` | 0.909505 PCC vs `golden_logits_step3.pt` | step-3 also clean |

## Key takeaways for future tuning iterations

1. **For per-op perf comparison vs E47**: use the warmup pass via
   `--op-support-count 10000` + tracy. That's what `E49 step-1 = 22,689 μs`
   used. Apples-to-apples with `regions_warm/E47warm_*.csv`.
2. **For "did the patch help step-2 / step-3 correctness too"**: run
   `pcc.py --verify-set-b ./tensors_step2` + `./tensors_step3`. Both ≥ 0.9
   means the patch isn't overfit to step-1 data.
3. **For "is the replay actually faster than the warmup"**: trust the
   `REPLAY_START / REPLAY_END` wall-clock window. Per-op breakdowns are
   only available via option C, which we haven't wired up yet.
4. **If a perf measurement comes back as 0 μs or shows fewer ops than
   expected**: profiler buffer overflowed. Bump `--op-support-count`.
