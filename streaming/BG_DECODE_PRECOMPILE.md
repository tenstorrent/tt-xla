# Background decode pre-compile (option A)

Idea: overlap the decode graph's cold compile with prefill's cold compile by spawning a background thread that calls `compiled(fake_decode_input)` in parallel with the main thread's prefill call. Goal: shave the decode cold-compile time off the critical path so the first decode token returns roughly `prefill_compile + 1×decode_exec` instead of `prefill_compile + decode_compile + 1×decode_exec`.

Companion docs: [HYBRID_PROGRESS.md](HYBRID_PROGRESS.md), [DEBUG_HYBRID_NOTES.md](DEBUG_HYBRID_NOTES.md).

## Why this isn't free

`torch.compile`'s wrapped function doesn't expose a compile-only mode in this torch_xla build (`torch_xla.compile` is JIT-style — calling it triggers trace + XLA compile + execute together). So the bg thread's `compiled(fake_token, fake_sp)` call doesn't just compile the decode graph — it also **executes** it on the real model, which means the model's mutable buffers (`kv_cache`, `kv_state`, `score_state` on every block) get written.

If the bg thread's exec phase overlaps with the main thread's prefill exec phase, both threads write the same KV buffers concurrently → races, potentially incorrect KV state for the real decode loop. Output correctness is not guaranteed in this mode.

We accept that risk for an experimental run because the streaming work explicitly does not require output correctness right now (the workload is a streaming/RAM benchmark, not an accuracy run). The bg thread re-zeros the mutable KV buffers after its exec to leave the model in the same post-ship state the foreground prefill expects.

## Compile vs exec phases — when does parallelism actually help?

The bg call decomposes (in order, inside the bg thread) as:
- dynamo trace (Python, fast)
- StableHLO → TTNN compile (in PJRT plugin; this is the slow part — ~tens of seconds to many minutes)
- XLA execute on device
- `torch_xla.sync(wait=True)` — wait for execute to finish

The main thread's prefill call has the same phases for the prefill graph. The two threads can usefully overlap on:
- bg compile vs main compile (if the PJRT plugin doesn't internally serialize compiles)
- bg compile vs main exec (if bg compile is slower than main compile)

They CANNOT safely overlap on:
- bg exec vs main exec (KV race)

The "lucky-case" timing where this saves time: bg's `compile + exec` total is shorter than main's compile alone. Then bg finishes while main is still compiling, KV is reinitialized to zeros by bg, and main's exec runs on clean KV with no race.

For the 10-layer expK measurements (extrapolating from expK5):
- Main prefill: compile ≈ 120s, exec ≈ 10s
- Bg decode: compile ≈ 110s, exec ≈ 1s, reinit ≈ 1s ⇒ total ≈ 112s
- Lucky-case wall time ≈ max(120, 112) + 10 + (decode_exec_hot 1s) ≈ 131s
- Without bg ≈ 120 + 10 + 110 + 1 ≈ 241s
- Savings ≈ 110s (≈ 45 %) — **only if bg compile and main compile actually run in parallel inside the PJRT plugin** (open question; see below).

For 43-layer the magnitudes scale super-linearly (compile is O(N²)-ish in layer count), so the absolute savings should be larger though the percentage similar.

## Implementation status

Patch is already in [run_hybrid.py](run_hybrid.py) behind env var `STREAM_HYBRID_BG_DECODE_COMPILE=1` (default off):
- A bg `threading.Thread` is spawned right before the prefill call.
- The thread calls `compiled(fake_token, fake_sp)` with `fake_token.shape == (bsz, 1)` matching the decode shape.
- After `sync` + `wait_device_ops`, the thread re-zeros all mutable KV buffers via `_reinit_mutable_kv_buffers`.
- Main thread joins the bg thread after its prefill call returns and before entering the decode loop, so any leftover bg work is drained before the real decode token reads the KV state.

Timing logs added:
- `[bg] decode pre-compile thread launched in parallel with prefill`
- `[bg] decode pre-compile: compile+exec=Xs reinit_kv=Ys` (printed from bg thread when it finishes)
- `[bg] joined after prefill, wait=Zs, bg total wall=Ws`

## Open questions to validate

1. **Are PJRT compiles actually parallel?** The plugin / tt-mlir compiler may hold a global mutex around compile, which would serialize the two compile calls. If so, no time savings — bg compile waits behind main compile (or vice versa). First validation run measures `bg total wall` vs main `prefill compile+exec` and the ratio tells us.

2. **Is dynamo's `compiled(...)` callable safe to invoke from two threads on the same wrapped module?** Dynamo's `_DynamoFrameState` and torch_xla's lazy IR construction may have thread-local state that interacts oddly under concurrent calls. First validation surface — if the run crashes / wedges, this is the cause.

3. **KV race in the lucky case**: even when bg finishes its exec before main's exec starts, the GPU/TT device queue scheduling may have main's compile already submitting work that races with bg's reinit. We're not synchronizing across threads at the CUDA/Metal queue level. Empirical: if outputs become wildly different from baseline, race is happening.

## Validation plan (when we come back to this)

1. **Sanity**: `STREAM_HYBRID_BG_DECODE_COMPILE=0` 5-layer run → record baseline timings (prefill compile+exec, decode 1 cold, decode 2 hot).
2. **Try BG=1**: same 5-layer config → record `[bg] ...` timing logs + main prefill timing + decode 1 elapsed (should drop dramatically if cache hit).
3. **Compare**:
   - If `bg total wall ≈ prefill compile` ⇒ compiles ran in parallel ⇒ savings as expected.
   - If `bg total wall ≈ prefill compile + bg compile` ⇒ PJRT serialized them ⇒ no savings, abandon this approach until plugin-side compile concurrency is unlocked.
   - If process crashes ⇒ dynamo / torch_xla thread safety issue.
4. If sanity-passing on 5 layer: try 10 layer for direct comparison with [expK5_10layer_2decode.log](../streaming_log/expK5_10layer_2decode.log).

## Cleaner alternatives (deferred)

- **Compile-only API in torch_xla**: would let bg thread emit HLO + run PJRT compile without execute. Removes the KV race entirely. Not exposed in current torch_xla build (probed: `torch_xla.compile` is JIT-style, no AOT-only entry point).
- **Separate KV buffer copies for bg**: bg uses cloned KV state, main uses real. Doubles device memory for those buffers temporarily. Implementation is moderate (forward hook to swap buffers in/out, plus a clone pass).
- **PJRT-side leak fix to enable pass=ON path**: makes the const-eval round-trip not be emitted at all (no `from_device` in const_eval functions on the new env), which removes the new-env hang and is the real production direction. See [HYBRID_PROGRESS.md](HYBRID_PROGRESS.md) "Outstanding work" section. Doesn't directly address compile-time savings but is the higher-priority fix.

## Decision log

- 2026-05-05 — patch landed; first validation deferred. User killed the baseline run before measurement.
- 2026-05-05 — validated, **fails with current torch_xla**. Two attempts:
  - **L1**: bg thread reuses the main thread's `compiled = torch.compile(model)` wrapper. Crash in dynamo: `RuntimeError('s72 (...) is not tracked with proxy for _ModuleStackTracer')`. dynamo's fake-tensor proxy state is per-OptimizedModule and not thread-safe.
  - **L2/L3**: bg thread creates its own `bg_compiled = torch.compile(model)` wrapper. Same crash — `_ModuleStackTracer` is tied to the model's module hierarchy, not the wrapper, so two threads still share trace state when traversing the same model.

Conclusion: dynamo + multi-thread compile of the same model is not viable in this torch_xla. Path forward would be (a) `torch.export` / AOT to decouple trace from compile, (b) FX-level pre-emission outside dynamo, or (c) wait for upstream fix. Patch left under `STREAM_HYBRID_BG_DECODE_COMPILE=1` for revisit; default-off so production runs are unaffected.
