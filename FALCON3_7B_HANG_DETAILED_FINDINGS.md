# Serving hangs permanently under sustained multi-bucket traffic (Falcon3-7B-Instruct, full model) — confirmed device-side dispatch stall, not a host synchronization bug

> **This is the detailed findings doc, not the issue to file.** See
> `ISSUE_DRAFT_falcon3_7b_completion_queue_hang.md` for the short version meant for filing
> upstream (Problem / Repro / Helpful notes, with a pointer back here). This doc has the full
> evidence, root-cause investigation (including dead ends worth knowing about), confidence
> assessment, and the minimal reproducer write-up. For the complete chronological lab notebook
> (every single experiment, in the order it happened), see `FALCON3_SINGLE_LAYER_HANG_DEBUG.md`.

## TL;DR

**Status: not yet fixed, but conclusively characterized.** Direct evidence (see "Decisive
finding" below) proves this is a **genuine on-device dispatch/completion-signaling stall**,
not a host-side thread-safety bug: (1) live instrumentation of every lock in the relevant
host-side class (`FDMeshCommandQueue`) shows **zero thread concurrency** at that lock during
the actual hang — a single application thread does all dispatch sequentially — and (2)
setting the existing `TT_METAL_OPERATION_TIMEOUT_SECONDS` env var (no code change) converts
the exact same silent, permanent hang into a clean, catchable exception
(`TIMEOUT: device timeout, potential hang detected, the device is unrecoverable`) at the
identical stack location, ~30 seconds after the stall begins. Two real, independently
confirmed host-side bugs were also found and fixed along the way (both missing a lock in
`FDMeshCommandQueue`, filed separately) — but the evidence above shows neither is (or
plausibly could be) the cause of *this* hang, since it structurally requires two-thread
lock contention this workload never produces. **The actual defect is now understood to be
in on-device dispatch/completion-signaling logic** (likely go-signal/worker-completion-count
sequencing, or an equivalent on-device state machine) rather than anywhere in
`FDMeshCommandQueue`'s host-side API surface — see "Where the fix likely belongs" for the
updated understanding, and "Workarounds" for a substantially better production mitigation
found this session (`TT_METAL_OPERATION_TIMEOUT_SECONDS`, zero steady-state cost). See "Where
this leaves the investigation" near the end for the most promising next steps.

## Summary

A Falcon3-7B-Instruct vLLM server on Tenstorrent hardware (P150, single Wormhole-class
chip) hangs permanently mid-request during serving. **Originally found on CI** running the
Falcon3-7B-Instruct release-eval workflow (ifeval + gpqa_diamond_generative_n_shot,
full/not-downsampled datasets); reproduced live, repeatedly, on a dev box with both
tt-media-server and pure standalone `vllm serve`, at full model depth. Once hung, the
server never recovers — it holds the device, the chip firmware stays healthy
(`TIMER_HEARTBEAT` ticking), but one native thread spins forever waiting for a completion
queue entry that never arrives. No crash, no error, no timeout on the device side; only the
client's own request timeout eventually surfaces anything (a `TimeoutError` ~30 min later
with lm_eval's default 1800s timeout).

This is **not** the previously-fixed vLLM scheduler deadlock (tt-inference-server#4521) —
confirmed independently on 8+ live catches (see Evidence).

## Where this was found

CI: Falcon3-7B-Instruct nightly/release eval workflow (ifeval → gpqa_diamond_generative_n_shot,
full datasets, via tt-inference-server's `run.py --workflow evals`).

Also reproduces on a dev box, both:
- tt-media-server (`device_worker_dynamic_batch.py` fire-and-forget burst admission)
- pure standalone `vllm serve` (vLLM-native admission, no tt-inference-server/tt-media-server
  anywhere in the process tree)

Model depth matters: single-layer (`NUM_HIDDEN_LAYERS=1`) debug configs did **not**
reproduce this under any admission pattern tested (smooth concurrency=32, bursty
fire-and-forget up to 566 in-flight); full model depth reproduces reliably. The hang always
occurs at (or shortly after) the ifeval→gpqa transition, i.e. under sustained serving with a
resident graph count near/at the full warmup set.

## Repro (tt-xla standalone, no tt-inference-server needed)

Two scripts at the tt-xla repo root reproduce this with only `vllm serve` +
`lm_eval`/a raw HTTP client — see `serve_falcon3_7b_forge.sh` / `run_falcon3_7b_evals.sh` in
this repo (config byte-verified against a live tt-media-server run: same
`additional_config`, same vLLM engine config dump, same KV cache size).

```bash
cd ~/tt-xla-2 && source venv/activate

# Server: untouched baseline config, FULL model depth (single-layer does not reproduce)
NUM_HIDDEN_LAYERS="" ./serve_falcon3_7b_forge.sh
# wait for "[warmup] WARMUP COMPLETE" in the log (~7-8 min, one-time compile of 54 graphs)

# Evals: ifeval downsampled (faster iteration), gpqa full -- hang occurs at the transition
./run_falcon3_7b_evals.sh --ifeval-limit 0.10 --gpqa-limit 1.0
```

Defaults baked into `serve_falcon3_7b_forge.sh` reproduce the exact hanging config:
`gpu_memory_utilization=0.35`, `bfp8` weights+KV, `optimization_level=1`, `enable_trace=true`,
`cpu_sampling=false`, b1-prefill on (`min_num_seqs=1`, `prefill_batch_threshold=16`),
`prefill_chunk_size=1024`, port 8019.

Also reproduces via the real tt-inference-server client
(`~/scripts/model_servers/run_evals_forge.sh --model Falcon3-7B-Instruct --port 8019`)
against the same standalone server — confirming the standalone repro is a faithful stand-in
for the CI/production client, not an approximation.

**Reproducibility**: 8/10 attempts on the untouched baseline config as of this writing
(across single requests/threads and multiple probe configurations, both admission clients,
both tt-media-server and standalone `vllm serve`). Every clean run at reduced graph count or
with an added synchronization point is expected (see Root cause below) — but critically,
**two of the clean runs happened against the exact same still-warm, unmodified server that
hung on a subsequent attempt minutes later** (same resident graph count, same compiled
state, no restart in between): two clean full-dataset ifeval→gpqa passes, then a hang on a
third attempt. This is direct evidence the trigger is a genuine timing-sensitive race
(probability-of-hit per attempt) rather than a deterministic function of static server
state — consistent with the separately-observed confound where verbose per-op logging alone
(no functional change) also happened to mask the hang once.

## Minimal reproducer (pure `ttnn`, no vLLM/tt-xla/tt-mlir-runtime) — the artifact that found the root cause

Three small standalone Python scripts (repo root: `ttnn_trace_race_repro.py`,
`ttnn_trace_race_repro_singlethread.py`, `ttnn_trace_race_repro_replay_only.py`), using only
`ttnn`'s low-level trace API (`begin_trace_capture`/`end_trace_capture`/`execute_trace`/
`release_trace`) against a single `ttnn.open_device(device_id=0, trace_region_size=0)` —
**zero vLLM, zero tt-xla, zero tt-mlir runtime**. Cleanly isolate the exact necessary
condition:

| Variant | What it does | Result |
|---|---|---|
| Sequential | 4 traces, 3 rounds of release+recapture, no concurrency at all | Clean |
| Replay-only | 4 traces captured once, then replayed concurrently forever from 4 threads (940k+ replays/20s), no eviction ever | Clean |
| Concurrent replay + recapture | 4 traces; 1 background thread continuously non-blocking-replays 2 of them; main thread *simultaneously* releases+recaptures the other 2 | **Crashes immediately, 2/2 runs** |

**The bug requires a trace being released+recaptured while a *different* trace's replay is
concurrently in flight on another thread.** Neither ingredient alone is sufficient. At just 4
tiny (256×256) traces this reproduces in seconds — no 54-graph production workload needed;
that just raises the odds of hitting the same window.

Crash signature (both pre-fix runs, identical):
```
Segmentation fault (11), Signal code: Address not mapped
  RingbufferCacheManager::add_manager_entry_no_evict(...)
  RingbufferCacheManager::get_cache_offset(...)
  FDMeshCommandQueue::record_end()
  MeshDeviceImpl::end_mesh_trace(...)
  ttnn::operations::trace::end_trace_capture(...)
```
(One run additionally hit a glibc `malloc(): invalid size` heap-corruption abort before the
segfault — confirming genuine memory corruption, not merely a null read.)

## Evidence (identical across every occurrence — 8+ live catches)

**CPU-delta thread sweep**: exactly one native thread accumulates ~100% of one core (all
other ~60-130 threads, including RPC/gloo/tqdm/zmq workers, flat at 0 CPU).

**gdb native backtrace** of that thread (byte-for-byte identical every time, only the
outermost couple of frames vary slightly by exact sample instant):
```
tt::umd::SysmemManager::read_from_sysmem(...)
tt::tt_metal::read_cq_host_ptr<true>(...)
tt::tt_metal::SystemMemoryManager::completion_queue_wait_front(...)
tt::tt_metal::buffer_dispatch::copy_completion_queue_data_into_user_space(...)
tt::tt_metal::distributed::FDMeshCommandQueue::copy_buffer_data_to_user_space(...)
tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue()
```

**py-spy** MainThread: genuinely mid-step inside a live `sample_tokens` call (not vLLM's
outer idle loop):
```
sample_tokens (vllm_tt/model_runner.py) — blocked at the .cpu() pull at the end of sampling
decorate_context → sample_tokens (vllm_tt/worker.py)
run_method → collective_rpc → sample_tokens (vllm/v1/executor/uniproc_executor.py)
step_with_batch_queue → _process_engine_step → run_busy_loop (vllm/v1/engine/core.py)
```

**Scheduler is healthy**: `py-spy --locals` shows a live, non-empty `SchedulerOutput`;
internal `[TT-SCHED]` accounting prints show ordinary admit/decode cycles right up to the
freeze, with no anomaly. This directly rules out a repeat of the earlier vLLM scheduler
deadlock (tt-inference-server#4521), which had the completion-queue thread idle and the
MainThread parked in an *empty*-schedule idle loop — the opposite of what's observed here.

**Chip firmware alive**: `tt-smi -s` shows `ARCCLK`/`DDR_STATUS` nominal and
`TIMER_HEARTBEAT` actively incrementing throughout the hang. Not a dead/crashed ASIC —
specifically the completion-queue/dispatch path has stopped advancing for one in-flight
buffer read.

**Compile timing rules out in-flight recompiles**: in more than one occurrence, every
`Compiling graph for config=...` line in the server log occurred during startup warmup only,
zero new compiles anywhere near the transition or the freeze itself — so this is not "a
recompile corrupts state," it's a **replay/eviction-time** issue.

## A confirmed bug found along the way (not yet shown sufficient to explain the full hang)

**`FDMeshCommandQueue::record_end()` (`tt_metal/distributed/fd_mesh_command_queue.cpp`),
invoked when a trace capture finishes (`end_trace_capture`), is missing a lock that every
sibling mutator of the same shared state takes.** `FDMeshCommandQueue` owns a single
`std::unique_ptr<RingbufferCacheManager> prefetcher_cache_manager_` — tt-metal's L1
prefetcher cache for dispatched kernel binaries (`tt_metal/impl/dispatch/ringbuffer_cache.{hpp,cpp}`),
a completely different subsystem from the DRAM buffer allocator this investigation spent
most of its time on before finding this. Confirmed by direct grep of the file:

- `enqueue_mesh_workload()` (`:385`), `record_begin()` (`:1269`), and `enqueue_trace()`
  (`:1226`, **the replay path**) each take `auto lock = lock_api_function_();` as their very
  first line before touching `prefetcher_cache_manager_`.
- `record_end()` (`:1307`) does **not** — it calls `reset_prefetcher_cache_manager()` and
  `swap(dummy_prefetcher_cache_manager_, prefetcher_cache_manager_)` with zero lock held.

A trace replay (`enqueue_trace`, which legitimately holds the lock) on one host thread can
therefore race against `record_end()`'s unlocked mutation of the identical
`RingbufferCacheManager` on another thread, corrupting the ring buffer's internal
offset/entry bookkeeping — verified via the minimal reproducer above (crash 2/2 before a
one-line fix, clean 5/5 + a 1.24M-replay stress test after). **This is a real, independently
confirmed bug, filed as its own focused report**
(`ISSUE_DRAFT_fd_mesh_command_queue_record_end_missing_lock.md`) since it stands on its own
regardless of this issue's outcome.

**However — applying this exact fix to the full Falcon3-7B production workload and
retesting end-to-end does NOT stop the original hang.** See "End-to-end verification" below.
So while this bug is real, confirmed, and worth fixing, it is evidently not the (sole) cause
of the hang described in this issue. The most likely explanation: the same general area
(`FDMeshCommandQueue`'s trace capture/replay/dispatch machinery manages several other pieces
of shared mutable state — `config_buffer_mgr_`, `expected_num_workers_completed_`,
`worker_launch_message_buffer_state`, `sysmem_manager()`, and more — any of which could have
a similar missing-synchronization issue that a tiny 4-trace reproducer doesn't happen to
exercise) likely harbors at least one more bug of the same general class.

## End-to-end verification: fix applied to production config — hang still occurs

Rebuilt tt-xla's plugin with the `record_end()` lock fix baked in (no other changes, no
env-var workarounds), relaunched the exact untouched-baseline Falcon3-7B repro
(`serve_falcon3_7b_forge.sh`, full model depth) and ran the same eval loop that has hung
repeatedly throughout this investigation. Warmup completed in the same ~465s as baseline
(confirming the fix adds no meaningful overhead — it's a mutex around infrequent
bookkeeping, not a hot path). **Hung on the first iteration, at the identical ifeval→gpqa
transition, with a byte-for-byte identical native signature**
(`is_dram_backed → read_cq_host_ptr → completion_queue_wait_front → ... →
read_completion_queue`; chip healthy, `TIMER_HEARTBEAT` still incrementing).

## Where this leaves the investigation

**Update**: a source audit of every shared mutable member of `FDMeshCommandQueue` found a
*second* real, same-shaped missing-lock bug: `wait_for_completion()` (reached only via
`ttnn.MeshDevice.quiesce_devices()`, a submesh-teardown API) mutates
`expected_num_workers_completed_`/`config_buffer_mgr_`/`cq_shared_state_->{sub_device_cq_owner,
worker_launch_message_buffer_state}` with zero locking anywhere in its call chain, exactly
like `record_end()` did. Filed separately
(`ISSUE_DRAFT_fd_mesh_command_queue_wait_for_completion_missing_lock.md`) and fixed the same
way. **However, `quiesce_devices()`/`wait_for_completion()` do not appear anywhere in
tt-mlir's `runtime/` or tt-xla's `python_package/` (zero grep matches) — this workload never
calls it** — and, as predicted before testing, applying this second fix on top of the first
and retesting end-to-end **still did not stop the hang**, identical signature. Two real bugs
found and fixed; neither is (so far) the actual cause of this specific hang.

- Two real, confirmed, independently-reproducible bugs were found and fixed
  (`record_end()`'s and `wait_for_completion()`'s missing locks) — report/land both
  regardless of the rest of this issue's status; they can only help elsewhere.
- The Falcon3-7B production hang persists after **both** fixes.

**Decisive update — the search moved off `FDMeshCommandQueue`'s host-side locks entirely
and onto the device side:**

1. **Live lock-trace instrumentation** of all 12 `lock_api_function_()` call sites in
   `FDMeshCommandQueue`, captured during an actual production hang, showed **every one of
   255,758 lock acquisitions across the whole run came from a single thread** — zero
   concurrency at this lock in the real workload. This retroactively explains why both
   confirmed bugs' fixes didn't help: they require two-thread contention on this exact lock,
   which this workload's actual (single-threaded, w.r.t. this lock) dispatch pattern never
   produces.
2. Following that, source-audited the completion-queue reader thread's *own* separate
   synchronization (`reader_thread_cv_mutex_`/`reads_processed_cv_mutex_`) and found it
   correctly implemented (producer and consumer consistently lock the same mutex around the
   matching state change) — ruling out a classic missed-wakeup bug there too.
3. That analysis pointed at the busy-poll itself
   (`SystemMemoryManager::completion_queue_wait_front()`) being **unbounded by default**
   (`TT_METAL_OPERATION_TIMEOUT_SECONDS` defaults to 0 = no timeout). **Tested with zero code
   changes**: setting this existing env var to 30 converts the identical silent hang into a
   clean `TT_THROW`/`RuntimeError` (`"TIMEOUT: device timeout, potential hang detected, the
   device is unrecoverable"`) at the exact same stack location, ~30s after the stall begins.

**This conclusively shows the defect is in on-device dispatch/completion-signaling logic,
not host-side thread safety.** The most promising next steps:
1. **Engage tt-metal's dispatch/firmware owners directly** with this repro and the
   `TT_METAL_OPERATION_TIMEOUT_SECONDS` finding — this now points at a specific, well-
   characterized subsystem (on-device dispatch/completion signaling, likely go-signal or
   worker-completion-count sequencing) rather than "somewhere in trace replay," and is
   likely to get a much faster, more targeted response than continued host-side guessing.
2. Instrument (or find existing tooling for) the on-device dispatcher's state directly —
   e.g. via `tt-triage` if it can be made safe to use (see this repo's own notes on a prior
   host-crash incident with that tool), or firmware-level debug prints/telemetry around
   go-signal and worker-completion counting.
3. Revisit the original DRAM-pool/generation-id eviction mechanism documented in "How this
   was found" (still real, verified against source) as a possible *trigger* for the
   device-side state getting into a bad configuration in the first place, even though it's
   now clear the actual stall manifests entirely on the device side.

### How the confirmed bug was found — investigation history (kept for context; the DRAM-pool avenue below turned out not to be sufficient on its own, but every experiment was real and the elimination process is what led to the confirmed fix above)

Two structurally-targeted interventions were tried first and both failed to prevent the
hang: isolating trace buffers into their own dedicated DRAM region
(`TT_RUNTIME_TRACE_REGION_SIZE`), and synchronizing the device precisely at the moment a
trace buffer's memory is released (a more surgical alternative to the blanket workaround
below). Only the blanket "sync after **every** trace replay" (`TT_RUNTIME_SYNC_AFTER_TRACE`)
had been proven effective at that point. Since that is also the only tested intervention
that unconditionally prevents any two trace replays from ever being
in-flight concurrently, the leading hypothesis has shifted: **the hazard is likely two (or
more) trace replays enqueued/in-flight concurrently, not specifically "DRAM address reuse at
eviction time."** The DRAM-pool-sharing and generation-id-eviction mechanism below is still
real and verified directly against source, and graph count/DRAM footprint is still a
genuine causal factor (more resident graphs → more distinct traces → more chances of two
being replayed close together in a busy serving window) — but the missing synchronization
is now believed to live in the trace **replay/dispatch enqueue path**, not the
eviction/release path specifically. The mechanism below is presented as background/context;
see "Confidence assessment" and "Where the fix likely belongs" for the current best
understanding of exactly where the missing synchronization needs to go.

1. **Trace buffers and const-eval results share one DRAM pool.** tt-metal's allocator only
   gives trace command buffers a dedicated, isolated region if `trace_region_size > 0`
   (`tt_metal/impl/context/context_descriptor.hpp`, default `0`). tt-xla's PJRT plugin only
   sets this if `TT_RUNTIME_TRACE_REGION_SIZE` is exported
   (`pjrt_implementation/src/api/client_instance.cc`) — never set in this whole
   investigation. In "dynamic allocation mode," `MeshTrace::populate_mesh_buffer`
   (`tt_metal/distributed/mesh_trace.cpp`) allocates trace buffers as `BufferType::DRAM` —
   the exact same pool used for weights, const-eval results (permanently retained per
   `platform.py`'s own "storing the entire model on device once per graph" comment, tracked
   separately as tt-mlir#3888), and activations.

2. **Every first-time trace capture invalidates every other already-captured trace,
   unconditionally.** `TraceCache` keeps one global monotonic `generationId` per device.
   `CaptureOrExecuteTraceOp::run()`
   (`runtime/lib/ttnn/operations/trace/capture_or_execute_trace.cpp`) calls
   `traceCache->incrementGeneration()` on every cache miss — not scoped to the specific
   trace being captured. Any other trace whose `generationId` is now stale gets evicted and
   recaptured next time it's needed, regardless of whether the new capture's memory
   footprint could ever have actually touched it. This is documented as a deliberately
   conservative, correctness-motivated design (the `TraceCache` class comment states trace
   reuse is "inherently unsafe" after any new capture).

3. **Trace eviction has a synchronous host-side path with zero device-side wait anywhere in
   it.** Full call chain: `TraceCache::erase()` → `ttnn::operations::trace::release_trace()`
   → `MeshDeviceImpl::release_mesh_trace()` → `SubDeviceManager::release_trace()` (a bare
   `unordered_map::erase`) → `~MeshTraceBuffer()` → `~MeshBuffer()` →
   `Buffer::deallocate_impl()` (`tt_metal/impl/buffers/buffer.cpp`). The only substantive
   call in that chain is `allocator_->deallocate_buffer(this)` — no `Finish()`,
   `Synchronize()`, event wait, or fence anywhere. Since tt-metal's command queue is
   asynchronous, the freed DRAM address becomes eligible for reuse **immediately** — by the
   very next line of code (the new capture) — with no guarantee the device has finished
   consuming whatever was previously enqueued against that address, including an in-flight
   replay of a different, still-valid trace. **This is the race.**

4. This is a known-but-lightly-tested hazard class upstream:
   `runtime/test/ttnn/python/n150/test_trace.py::test_trace_memory_overwrite_multi_graph`
   encodes the same scenario in miniature (2 graphs, single loop) and its docstring
   describes the generation-id mechanism as the intended mitigation — but it never exercises
   the scenario under the graph-count/concurrency pressure needed to expose the race, so it
   passes while this workload hangs.

### Supporting experimental evidence

- **Resident graph count / DRAM footprint is causally necessary**: adding *inert* decoy
  compiled graphs (compiled during warmup, provably never invoked by real traffic — verified
  by grepping the server log for any request touching a decoy bucket size, zero matches)
  alone reproduces the hang once total graph count crosses the baseline's 54-graph
  threshold. Every tested config below ~54 resident graphs (regardless of which axis —
  `num_reqs=1` b1-prefill, token-padding bucket count, greedy/grammar warmup variants — was
  used to shrink it) has been clean; every config at the untouched baseline's 54 has hung,
  6/6.
- **...but not sufficient on its own**: forcing a device sync after every trace *replay*
  (see Workarounds) prevents the hang at the *same* graph count/DRAM pressure that reliably
  triggers it otherwise — i.e. serializing trace executions removes the opportunity for the
  race regardless of DRAM pressure. This is the expected signature of a race condition:
  footprint creates the *conditions*, timing/overlap triggers the *actual* corruption/stall.
- **Confound, and independent confirmation this is a genuine timing-sensitive race, not
  deterministic**: verbose per-op `LOG_INFO` logging alone (no explicit sync) also happened
  to mask the hang once, on the exact config that hung 6/6 without it — classic
  "heisenbug" behavior where any added per-op latency perturbs the race window. Any future
  experiment measuring this needs `TTMLIR_RUNTIME_LOGGER_LEVEL=WARNING` to avoid this
  confound while still observing whether the hang itself reproduces. Independently
  reconfirmed overnight: the same warm server produced two clean full-dataset passes
  followed by a hang on a third attempt, with zero change to server state in between.
- **`TT_RUNTIME_TRACE_REGION_SIZE` isolation does NOT prevent the hang — refines the
  mechanism**: isolating trace buffers into their own dedicated DRAM region (structurally
  separate from weights/const-eval) still hangs, identical signature, once the region is
  sized large enough to avoid a separate DRAM-OOM failure mode (4GiB was too large and
  OOM'd during serving; 1GiB compiled and served cleanly up to the hang). This shows the
  race is trace-buffer-vs-trace-buffer, not trace-buffer-vs-other-pool: the zero-sync
  eviction bug in `Buffer::deallocate_impl()` applies identically to any pool containing
  multiple trace buffers, so confining all 54 traces to their own region does not remove
  the hazard — it just moves the same race into a smaller box.

## Confidence assessment

**Confirmed, but shown insufficient alone to explain this issue's hang:**
- `FDMeshCommandQueue::record_end()` is missing `lock_api_function_()`, unlike every sibling
  mutator of `prefetcher_cache_manager_`/`dummy_prefetcher_cache_manager_`. Verified by direct
  source inspection, and by a minimal reproducer (crash 2/2 before a one-line fix, clean 5/5 +
  a 1.24M-replay stress test after). **Filed as its own issue**
  (`ISSUE_DRAFT_fd_mesh_command_queue_record_end_missing_lock.md`) since it's real regardless.
- Applying that exact fix to the full Falcon3-7B production config and retesting
  end-to-end: **hang still occurs**, identical signature. So this bug, while real, is not
  the (sole) cause of the hang described in this issue.

**Still real, verified against source, but not shown to be the cause either:**
- Trace buffers and const-eval results share one DRAM pool by default; generation-id
  invalidation is global/unconditional; the eviction path has zero device-side sync. All
  independently verified against source. Two interventions targeting this specifically
  (`TT_RUNTIME_TRACE_REGION_SIZE` pool isolation; a surgical sync at `TraceCache::erase()`)
  were each tried and neither prevented the hang alone — this doesn't disprove the mechanism,
  it shows those two *particular* interventions aren't sufficient by themselves. Whether this
  mechanism is a genuine contributing factor (perhaps requiring a fix here *in addition to*
  a `FDMeshCommandQueue` fix) is still open.
- `TT_RUNTIME_SYNC_AFTER_TRACE` (syncs after every replay) remains the only tested
  intervention proven to prevent the hang — consistent with either (or both) of the above
  mechanisms, since it serializes dispatch broadly enough to reduce the odds of hitting
  *any* of the unsynchronized windows discussed in this issue.

**Confirmed, decisive (live evidence, not inference):**
- **Zero `FDMeshCommandQueue::api_mutex_` contention during the hang.** Instrumented all 12
  lock-acquisition call sites in this class and captured a live production hang: all 255,758
  acquisitions across the entire run, cleanly matched by release, came from a single thread.
  This rules out the entire class of bug the two confirmed-and-fixed bugs belong to as the
  cause of *this specific* hang — they require two-thread contention this workload never
  produces.
- **This is a device-side stall, not a host synchronization bug.** Setting the existing
  `TT_METAL_OPERATION_TIMEOUT_SECONDS` env var (zero code change) converts the identical
  silent hang into a clean, catchable `TT_THROW`
  (`"TIMEOUT: device timeout, potential hang detected, the device is unrecoverable"`) at the
  same stack location, ~30s after the stall begins. tt-metal's own diagnostic explicitly
  characterizes this as a device-side, unrecoverable-session stall.

**Genuinely still open — the hang's exact device-side cause is not yet found:**
- The specific on-device mechanism (go-signal sequencing, worker-completion counting, or
  equivalent dispatch-firmware state) that stops advancing is not yet identified — this
  requires device-side tooling/expertise beyond what this session's host-side investigation
  could reach.
- Whether the DRAM-pool/generation-id eviction mechanism documented in "How this was found"
  is a genuine *trigger* for the device-side state going bad (as opposed to being unrelated)
  is still open.

(This issue will be updated in place as progress continues — see the companion working notes
at `FALCON3_SINGLE_LAYER_HANG_DEBUG.md` in tt-xla for the full blow-by-blow investigation, and
"Where this leaves the investigation" above for concrete next steps.)

## Where the fix belongs

**On-device dispatch/completion-signaling logic — not `FDMeshCommandQueue`'s host-side API
surface.** Live evidence (zero lock contention during the hang; the hang converts into a
clean, catchable timeout with `TT_METAL_OPERATION_TIMEOUT_SECONDS` set) rules out a host
thread-safety bug as the cause of *this* hang. `FDMeshCommandQueue::record_end()` is still
missing `auto lock = lock_api_function_();` as its first line (a real, separately-confirmed
bug via its own reproducer, see its own issue draft) — land that regardless, it's correct
and harmless, just not the cause here. The actual fix needs someone with visibility into the
on-device dispatch firmware/state machine (go-signal sequencing, worker-completion counting,
or equivalent) — this is squarely a tt-metal dispatch/firmware issue, not a vLLM scheduler or
tt-xla plugin bug (no scheduler frames appear anywhere in any hang capture, and the
scheduler's own accounting is provably healthy throughout).

## Workarounds available today

1. **`TT_METAL_OPERATION_TIMEOUT_SECONDS=30`** (or similar) — **recommended as the primary
   production workaround.** Existing tt-metal env var, zero code changes. Does not prevent
   the underlying device stall, but converts a silent, permanent, undetectable hang into a
   fast (~N seconds), loud, catchable failure — vLLM already handles this by killing the
   EngineCore and returning `500` to in-flight clients, enabling automated monitoring/
   restart instead of an indefinite silent wedge. **Zero steady-state performance cost**
   (only matters once a stall has begun) — meaningfully cheaper than option 2 below for any
   deployment with crash/restart automation already in place. **Caveat**: does not reduce
   how often the stall occurs, only how quickly it's detected and recovered from; in-flight
   requests at the time of the stall are lost.
2. **`TT_RUNTIME_SYNC_AFTER_TRACE=1`** (tt-mlir env var, requires the
   `kmabee/ops_debug_prints_and_sync` cherry-pick / commit `020b756622` or later, already
   pinned in this tt-xla branch) — forces a full device sync after every trace replay.
   Proven to prevent the hang from happening at all, at a real, ongoing throughput cost
   every decode step (serializes host-side dispatch enough in practice to avoid whatever
   device-side sequencing issue triggers the stall). Combine with option 1 for
   defense-in-depth if uncertain.
3. **Reduce resident graph count below the empirically-observed danger threshold** — e.g.
   `TTXLA_WARMUP_GREEDY_ONLY=1 TTXLA_WARMUP_NO_GRAMMAR=1` (skip precompiling sampling-mode
   variants the workload never uses), or reduce the token-padding bucket ladder
   (`MIN_CONTEXT_LEN`, `PREFILL_CHUNK_SIZE`), or disable b1-prefill
   (`MIN_NUM_SEQS=32 PREFILL_BATCH_THRESHOLD=0`). Any reduction below ~54 total compiled
   graphs (exact threshold not yet pinned down between 36 and 54) has been clean in every
   test so far, regardless of which axis did the cutting. **Cost**: varies by knob — some
   (disabling b1-prefill) meaningfully slow steady-state decode; the greedy/grammar knobs are
   closer to free if the workload genuinely never uses those paths (true for both ifeval and
   gpqa here). Reduces *risk* only, doesn't eliminate it.
4. ~~`TT_RUNTIME_TRACE_REGION_SIZE` set nonzero~~ — **tested, does NOT prevent the hang**.
   Isolating trace buffers into their own dedicated allocator region doesn't address the
   actual hazard (now understood to be device-side, not DRAM-pool-sharing related). Not a
   usable workaround; a too-large region size (4GiB tried) can additionally introduce an
   unrelated DRAM-OOM failure mode by starving the shared pool with a non-reclaimable
   carve-out.

## Repro artifacts / working notes

Full investigation, evidence, and running list of tried/untried experiments:
`FALCON3_SINGLE_LAYER_HANG_DEBUG.md` (tt-xla repo root, branch
`kmabee/falcon3_7b_hang_debug`).
