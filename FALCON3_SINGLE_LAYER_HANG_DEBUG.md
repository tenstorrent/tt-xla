# Falcon3-7B-Instruct single-layer hang — debug session notes

Working notes for an eventual issue. Not yet root-caused; this is the evidence gathered so far and the plan for what's next.

## Symptom

A Falcon3-7B-Instruct server hangs during serving — reproduces on **both** tt-media-server and pure standalone `vllm serve` (i.e. tt-xla alone, no tt-inference-server anywhere in the process tree), and at **both** single-layer and full model depth, though so far only ever caught live at full depth. Originally only seen single-layer (`NUM_HIDDEN_LAYERS=1`); **confirmed 2026-07-28 to also reproduce at full model depth**, and **confirmed the same day to reproduce with zero tt-inference-server/tt-media-server involvement at all** — narrowing this decisively toward the tt-xla vLLM plugin / tt-metal stack itself rather than anything about tt-media-server's driver or admission pattern.

| Occurrence | PID (EngineCore) | Config | Server | Caught live? |
|---|---|---|---|---|
| 1 | 3794128 | single layer, default launch script settings (bfp8 weights + bfp8 KV, gmu=0.35) | tt-media-server | no, found already-wedged |
| 2 | 410948 | single layer, bf16 weights (`WEIGHT_DTYPE=`), bf16 KV (`KV_CACHE_DTYPE=none`), gmu=0.20 | tt-media-server | no, found already-wedged |
| 3 (`tt-inference-server-2` log, 2026-07-22) | unknown | single layer, default settings | tt-media-server | no, found already-ended (log just stops) |
| 4 | 875071 | full model (no `NUM_HIDDEN_LAYERS`), default settings | tt-media-server | yes — first live catch, 2026-07-28 |
| 5 | 303 | full model, default settings | pure standalone `vllm serve` — no tt-media-server | yes, 2026-07-28, same day — see below |
| **6** | **1335** | **full model, default (baseline) settings, `ifeval --limit 0.5` / gpqa full** | **pure standalone `vllm serve`** | **yes, 2026-07-28 — third live catch, see below** |

Same native signature across all five (see below) — the hang does not appear tied to weight/KV dtype, memory-utilization, model depth, or which server process hosts vLLM.

## Diagnostic method

`~/scripts/gdb_debug_hang.sh` (written this session): samples per-thread `/proc/<pid>/task/*/stat` CPU ticks over a few seconds to find which thread, if any, is actively spinning vs. idle, then dumps `py-spy` (Python stacks) and a `gdb -batch -ex "thread apply all bt"` (native stack) for the hot thread.

## Evidence (both occurrences, identical)

**CPU-delta**: exactly one thread accumulates CPU time during the sample window (e.g. +230 to +316 ticks over 2-3s, ~100% of one core); every other thread (RPC/gloo/tqdm/zmq workers) is flat at 0.

**gdb native backtrace** of that thread:
```
tt::tt_metal::SystemMemoryManager::completion_queue_wait_front(...)
tt::tt_metal::buffer_dispatch::copy_completion_queue_data_into_user_space(...)
tt::tt_metal::distributed::FDMeshCommandQueue::copy_buffer_data_to_user_space(...)
tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue()
```

**py-spy** MainThread Python stack (server still mid-request, not idling between requests):
```
__torch_function__ (tt_torch/torch_overrides.py:149)
sample_tokens (vllm_tt/model_runner.py:2010)
decorate_context (torch/utils/_contextlib.py:124)
sample_tokens (vllm_tt/worker.py:346)
run_method (vllm/v1/serial_utils.py:510)
collective_rpc (vllm/v1/executor/uniproc_executor.py:97)
sample_tokens (vllm/v1/executor/uniproc_executor.py:125)
step_with_batch_queue (vllm/v1/engine/core.py:515)
_process_engine_step (vllm/v1/engine/core.py:1238)
run_busy_loop (vllm/v1/engine/core.py:1199)
```
i.e. blocked deep inside a live `sample_tokens` call, waiting on the device to return a sampled token — not in vLLM's outer idle loop between steps.

**Device holders**: only the hung PID holds `/dev/tenstorrent/0` — no other process contending for the device at time of hang.

## Comparison against #4521 (prior vLLM scheduler deadlock)

Per [tt-inference-server#4521 comment (2026-07-16)](https://github.com/tenstorrent/tt-inference-server/issues/4521#issuecomment-4993488762), that investigation *also* found a thread sitting in `FDMeshCommandQueue::read_completion_queue` in every gdb capture, and initially suspected a device-read stall — but the actual root cause turned out to be a pure Python scheduler deadlock in `AscendScheduler` (partial-prefill request IDs never cleared from `scheduled_req_ids`, permanently blocking decode admission). In that case the completion-queue thread was healthy and idle, just waiting for work the scheduler had stopped producing, and the MainThread was spinning harmlessly in vLLM core's own empty-schedule idle loop.

Current signature differs in two concrete ways that argue against a repeat of #4521:

| | #4521 (scheduler deadlock) | This hang |
|---|---|---|
| Completion-queue thread CPU | idle/blocked, ~0% | **actively spinning**, ~100% of one core |
| MainThread Python stack | outer idle loop (`run_busy_loop`), empty schedule | **inside a live `sample_tokens` call**, mid-step |
| Root cause location | Python scheduler bookkeeping | native code, no scheduler frames in either stack |

Confirmed the #4521 fix is present in this tt-xla-2 checkout (not a regression of the old bug): `ascend_scheduler.py` has the partial-prefill `scheduled_req_ids`/`running` split described in the fix, and the regression test `tests/integrations/vllm_plugin/test_ascend_scheduler_deadlock.py` exists.

## BREAKTHROUGH (2026-07-29): resident graph count/DRAM footprint alone reproduces the hang — decoy graphs that are *never invoked* are sufficient

The single most decisive result in the investigation. Ran `TTXLA_WARMUP_GREEDY_ONLY=1 TTXLA_WARMUP_NO_GRAMMAR=1 TTXLA_WARMUP_DECOY_PADDINGS=1152,1280,1408,1536,1664,1792,1920,2048,2176,2304,2432,2560` (12 decoy token buckets, all forced to compile at `num_reqs=1` only — see the `TTXLA_WARMUP_DECOY_PADDINGS`/`_decoy_num_tokens` fix above) — **63 total compiled graphs**, past baseline's 54-graph threshold. Warmup succeeded cleanly (541.94s, no OOM). Evals (`ifeval@0.25` → `gpqa@1.0`) hung at the exact same transition point as every prior occurrence, with the **identical native signature**:

```
completion_queue_wait_front → read_cq_host_ptr → is_dram_backed
FDMeshCommandQueue::read_completion_queue / copy_buffer_data_to_user_space
```

**Verified the decoy buckets were never touched by real traffic** before drawing conclusions: grepped the full server log for any `[SAMPLING-FLAGS]`/`[NEW-REQUEST]` line referencing any decoy bucket size (1152-2560) — zero matches. A real 1791-token gpqa prompt's prefill correctly chunked into the existing `1024` bucket (`input_ids_shape=(1, 1024)`), never touching the adjacent `1792` decoy. The decoy graphs sat compiled and completely inert in DRAM the entire run.

**This means the hang requires only resident graph count/DRAM footprint — not any specific feature path, not b1-prefill, not bucket diversity being exercised, not an invocation/switching sequence.** The b1-prefill and bucket-count correlations found earlier were real but incidental: they happened to be knobs that reduce total resident graph count, and *any* reduction below some threshold avoids the hang, while pushing count back up — even with graphs that are provably dead weight — brings it back. This also resolves the "same graphs throughout, so why does gpqa differ from ifeval" puzzle from earlier: it isn't about *which* graphs get exercised at all — it's that gpqa's admission burst (longer generations, higher sustained KV cache usage) is where the cumulative DRAM pressure from many resident graphs (real or decoy) tips over into whatever completion-queue race/corruption exists, regardless of whether those resident graphs are ever used.

**Automated capture**: this run was driven unattended via a background orchestrator script (reset device → launch server → poll for warmup → run evals → auto-capture py-spy/gdb/tt-smi evidence on the first sustained 0.0 tok/s stall) — logs under `debug_logs/decoy_paddings_big/`. The automated gdb thread-selection had a minor bug (grabbed the MainThread's `DeviceLocker::Barrier` frame instead of the async completion-queue thread's, due to `info threads` grep matching multiple lines) but the MainThread signature alone was already conclusive, and a manual follow-up (`gdb -p <pid> -batch -ex "thread 48" -ex "bt"`) confirmed the full `completion_queue_wait_front` chain independently.

**Next steps this unlocks**:
- `🔲` Find the actual graph-count threshold precisely — bisect between 33 (clean) and 63 (hangs) using decoy paddings alone (e.g. try 6 decoys ≈ 45 graphs, then 8 or 4 depending on result).
- `🔲` This reframes the root-cause direction: since it's provably about resident DRAM footprint from const-eval'd graphs, not runtime behavior, the fix likely lives in tt-metal/tt-mlir's DRAM allocator or const-eval storage (per `platform.py`'s own warning about "storing the entire model on device once per graph"), not in vLLM scheduler logic or the TT plugin's request-handling code — worth reporting upstream with this exact reproduction.
- `🔲` A debug build of tt-metal/tt-umd (see section B) would now be higher-value than before — could directly inspect DRAM allocator statistics/fragmentation at the moment of hang to see exactly what's being starved.

## SECOND BREAKTHROUGH (2026-07-29): syncing after every trace execution prevents the hang

Prompted by a suggestion raised in a meeting: try syncing not between every op (impossible during trace capture — a trace, by design, records a sequence of commands without executing/completing them individually) but between every trace *execution* (replay), which remains possible since `enable_trace=True` doesn't prevent syncing around the opaque replay call itself.

**Implementation** (tt-mlir, `runtime/lib/ttnn/program_executor.cpp` / `.h`): cherry-picked commit `6cc62ef5f2` (Kyle Mabee, `kmabee/ops_debug_prints_and_sync` branch, 2026-07-16) which un-gates `syncAfterOpIfNeeded()`/`TT_RUNTIME_SYNC_AFTER_OP` from a debug-only build flag and promotes "Starting execution of program"/"Executing operation"/"Finished execution of program" from `LOG_DEBUG` to unconditional `LOG_INFO`. On top of that, added a **new, independent mechanism**: `syncAfterTraceExecuteIfNeeded()`, gated by a new `TT_RUNTIME_SYNC_AFTER_TRACE` env var, called specifically from the `ExecuteTraceOp` case in `ProgramExecutor::runOperation()` (`program_executor.cpp:643-646`). This is deliberately separate from `TT_RUNTIME_SYNC_AFTER_OP`: capture never reaches the `ExecuteTraceOp` case (only replay does — confirmed via research into the call chain: `capture_or_execute_trace.cpp` → `execute_trace.cpp` → `ttnn::operations::trace::execute_trace` → `MeshDeviceImpl::replay_mesh_trace` → `FDMeshCommandQueue::enqueue_trace`), so this flag syncs after every actual trace replay without forcing a sync after every eager op elsewhere (which would also fire during capture, unlike the trace-specific flag).

**Build note**: for local-only experimentation, applied the diff as *uncommitted* working-tree changes with the tt-mlir submodule left at the exact pinned hash in `third_party/CMakeLists.txt` — no commit, no push, no pin update needed. `cmake --build build`'s update step (`git checkout ${TT_MLIR_VERSION}`) is a safe no-op when already at that ref and doesn't discard uncommitted changes; only *changing* the pin away from a real committed hash would require pushing that hash first (needed only if you want the change to survive a `git checkout` to a *different* ref, e.g. for CI or a real handoff). Confirmed the new symbol landed in the rebuilt `.so` via `strings ... | grep TT_RUNTIME_SYNC_AFTER_TRACE`.

**Result — ran the exact same 63-graph decoy config that reliably hung (see BREAKTHROUGH above) with `TT_RUNTIME_SYNC_AFTER_TRACE=1` added, nothing else changed: 2/2 loops completed clean, no hang** (`debug_logs/decoy_paddings_trace_sync/`, 18m28s total including the extra per-op logging + sync overhead; verified no hidden stall via max-gap analysis — the one large gap, 88s, corresponds to normal steady-state decode with no admission changes, not a stall). This directly contradicts the initial expectation ("I think it likely will [still hang], but worth checking").

**What this means, combined with the graph-count finding above**: resident graph count/DRAM footprint alone is *necessary* (inert decoy graphs reproduce the hang) but this result shows it is **not sufficient on its own** — forcing serialization between trace executions eliminates the hang even at the same DRAM pressure that reliably triggers it otherwise. The cleanest unifying explanation: DRAM pressure from many resident graphs creates the *conditions* for a race (tight buffer reuse/allocation, aggressive eviction, address reuse across trace executions), but the actual corruption/stall requires two trace executions' completion-queue reads or dispatches to be in flight *concurrently* (unsynchronized). Forcing a full device sync after each trace execution serializes them, so no two are ever in flight at once — removing the opportunity for the race regardless of how much DRAM pressure exists. This reframes the likely root cause from "insufficient DRAM headroom" to "a race condition between overlapping/pipelined trace executions under DRAM pressure" — almost certainly in tt-metal/tt-mlir's dispatch or completion-queue/buffer-lifetime management, not in vLLM or the TT plugin.

**Caveat**: only 2 clean iterations so far (vs. 6/6 hangs on the untouched baseline) — strong given the mechanism has a clear causal story, but worth a few more runs for confidence before treating this as fully proven. The performance cost of `TT_RUNTIME_SYNC_AFTER_TRACE=1` (extra full-device syncs plus the very verbose per-op `LOG_INFO` output) makes this unsuitable as a production workaround, but as a diagnostic it strongly narrows where the actual fix needs to happen.

**Live toggle added**: `syncAfterTraceExecuteIfNeeded()` now has a second, independent control — `TT_RUNTIME_SYNC_AFTER_TRACE_FILE=/path/to/file` checks that path's existence on *every* call (cheap relative to the `Synchronize()` it may trigger) rather than once at startup, so sync can be toggled on/off between (or during) experiments via `touch`/`rm`, without restarting the server or repaying the warmup cost. The original `TT_RUNTIME_SYNC_AFTER_TRACE` env var is unchanged (still a startup-set master switch, checked once) — either control being active triggers the sync.

## ROOT CAUSE IDENTIFIED (2026-07-29): trace-eviction/recapture races with in-flight replay because trace buffers share the general DRAM pool, and eviction has zero device-side synchronization

Deep-dive research (verified line-by-line against source, not taken on faith) into tt-mlir/tt-metal's trace-capture, DRAM-allocator, and const-eval internals. This explains *why* graph count matters, *why* the sync fix works, and *why* the two earlier findings aren't in tension with each other.

### The mechanism, end to end

1. **Trace buffers and const-eval results share the same general DRAM pool — confirmed, not just suspected.** tt-metal's allocator only gives trace command buffers a *dedicated, isolated* region if `trace_region_size > 0` (`tt_metal/impl/context/context_descriptor.hpp:45,106` — default `0`). tt-xla's PJRT plugin only sets this if `TT_RUNTIME_TRACE_REGION_SIZE` is exported (`pjrt_implementation/src/api/client_instance.cc:664-671`) — **we never set it in any experiment**, so every server in this whole investigation has been running in "dynamic allocation mode." In that mode, `MeshTrace::populate_mesh_buffer` (`tt_metal/distributed/mesh_trace.cpp:63-70`) explicitly allocates trace buffers as `BufferType::DRAM` — the *exact same pool* used for weights, const-eval results (permanently retained per `platform.py`'s own comment, tracked upstream as [tt-mlir#3888](https://github.com/tenstorrent/tt-mlir/issues/3888)), and activations. This directly confirms the "precompiled graphs infringing on trace buffer DRAM" hypothesis from earlier in this investigation.

2. **Every first-time trace capture invalidates *every other* already-captured trace, unconditionally.** `TraceCache` (`runtime/include/tt/runtime/detail/ttnn/types/trace_cache.h`) keeps one global monotonic `generationId` counter per device. `CaptureOrExecuteTraceOp::run()` (`runtime/lib/ttnn/operations/trace/capture_or_execute_trace.cpp:207-259`, verified directly) calls `traceCache->incrementGeneration()` on **every** cache miss (line 225) — not scoped to the specific trace being captured. Any other trace whose own `generationId` is now stale (`traceData->generationId < traceCache->getGenerationId()`, line 242) gets evicted and recaptured the next time it's needed, regardless of whether the new capture's memory footprint could ever have actually touched it. This is a conservative, correctness-motivated (not performance-motivated) design — the `TraceCache` class comment says outright that trace reuse is "inherently unsafe" after any new capture, since the old trace's captured commands could reference memory the new capture now occupies.
   - **This directly explains the graph-count threshold.** More resident graphs (including inert decoys) means more first-time-capture events keep happening well past warmup, into live serving — each one invalidating every other trace. We directly observed this: `run_and_capture_trace_0_main` (the tt-mlir capture path) fired 2-3 times per server lifetime for essentially every trace, not once — genuine on-device recapture during live serving, for traces used by ifeval, gpqa, and both alike (not gpqa-specific, as first suspected — see below). More graphs → more generation bumps → more recapture churn per unit time → more exposure to the race in the next step.

3. **Trace eviction has a synchronous host-side path with *zero* device-side wait anywhere in it — this is the race.** Traced the full call chain (`TraceCache::erase()` → `ttnn::operations::trace::release_trace()` → `MeshDeviceImpl::release_mesh_trace()` → `SubDeviceManager::release_trace()` — a bare `unordered_map::erase`, `sub_device_manager.cpp:143` — → `~MeshTraceBuffer()` → `~MeshBuffer()` → `Buffer::deallocate_impl()`, `tt_metal/impl/buffers/buffer.cpp:524-559`). The only substantive call in that entire chain is `allocator_->deallocate_buffer(this)` — verified directly: **no `Finish()`, `Synchronize()`, event wait, or fence anywhere in the path.** It's pure host-side address-space bookkeeping. Since tt-metal's command queue is asynchronous (host enqueues and returns immediately; device consumes on its own timeline), the freed DRAM address becomes eligible for reuse **immediately** — by the very next line of code, `runTraceProgramAndCaptureTrace()` for the new capture (`capture_or_execute_trace.cpp:249`) — with no guarantee the device has finished consuming whatever was previously enqueued against that address, including an in-flight replay of a different, still-valid trace. **This is the race**, and it's a known-but-under-tested hazard class upstream: `runtime/test/ttnn/python/n150/test_trace.py::test_trace_memory_overwrite_multi_graph` encodes the exact same scenario in miniature (2 graphs, single loop) and its docstring describes this generation-id mechanism as the intended mitigation — but that test never exercises it under the graph-count/concurrency pressure needed to expose the race, so it passes while our workload hangs.

4. **Why `TT_RUNTIME_SYNC_AFTER_TRACE` works — and why a more surgical fix should be smaller/cheaper.** The sync call only fires from the `ExecuteTraceOp` case (`program_executor.cpp:643-646`), not from anywhere in the erase/recapture path. But replays are so frequent in a serving loop (effectively every decode step, every request) that forcing a full `Synchronize()` after each one ends up serializing nearly the entire op stream as a side effect — including, incidentally, whatever erase()/recapture sequence for some *other* graph happens to be interleaved. **It works by adjacency, not because it's wired to the actual unsafe transition.** The more targeted fix would be a `Synchronize()`/`Finish()` inserted immediately before `traceCache->erase(...)` (`capture_or_execute_trace.cpp:248`) or inside the erase/release/deallocate chain itself — draining only what's actually necessary (in-flight work against the about-to-be-freed trace buffer) rather than draining the whole device after every single replay. Not yet implemented/tested — a good next step.

### Introspection tools discovered (not yet used, but ready)

`ttnn.device.GetMemoryView(device, buffer_type)` (Python-bound in `ttnn/cpp/ttnn-nanobind/device.cpp:462-520`, backed by `tt::tt_metal::detail::MemoryReporter`) returns a live block table (`address`/`size`/`allocated`/`prevID`/`nextID`) for a given `BufferType`, including `BufferType.TRACE` specifically (unlike `DumpDeviceMemoryState`'s CSV dump, which only covers `DRAM`/`L1`/`L1_SMALL`). Calling this immediately before/after a detected recapture event (keyed off the `LOG_DEBUG`/`LOG_INFO` "Trace is stale... recapturing" line) and diffing the block tables would directly show which addresses the evicted trace held and whether the new capture's allocation actually collides with them — the most direct way to catch the overlap in the act, if ever needed as further proof.

### Status / what's confirmed vs. still open

- `✅` Trace buffers and const-eval results share one DRAM pool (verified via `mesh_trace.cpp`/`context_descriptor.hpp`/`load_cached.cpp`).
- `✅` Generation-id invalidation is global and unconditional on any first capture (verified via `capture_or_execute_trace.cpp`).
- `✅` Eviction path has no device-side sync anywhere (verified via `trace_cache.cpp` → `sub_device_manager.cpp` → `buffer.cpp`).
- `✅` This is a known-but-lightly-tested hazard class upstream (existing `test_trace_memory_overwrite_multi_graph`, tt-mlir#3888 for the OOM framing — no existing issue for the race/hang framing found).
- `⏳` `GetMemoryView`-style collision capture (overnight session, 2026-07-29) — added lightweight `TT_RUNTIME_TRACE_ALLOC_DEBUG=1` instrumentation (raw `fprintf(stderr,...)`, bypassing the tt-logger level gate to avoid the verbose-logging confound) at three points: `Buffer::allocate_impl`/`deallocate_impl` (`tt_metal/impl/buffers/buffer.cpp`, logs `type/dev/addr/size/unique_id/ts` for every DRAM/TRACE buffer) and around `TraceCache`'s stale-recapture branch (`capture_or_execute_trace.cpp`, logs `EVICT_TRACE traceId=.../genAtCapture=.../genNow=...` and `RECAPTURE_DONE`). Caught a live hang with this active (occurrence — see below): **75 total EVICT_TRACE events over the run, with the last 5 (distinct trace IDs 12, 85, 78, 98, 91) all clustered immediately before the freeze, all at generation 53** — direct confirmation of the "many different trace evictions/recaptures in a tight window right before the freeze" pattern that was previously only inferred from bucket-cycling timing. Also observed the same DRAM address (`2781755008`) being freed and immediately reallocated (varying sizes 6MB/3MB/1.5MB, consistent with different prefill-bucket activation sizes) dozens of times in the seconds before the freeze, with nanosecond-scale gaps between each dealloc and the next alloc at that address — direct empirical evidence of the aggressive, unsynchronized address-reuse behavior central to the theory. **Limitation**: the `EVICT_TRACE` log point (inside `capture_or_execute_trace.cpp`, before the actual buffer deallocation happens deeper in the call stack) doesn't yet carry the freed buffer's address, so this doesn't (yet) prove a literal address overlap between an evicted trace's old buffer and a new allocation — it proves tight temporal clustering of eviction events immediately pre-freeze, and aggressive address churn generally, but not a byte-for-byte address match between the two. Threading the actual freed address through from `Buffer::deallocate_impl` to the `EVICT_TRACE` log line (or just correlating by nearest-in-time `DEALLOC`/`ALLOC` pairs post-hoc) is the natural next step to close this gap fully.
- `❌` The more surgical fix (sync only around the erase/recapture transition, not after every replay) — **tried (`TT_RUNTIME_SYNC_BEFORE_TRACE_RELEASE`), hung 2/2**. Confirmed NOT effective. Combined with the `TT_RUNTIME_TRACE_REGION_SIZE` result below, this refines the mechanism away from "DRAM address reuse specifically at eviction time" and toward "two trace replays in flight concurrently, eviction or not" — see occurrence #14 below for the full reasoning. `TT_RUNTIME_SYNC_AFTER_TRACE` remains the only proven-effective intervention, and it's also the only one that unconditionally prevents any two replays from overlapping.
- `❌` `TT_RUNTIME_TRACE_REGION_SIZE` (overnight session, 2026-07-29) — **ruled out as a workaround**. First attempt (4GiB) OOM'd during serving (too large a fixed carve-out); retried at 1GiB (clean compile, no OOM) — **hang still reproduced**, identical signature. Refines the mechanism: isolating trace buffers from weights/const-eval doesn't help because the zero-sync eviction race is trace-vs-trace, not trace-vs-other-pool. See the dedicated section below.

## Sizing / DRAM-budget breakdown (overnight session, 2026-07-29)

Concrete numbers, derived from live logs rather than guessed, to make the mechanism above less abstract:

- **Board**: P150 (single Wormhole-class chip), `device DRAM = 31.88 GiB` (logged directly by `vllm_tt.worker`), **8 DRAM banks**.
- **Baseline (`trace_region_size=0`) bank size**: `4272341376 B/bank` = **4074.75 MiB/bank** — derived two ways that agree: (a) the RMSNorm-fusion issue's independent OOM message on the same board class quotes this exact figure; (b) back-calculated from the 4GiB-trace-region OOM below (`3735470464 + 4294967296/8 = 4272341376`, exact).
- **KV cache budget** (`gpu_memory_utilization=0.35`): `11.16 GiB` total = **1428.5 MiB/bank** reserved.
- **Checkpoint size**: `13.89 GiB` on disk (bf16). On-device weight footprint at `bfp_bf8` is not logged directly, but bfp8 is roughly half of bf16's byte width (block-exponent 8-bit format vs. 2-byte float) — a rough estimate is **~7-9 GiB on device** (~900-1150 MiB/bank), not yet directly measured.
- **Remaining shared pool for weights + const-eval + 54 trace command buffers + transient activations**: `4074.75 − 1428.5 ≈ 2646 MiB/bank`, of which weights consume an estimated ~1000 MiB/bank, leaving roughly **~1600 MiB/bank** for const-eval + all 54 resident trace buffers + compile/activation transients — this is the tight margin the root-cause section's "razor-thin DRAM margin" observations are describing concretely.

### What the `TT_RUNTIME_TRACE_REGION_SIZE=4GiB` OOM teaches us about actual trace-buffer size

Carving out a **fixed 4 GiB** dedicated trace region (512 MiB/bank) — intended to isolate trace buffers from the shared pool per the root-cause theory — instead **crashed with a DRAM OOM during live serving** (not at warmup): `bank size is 3735470464 B ... allocated: 3415259520 B, free: 320210944 B, largest free block: 165134720 B` — i.e. only ~305 MiB/bank truly free, and a 188.7 MiB/bank allocation request didn't fit.

This is informative rather than just a failed experiment: **the untouched baseline (trace buffers sharing the general pool) never OOMs** — every one of its 8+ documented hangs occurs with the device otherwise healthy, no OOM anywhere. If the real per-bank trace-buffer footprint for 54 graphs were close to the 512 MiB/bank we reserved, isolating it should have been roughly neutral (same total consumption, just partitioned differently). Instead it made things *strictly worse*. The most likely explanation: **a `trace_region_size` reservation is a hard, non-reclaimable carve-out** — even if a given trace is released/evicted and its footprint shrinks, that freed space stays inside the trace region and can never be lent back to weights/activations, unlike the shared-pool baseline where trace buffers and everything else commingle and reclaim each other's freed space fluidly. An oversized fixed reservation therefore starves the shared pool faster than 54 graphs' *actual* (and here, clearly much smaller than 512 MiB/bank) footprint ever would sharing the pool dynamically.

### `TT_RUNTIME_TRACE_REGION_SIZE=1GiB` retry — compiles clean, but **the hang still reproduces** — major result, refines the root cause

Retried with a much smaller dedicated region (1 GiB total = 128 MiB/bank). This time: **warmup compiled clean, no OOM** (465.56s, matching baseline timing almost exactly). Ran the standalone eval loop (`--ifeval-limit 0.10 --gpqa-limit 100000`) against it — **hung on iteration 1/8, at the identical ifeval→gpqa transition**, identical signature: CPU-delta sweep found one thread (LWP 193531) spinning at ~100% (308 ticks/3s), `gdb` confirmed the same `completion_queue_wait_front → copy_completion_queue_data_into_user_space → copy_buffer_data_to_user_space → read_completion_queue` chain (outermost frame this capture `MetalContext::get_cluster()`), `tt-smi` heartbeat still ticking, RSS unremarkable.

**This meaningfully refines, rather than falsifies, the root-cause mechanism.** `TT_RUNTIME_TRACE_REGION_SIZE>0` isolates trace buffers from *non-trace* buffers (weights/const-eval/activations) — it does **not**, and structurally cannot, isolate one trace buffer from *another* trace buffer. The zero-device-sync eviction bug in `Buffer::deallocate_impl()`/`release_trace()` (see ROOT CAUSE section above) applies identically regardless of which pool a buffer lives in: with 54 traces now confined to their own 1 GiB region, trace A's freed slot can still be immediately reused by newly-recaptured trace C while trace B's non-blocking replay has work in flight nearby — the exact same race, just entirely *within* the isolated trace region instead of against the shared pool.

**Updated verdict**: the "trace buffers share a pool with weights/const-eval" framing is real and verified via source, but **is not the necessary ingredient** — it was never really about *which* pool, it's that *any* pool containing multiple trace buffers subject to the same zero-sync eviction path is sufficient. This sharpens the fix target: **isolating the pool (`TT_RUNTIME_TRACE_REGION_SIZE`) is not a viable workaround** (confirmed not to prevent the hang); the fix has to be at the actual eviction/sync transition (the surgical `TT_RUNTIME_SYNC_BEFORE_TRACE_RELEASE` patch, next) or the blanket `TT_RUNTIME_SYNC_AFTER_TRACE` workaround, not a memory-layout change.

### gpqa-only from a fresh server (skip ifeval): clean — suggests ifeval-first is a real precondition, not just gpqa's own burst pattern

Ran the untouched baseline (verbose logging suppressed via `TTMLIR_RUNTIME_LOGGER_LEVEL=WARNING`, to avoid the same confound as above) with `--tasks gpqa` only, full 198-doc dataset, against a **freshly-warmed server that never saw ifeval traffic** (`debug_logs/gpqa_only_fresh/`). **Completed clean — no hang.** Verified no hidden stall (max gap 99.9s, but that's the eval client's own startup latency — dataset load + model init — between warmup-complete and the first real request, not a server stall).

(First attempt at this test was invalid and had to be redone: `--gpqa-limit 1.0` passed directly to the standalone script actually ran only **1** document, not the full 198 — `lm_eval`'s own `get_sample_size()` does `int(math.ceil(n*limit)) if limit < 1.0 else int(limit)`, so exactly `1.0` hits the `int(limit)` branch = 1 document, not "100%". This only affects this one raw invocation; every earlier "gpqa@1.0" run in this investigation went through `run_evals_forge.sh`/`run.py`'s own `eval_config.py`-driven limit handling, a different, unaffected code path. Fixed by passing a large integer, e.g. `--gpqa-limit 100000`, instead.)

**This is a real, if `n=1`, meaningful result**: every prior occurrence of the hang followed the exact ifeval-then-gpqa sequence (6/6 hangs); gpqa alone against a fresh server did not reproduce it. This is fully consistent with the race-condition mechanism above rather than contradicting it: DRAM allocator fragmentation state is inherently history-dependent (a first-fit/best-fit allocator's layout depends on the exact sequence of prior allocations/deallocations), so "ifeval's admission pattern, then gpqa's burst" and "gpqa's burst alone against a pristine layout" can plausibly produce genuinely different fragmentation histories — only one of which happens to expose the specific address-reuse-before-drain race. Worth at least one more repetition before treating "ifeval-first is necessary" as settled, given n=1 — but this is a good, concrete lead on exactly *which* allocation history matters, not just that resident graph count in the abstract matters.

### Confidence assessment (2026-07-29)

**Very likely correct and specific enough to act on — but not yet fully proven.** Calibration, so a future session picking this back up knows exactly what's solid vs. still open:

**High confidence (verified directly against source, not inferred from a research pass):**
- Trace buffers and const-eval results genuinely share one DRAM pool (`trace_region_size=0` default, never overridden by us).
- Generation-id invalidation is global and unconditional — any first-time capture invalidates every other captured trace, regardless of overlap.
- The eviction path (`erase → release_trace → deallocate_buffer`) has zero device-side synchronization anywhere in it.

**Moderate-high confidence (strong causal experiments, not yet a smoking gun):**
- Resident graph count/DRAM footprint causally matters — proven via inert decoy graphs (compiled, never invoked by real traffic) reproducing the hang. Clean, controlled result.
- Forcing sync after every trace replay prevents it at the same graph count — but only 2 clean iterations vs. 6/6 baseline hangs.

**What's still missing before calling this fully proven:**
1. No direct observation of the actual race — haven't used `GetMemoryView` to catch a live address collision between an evicted trace's freed buffer and a subsequent allocation.
2. Haven't tried the surgical fix (sync only at the erase/recapture transition, not after every replay). If that alone fixes it too, it confirms the *right* transition was found, rather than a synchronization point that works via serialization side-effects.
3. **Real confound**: verbose logging alone (no explicit sync) also masked the hang once — confirms this is genuinely probabilistic/timing-sensitive, not deterministic. So any single "ran clean" result (including the `n=1` gpqa-only test above) is weaker evidence than it looks; can't fully rule out some clean runs just got lucky rather than being structurally safe.
4. One unexcluded alternative: the completion-queue-hang signature is *consistent* with a DRAM address-reuse race, but the exact causal link from "address collision" → "completion queue never posts" hasn't been demonstrated. Some other capacity-constrained resource with similar dynamics (command-queue depth, event/semaphore exhaustion) isn't strictly ruled out, just less parsimonious given everything else fits.

**Highest-value remaining experiments to close the gap**: `TT_RUNTIME_TRACE_REGION_SIZE` set nonzero (isolates the shared-pool half specifically, no sync changes needed), and the surgical erase-time-sync fix (isolates the "right transition" half). Good enough to write up as the leading hypothesis in an upstream issue now; not yet good enough to close the investigation.

### Important confound discovered: verbose per-op logging alone also masks the hang

Ran the **untouched baseline** (no decoys, no `TT_RUNTIME_SYNC_AFTER_TRACE`, nothing changed except the now-unconditional `LOG_INFO` per-op logging from the rebuild) — `debug_logs/baseline_verbose_log/`. **It completed clean, no hang**, despite this being the exact config that hung 6/6 times before this rebuild.

Recapture-churn data from this run: 53 distinct trace programs, individual traces recaptured **1-5x** each over the run (`trace_18/24/29/37/41/47` all hit 5x — *more* recapture churn than the 63-graph decoy run that did hang, which topped out at 2-3x per trace). So recapture frequency alone doesn't cleanly predict hang/no-hang either.

**This is a real confound, not a new finding of "logging prevents the hang."** The most likely explanation is the same mechanism as the sync fix: writing a full MLIR op string to stdout for every single op adds real per-op latency, which — like `TT_RUNTIME_SYNC_AFTER_TRACE` — changes timing enough to avoid the race window, without being a real fix. This is classic race-condition behavior (a "heisenbug" — instrumentation that adds delay tends to hide timing-sensitive bugs) and is itself indirect supporting evidence that this is genuinely a timing-sensitive race, not a deterministic function of graph count or recapture count alone. **Practical implication**: any future experiment using this verbose logging to inspect trace/recapture behavior needs `TTMLIR_RUNTIME_LOGGER_LEVEL=WARNING` (suppresses the `LOG_INFO` per-op prints while keeping the underlying mechanism) if the goal is to also observe whether the hang itself reproduces cleanly — otherwise the observation itself perturbs the result.

## Running list: debug ideas, knobs, and time-to-repro improvements

Living checklist — update status inline as items get tried rather than adding new prose sections. `🔲` not yet tried · `⏳` tried, inconclusive/partial · `✅` done/resolved · `❌` ruled out.

### A. Root-cause isolation (config variants to run against the hang)

- `✅` `enable_trace=False` — 3/3 clean, no hang (Variant 1). Strongest single lever so far.
- `⏳` `cpu_sampling=True` (default GMU=0.35) — hit a *different* bug (DRAM OOM at the ifeval→gpqa transition), not hang-conclusive either way.
- `⏳` `cpu_sampling=True` + `GPU_MEMORY_UTILIZATION=0.25` — OOM avoided, 1 clean iteration; never pushed further.
- `❌` baseline (`cpu_sampling=False`, `enable_trace=True`) + `GPU_MEMORY_UTILIZATION=0.25` (Test B) — hang still occurred. Rules out "DRAM tightness alone" as the trigger.
- `🔲` `max_model_len=4K` + chunked prefill disabled (Variant 3) — script bug (`PREFILL_CHUNK_SIZE:-` vs `-`) fixed, but never re-launched with the fix. Still owed.
- `✅` **`MIN_NUM_SEQS=32 PREFILL_BATCH_THRESHOLD=0 MIN_CONTEXT_LEN=512`** (disables the b1-prefill serial-single-request admission path — `platform.py:71-75`, `model_runner.py:2578-2580`) — **ran clean, first time ever.** Both `ifeval` and `gpqa_diamond_generative_n_shot` completed via the real `run.py` eval client (~43 min total, `falcon3_evals_baseline_trim_flags_ds_0.5_1.0.log`: `✅ llm_eval blocks=2 (2582.2s)`), zero freeze (max gap between `[SAMPLING-FLAGS]` steps: 31.5s the whole run, log: `falcon_serve_full_model_trim_flags.log`). Strongest single result in the whole investigation — every prior baseline (b1-prefill *on*) run through this exact transition has hung (occurrences 4, 5, 6); this is the first time it didn't. **Caveat**: `MIN_CONTEXT_LEN=512` was changed at the same time, so this doesn't cleanly isolate b1-prefill from the token-ladder change yet — see the follow-up below. One clean run also isn't proof at n=1; worth a repeat if time allows.
- `✅` **Follow-up to isolate which change actually fixed it**: ran `PREFILL_CHUNK_SIZE=256` with b1-prefill left **on** (`MIN_NUM_SEQS=1 PREFILL_BATCH_THRESHOLD=16`, unchanged defaults) — **also ran clean** (warmup 170.73s, ifeval ~4min at normal pace, `run_evals_forge.sh` completed both tasks; 2nd confirmation run in progress as of 2026-07-28). **This weakens the b1-prefill theory**: b1-prefill was ON here (the suspected trigger, matching every hung baseline), yet it still didn't hang. b1-prefill's on/off state does not track with the hang/no-hang outcome across the two non-baseline runs so far (one had it off, one had it on — both clean).
- `✅` **Missing-cell test**: `MIN_NUM_SEQS=32 PREFILL_BATCH_THRESHOLD=0` **alone** (full original 4-bucket ladder `[1,128,256,512,1024]` intact, `MIN_CONTEXT_LEN`/`PREFILL_CHUNK_SIZE` untouched) — **ran clean** (`run_evals_forge.sh`: 7min ifeval + 25min gpqa, no freeze). b1-prefill-off alone, with full bucket diversity preserved, was sufficient to avoid the hang.

**AND-gate theory — falsified 2026-07-28.** It fit 4 data points but broke on the 5th: `TTXLA_WARMUP_GREEDY_ONLY=1 TTXLA_WARMUP_NO_GRAMMAR=1 DISABLE_PREFIX_CACHING=1` (b1-prefill left **ON**, full 4-bucket ladder left **intact** — both AND-gate variables identical to baseline) ran clean 3x. The AND-gate predicted this should hang. It didn't.

**Current leading theory — total compiled/const-eval graph count (equivalently, resident DRAM trace footprint). Fits every data point without exception:**

| Config | Compiled graphs | Warmup (s) | Result |
|---|---|---|---|
| **Baseline** (untouched) | **54** | **464.84** | **HANGS, 6/6** |
| `MIN_NUM_SEQS=32 THRESH=0` alone | 30 | — | Clean |
| `MIN_NUM_SEQS=32 THRESH=0 MIN_CONTEXT_LEN=512` | 18 | 312.55 | Clean |
| `PREFILL_CHUNK_SIZE=256` | 30 | 170.73 | Clean |
| `TTXLA_WARMUP_GREEDY_ONLY TTXLA_WARMUP_NO_GRAMMAR DISABLE_PREFIX_CACHING` | 27 | 252.97 | Clean, 3x |
| `TTXLA_WARMUP_GREEDY_ONLY TTXLA_WARMUP_NO_GRAMMAR` alone (no prefix-cache change) | 27 | 255.01 | Clean, 5x |
| `TTXLA_WARMUP_NO_GRAMMAR` alone (no `GREEDY_ONLY`) | 36 | 324.69 | Clean, 3x |

Every config with fewer than ~54 compiled graphs has been clean, regardless of *which* axis (num_reqs, token buckets, greedy/grammar variants) got cut to achieve the reduction; the only configs that hit the full 54 (untouched baseline) have hung, 6/6. Ties back to `platform.py`'s `enable_const_eval` comment: const-eval graphs are stored on device *permanently* ("essentially storing the entire model on device once per graph") — more resident graphs likely means more permanent DRAM footprint/fragmentation, and 54 may just sit at whatever edge trips the completion-queue stall under sustained multi-bucket-cycling load. Any reduction, via any axis, gives enough margin to avoid it.

**`DISABLE_PREFIX_CACHING` confirmed a non-factor** — identical graph count (27) and warmup (252.97s vs 255.01s) with or without it, clean either way (3x and 5x respectively). Prefix caching disable never mattered; graph-count reduction alone did all the work. Matches the earlier prediction from its low 2.6-2.7% observed hit rate.

- `✅` `TTXLA_WARMUP_GREEDY_ONLY=1 TTXLA_WARMUP_NO_GRAMMAR=1` alone (no `DISABLE_PREFIX_CACHING`) — ran clean 5x at ifeval `0.25`/gpqa `1.0`. Confirms graph-count reduction alone is sufficient; prefix caching is a non-factor (see above).
- `✅` `TTXLA_WARMUP_NO_GRAMMAR` alone (no `GREEDY_ONLY`) — 36 graphs, ran clean 3x. Narrows the danger zone: hang requires somewhere in **(36, 54]** graphs, not just "any reduction from 54." Every test below 54 has been clean regardless of which axis did the cutting (18, 27, 30, 36 all clean; only the untouched 54 hangs).
- `🔲` **Next**: complete the mirror-image test — `TTXLA_WARMUP_GREEDY_ONLY` alone (no `NO_GRAMMAR`) — to see where that lands (likely similar ballpark to 36, given the two axes are roughly symmetric) and finish characterizing the greedy/grammar family before switching axes.
- `🔲` **Then**: a data point from a *different* axis landing in the same (36,54] danger zone would meaningfully sharpen this — e.g. `MIN_CONTEXT_LEN=256` alone (untouched greedy/grammar, ladder becomes `[1,256,512,1024]`, 3 buckets instead of 4) should land somewhere in the 40s. If that's also clean, it further supports "any reduction below 54 helps, regardless of axis"; if it hangs, the specific axis (not just total count) matters after all, and the graph-count theory needs refining to "which graphs" rather than "how many."

**Refinement (2026-07-28): "same graphs throughout" (confirmed — zero new compiles during serving) doesn't mean "same *invocation sequence*."** ifeval's prompts are short/uniform (~60-150 tokens) — almost certainly repeatedly hitting the same 1-2 small buckets the whole task, a narrow/repetitive access pattern. gpqa's prompts are long/variable (2637-9993 chars) — every hang capture shows its admission burst rapidly alternating between 3-4 *different* buckets for consecutive single-doc admissions. Same compiled artifacts the whole run, but gpqa is the only point that exercises certain trace regions in that specific rapid-switching sequence. This reconciles the `enable_trace=False`-avoids-it finding with "no recompiles happen": if trace buffers for different buckets sit in DRAM regions that are adjacent/reused/misassumed-size, a narrow repetitive pattern (ifeval) would never expose an overlap/allocation race; rapid switching between many distinct trace regions (gpqa) creates far more chances to hit it.

- `🔲` **The decoy-padding experiment (`TTXLA_WARMUP_DECOY_PADDINGS`) can distinguish two versions of the graph-count theory, not just confirm/deny one**: decoys are compiled (real DRAM footprint, real const-eval treatment) but **never invoked** — chunked prefill caps every real step at ≤1024 tokens, so a 2048+-token decoy bucket is structurally unreachable. If decoys alone bring the hang back → static resident DRAM occupancy/footprint is sufficient by itself, independent of any invocation pattern. If they do **not** → static occupancy isn't enough; it's specifically the *invocation/switching sequence* among buckets actually being exercised that matters (consistent with the trace-buffer-overlap idea above) — meaning the next step would need to mimic gpqa's switching pattern using the currently-safe graph set, not just add more inert compiled graphs.
  - `TTXLA_WARMUP_DECOY_PADDINGS` at large sizes (2048-32768) hit a **known, expected** DRAM OOM during warmup (`bank_manager.cpp:462`, ~10GiB matmul-output allocation attempt) — not a new finding, just recreating exactly the scenario chunked prefill (`prefill_chunk_size=1024`) exists to avoid. Use modest sizes (e.g. `1536,2048`) instead.
  - `❌` **`TTXLA_WARMUP_DECOY_NUM_REQS` (the complementary decoy axis, inflating `num_reqs` instead of `num_tokens`) is broken — found a genuine, previously-unknown, unrelated bug**, not a DRAM issue: compiling `_model_prefill` at `num_reqs=16` (any decoy batch size strictly between `min_num_reqs=1` and `max_num_reqs=32` — production code never compiles at anything else) fails with `loc("custom-call.79"): error: 'ttir.paged_fill_cache' op Batch index tensor must have dim 0 equal to input batch (16), got 32`. The warmup path's dummy-input construction for the KV-cache batch-index tensor (feeding `paged_fill_cache`, which routes computed KV values to the correct per-request cache blocks) appears hardcoded to `dim0=32` regardless of the actual requested `num_reqs` — never triggered before since real configs only ever use `num_reqs ∈ {1, 32}`. **This is a real bug worth its own report, but it's a compile-time TTIR shape-verifier error, categorically different from the runtime completion-queue hang under investigation — not evidence either way about the hang itself.** Abandoned this decoy axis; use `TTXLA_WARMUP_DECOY_PADDINGS` (modest sizes) instead to inflate graph count.
  - `⏳` **`TTXLA_WARMUP_DECOY_PADDINGS` also OOM'd at first** — `1536,2048` at the *full batch size* (`num_reqs=32`) needed 2.8GB with only 411MB free. Confirms the DRAM margin is razor-thin: real `1024×32` already consumes nearly all available headroom, so even a modest 2x token-count increase at max batch overflows it. **Fixed** (`model_runner.py:462-476`, `2696-2698`): decoy token buckets now compile at `num_reqs=1` only (~32x cheaper transient footprint, still fully unreachable by real traffic) via a new `self._decoy_num_tokens` filter on the warmup `configs` list. With that fix, `TTXLA_WARMUP_DECOY_PADDINGS=1536,2048` (alongside `GREEDY_ONLY`/`NO_GRAMMAR`) compiled clean — 33 graphs (up from 27), warmup 304.89s, no OOM — and ran clean through **2 full loops** of ifeval@0.25/gpqa@1.0 (`debug_logs/decoy_paddings_test/`, 9m52s total, max gap 21.9s across 2131 `[SAMPLING-FLAGS]` lines). Still below the established 36-graph clean ceiling though, so not decisive alone. Pushing a much larger decoy set (still `num_reqs=1`-only, so no OOM risk) to get near/past the 54-graph danger zone next — in progress (2026-07-29), `debug_logs/decoy_paddings_big/`.
- `✅` **Re-confirm the untouched baseline still hangs reliably** — reproduced again at ifeval downsampled to `0.25` (136 docs, 2m33s) and again at `0.10` (55 docs, 1m30s). Baseline reproducibility holds at 6/6 now; loop is much faster (ifeval down to ~1.5min from ~4-11min). **Correction**: the client always *displaying* `17/198` at the point it goes silent is **not** a genuine "always the same request count" signal — see the buffered-client-output finding below. Ignore the specific number; it's an artifact of buffering, not a real stall point.
- `✅` **`17/198` was a stale/buffered display artifact, not the true count — confirmed via server-side counting.** The eval client's tqdm progress line is delayed reaching the log file (same buffering mechanism first spotted in occurrence 6's 29-minute-delayed `TimeoutError`), so whatever's last visible when it "goes silent" is not the true progress. **Reliable check**: count `POST /v1/completions 200 OK` in the *server's* log after the eval-transition point — uvicorn writes this synchronously when a response is actually sent, unaffected by the client's buffering. Done for the `ds_0.10` run (`falcon_serve_full_model_debug_scheduler_prints_more2.log`): split at line 750 (gpqa's first `[NEW-REQUEST]`) → 56 completions before (ifeval), **28 after (gpqa)** — not the displayed `17`. No invasive action needed (no killing the client, no waiting ~30min for its timeout) — just grep the server log.
- `🔲` If the isolating test above stays clean: try to find the actual *threshold* — e.g. a config that only mildly reduces graph count (somewhere between 54 and 30) to see whether risk scales gradually or there's a sharper cliff. The old "3-bucket intermediate" idea still applies here, just reinterpreted under the graph-count theory instead of the falsified AND-gate.
- `🔲` Revert tt-xla commit `08fbdbe7706eb05dd38623f47fcb0f9f3ccf37da` (restores the scheduler debug logging added in `44eeef909`) and run in a hang-reproducing config — explicitly planned, never started.
- `🔲` `--tasks gpqa` alone (skip ifeval entirely) — tests whether ifeval-first is a necessary precondition or whether a cold gpqa-only burst hangs just as readily. Also a time-to-repro win (see section C).
- `🔲` `prefill_kv_watermark` tuned away from default (0.25) — b1-prefill and the KV-watermark admission throttle both live in `AscendScheduler`; worth ruling in/out an interaction between the two.

### B. Diagnostic technique / tooling ideas (apply on the next live catch)

- `✅` `py-spy dump --locals` (all threads) — routinely useful, safe. Gets Python-level scheduler/engine state (`scheduler_output`, batch-queue locals) directly, no need to guess from log timing alone.
- `✅` `gdb -p <pid> -batch -ex "thread apply all bt"` + a CPU-delta sweep across `/proc/<pid>/task/*/stat` — safe, reliably finds the one actively-spinning thread among 100+ idle ones and its native stack.
- `✅` `tt-smi -s` telemetry (read-only, no Inspector/tt-triage RPC) — safe, confirms chip firmware alive/ticking (`TIMER_HEARTBEAT` advancing) independent of the completion-queue stall. Good cheap sanity check on every future catch.
- `❌` `tt-triage` against a live/hung server — **do not use**, confirmed to crash the host (see `TT_TRIAGE_NOTES.md`). Only exception would be with the documented guardrails, which still only delay (not prevent) the crash — treat as unsafe regardless.
- `❌` gdb's CPython pretty-printer (`py-bt`) — not available for this interpreter build (`Undefined command: "py-bt"`); `py-spy --locals` is the working substitute.
- `🔲` **Per-op sync + debug logging** (#4521 methodology: disable trace, force a full sync + log after every device op) — most direct way to identify exactly which dispatched op's completion never arrives, rather than inferring from timing/bucket correlations. Not yet run.
- `🔲` **Debug build of tt-metal/tt-umd** (with `-g` symbols) — would unlock `info args`/`info locals` inside the native `completion_queue_wait_front`/`read_from_sysmem` frames (currently `No symbol table info available`), potentially exposing the actual queue read/write cursors and in-flight command IDs. Heavier lift; consider if per-op logging doesn't pin it down.
- `🔲` Re-check whether occurrences 4 and 5's already-captured evidence also shows single-request (`num_reqs=1`) prefill activity right before their freeze, same as occurrence 6 — strengthens (or weakens) the b1-prefill lead without needing a new run.
- `🔲` `TTXLA_LOGGER_LEVEL=DEBUG` + `TTXLA_DUMP_SAMPLING_FLAGS=1` **together** on the next catch (occurrence 6 only had the latter; Test B only had the former) — combined gives IR-level compile detail and per-step shape/flag detail on the same timeline.

### C. Time-to-repro / iteration-speed improvements

- `🔲` **New opt-in knobs added 2026-07-28** (safe, zero effect unless set — same pattern as `TTXLA_DUMP_SAMPLING_FLAGS`):
  - `TTXLA_WARMUP_GREEDY_ONLY=1` (`model_runner.py:2572-2577`) — skips precompiling `all_greedy=False`. Both evals are always greedy (confirmed via every `[SAMPLING-FLAGS]` capture), so this branch is pure waste for this workload.
  - `TTXLA_WARMUP_NO_GRAMMAR=1` (`model_runner.py:2578-2582`) — skips precompiling `apply_grammar=True`. Neither eval uses structured output.
  - Together these cut the warmup cartesian product ~4x (2×2) — bigger than any single knob tried so far (`MIN_NUM_SEQS`/`MIN_CONTEXT_LEN`/`PREFILL_CHUNK_SIZE` each gave ~1.5-2x). Composes with whichever bucket/b1-prefill config is under test — orthogonal to the AND-gate variables in section A.
  - `DISABLE_PREFIX_CACHING=1` (`serve_falcon3_7b_forge.sh`) — passes `--no-enable-prefix-caching` to `vllm serve`. **Not a compile-time lever** (vLLM's cross-request KV reuse is pure runtime bookkeeping, not a graph-shape axis) — this one's purely to test whether prefix caching itself affects the hang, not for iteration speed. Belongs conceptually in section A; listed here since it was added alongside the other two.
- `✅` **`[SAMPLING-FLAGS]` trimmed + deduped** (`model_runner.py:1933-1962`) — dropped 8 boolean fields that never varied across the whole investigation (`all_random`, `logprobs`, `no_penalties`, `no_min_tokens`, `no_logit_bias`, `no_bad_words`, `no_allowed_token_ids`, `no_generators`); kept `all_greedy`/`apply_grammar`/`cpu_sampling`. Now only logs on state change instead of every engine step — steady-state decode (same batch, same shape, same step after step) collapses to one line; distinct prefill admissions still each get their own line since real per-request content length (`total_sched_tokens`) differs almost every time even within the same bucket.
- `✅` **New `TTXLA_DUMP_NEW_REQUESTS=1`** (`model_runner.py:899-922`) — one `[NEW-REQUEST] req_id=... prompt_tokens=...` line per brand-new request admission (hooks `_update_states()`'s loop over `scheduler_output.scheduled_new_reqs`), never per decode continuation. Gives a clean per-doc marker independent of the step-level bucket noise, to correlate against the eval client's own progress bar. Also carries that request's actual `SamplingParams` (`temperature`, `top_p`, `top_k`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, `max_tokens`, `min_tokens`, `logprobs`, `structured_outputs`) — the right place for these (per-request properties), unlike the per-step `[SAMPLING-FLAGS]` booleans (aggregate-batch state) they were trimmed out of.


- `✅` `MIN_NUM_SEQS=32 PREFILL_BATCH_THRESHOLD=0 MIN_CONTEXT_LEN=512` — warmup confirmed **312.55s vs 464.84s baseline** (~1.5x, less than the 3-4x originally estimated — correcting that earlier guess) — kills the `num_reqs=1` axis (`platform.py:376-378`) and shrinks the ladder to 3 buckets. **Cost**: noticeably slower ifeval (~4x fewer it/s observed) — *expected*, not a red flag: with b1-prefill off, `num_reqs_options` collapses to `{32}` only, so every single-request replacement admission (the steady-state pattern once the initial burst drains) now runs the full 32-row graph instead of a cheap 1-row one. Acceptable cost here since this run also happened to be the one that didn't hang (see section A) — not the config to use for fast iteration on the *original* bug though (see `PREFILL_CHUNK_SIZE=256` below for that).
- `⏳` `MIN_CONTEXT_LEN=512` — shrinks the token-padding ladder from `[1,128,256,512,1024]` to `[1,512,1024]` (`_get_token_paddings`, `model_runner.py:3938`). Tried alongside the above (2026-07-28): **also slows ifeval** — its tiny prompts (~60-150 tokens) now pad up to 512 instead of 128 (~3.5-8x token-dimension waste per prefill). Secondary contributor to the slowdown vs. the `MIN_NUM_SEQS` effect above.
- `🔲` **Better alternative for fast iteration without hurting ifeval**: keep `MIN_NUM_SEQS=1 PREFILL_BATCH_THRESHOLD=16` (original — b1-prefill *on*) but set `PREFILL_CHUNK_SIZE=256` (down from `1024`). Ladder becomes `_get_token_paddings(min=128, max=256)` = `[1,128,256]` — same bucket-count win (5→3) as `MIN_CONTEXT_LEN=512` gave, but keeps the `128` bucket (ifeval stays cheap) and keeps b1-prefill enabled (single-doc replacement admissions stay cheap). Tradeoff: gpqa's long docs (~1000+ tokens) now chunk into ~4-6 sequential 256-token continuation steps instead of ~1-2 at the 1024 budget — more per-doc dispatch overhead, but not the batch-padding blowup the other two knobs cause. Not yet tried; does **not** test the b1-prefill root-cause theory (b1-prefill stays on) — use this for "reproduce the original bug fast," and the `MIN_NUM_SEQS=32` config above for "test whether disabling b1-prefill changes the hang."
- `❌` `PREFILL_CHUNK_SIZE=0` to kill the `prefix_chunk` compile axis (`_chunked_sdpa_active`, `model_runner.py:433`) — **don't**: this makes `prefill_chunk_budget == max_model_len` (32768), which *extends* the padding ladder instead of shrinking it. Net effect is worse.
- `🔲` `./run_falcon3_7b_evals.sh --ifeval-limit 0.15 --gpqa-limit 1.0` (or lower) — reaches the transition in ~1-2 min instead of ~4-11, without touching gpqa.
- `🔲` `--tasks gpqa` alone — skips ifeval's ~4-11 min entirely (also a root-cause test, see section A).
- `🔲` **Shorten lm_eval's client-side request timeout.** `TemplateAPI.__init__` defaults `timeout: int = 1800` (`lm_eval/models/api_models.py:139`, used as `aiohttp.ClientTimeout(total=self.timeout)`) — this is why occurrence 6's client took **~29 more minutes** after the server froze before logging anything else (a `TimeoutError`/retry burst at 16:35:15, ≈1800s after those particular requests were originally sent — see occurrence 6 for the full explanation). Passing `timeout=60` (or similar) via `--model_args` in `run_falcon3_7b_evals.sh` would make the eval client itself give up on a stuck request in ~1 min instead of ~30, which matters for any automated loop that waits on the *client* to notice a hang — though checking the *server's* own throughput log / py-spy is still far faster (seconds) and doesn't require touching the eval script.

### Resolved / no-longer-open questions (kept for reference)

- Does this reproduce on the full model, not just single-layer? — **Yes**, confirmed 2026-07-28 (occurrences 4-6 all full model).
- Does it require tt-media-server's admission driver? — **No**, reproduces on pure standalone `vllm serve` too (occurrences 5, 6).
- Is it the #4521 scheduler deadlock? — **No**, confirmed independently on four separate live catches (native spin signature + non-empty scheduler state each time).
- Reproducible on demand, or only after long serving? — Reproduces readily once past the ifeval→gpqa transition; not yet seen outside that transition window.

---

## Standalone tt-xla reproducer (2026-07-27, no tt-inference-server in the loop)

Three scripts at the tt-xla repo root reproduce the forge P150 config with only `vllm serve` + `lm_eval`/a raw HTTP client, so a hang can be pinned on the tt-xla vLLM plugin rather than tt-media-server's driver (see `HANDOFF_falcon3_7b_hang_hunt.md` for full background):

- `serve_falcon3_7b_forge.sh` — stock `vllm serve`, config byte-verified against a live tt-media-server run (see below).
- `run_falcon3_7b_evals.sh` / `run_falcon3_7b_evals_loop.sh` — the real ifeval/gpqa eval traffic, loopable N times with per-iteration logs.
- `burst_falcon3_7b_forge.py` — synthetic load driver that reproduces tt-media-server's *admission pattern* specifically (see below), independent of lm_eval.

**Config parity confirmed byte-identical** between a live tt-media-server run (`~/tt-inference-server-2/falcon3_single_layer_server.log`, 2026-07-22) and the standalone `serve_falcon3_7b_forge.sh` run (`~/tt-xla-2/falcon_single_layer_server.log`, 2026-07-27): `additional_config`, the full vLLM `V1 LLM engine` config dump, `GPU KV cache size: 5,849,088 tokens`, `AscendScheduler`, and `Sampling on xla:0 (cpu_sampling=False)` all match exactly (only pids/timestamps/compile-time — 55s vs 105s cache-warmth noise — differ). **Config is not the variable between the two paths.**

### Admission-driver difference: tt-media-server vs stock `vllm serve`

This is the one structural difference that *is* real between the two paths, traced to source (`tt-media-server/device_workers/device_worker_dynamic_batch.py`, byte-identical in `~/tt-inference-server` and `~/tt-inference-server-2`):

| | tt-media-server (`device_worker_dynamic_batch.py`) | stock `vllm serve` + `lm_eval --num_concurrent 32` |
|---|---|---|
| **Admission shape** | `request_feeder()` (lines 155-186): blocks for 1 item, greedily drains up to `MAX_NUM_SEQS` more via `TTQueue.get_many()`, fires each as a **fire-and-forget** `asyncio.create_task`, loops straight back — never awaits the batch it just launched. New batches launch while old ones are still generating → **unbounded overlapping bursts**; instantaneous concurrency can exceed `MAX_NUM_SEQS`. | lm_eval's client pool holds concurrency at a steady 32 — a new request is submitted only once a prior one frees a slot. Concurrency **plateaus**, never bursts past it. |
| **Batch size meaning** | `MAX_NUM_SEQS` env var (`settings.vllm.max_num_seqs`), not hardcoded — 32 for this Falcon3 single-layer config. | `--num_concurrent` flag, numerically the same (32) here, but a hard ceiling, not a per-round drain size. |
| **Seed** | Force-dropped unconditionally (`utils/sampling_params_builder.py:42-48`) — Forge device sampler ignores per-request seeds anyway (tt-xla#4539), and a non-null seed forces a ~5x slower seeded path (tt-inference-server#4338). | Honored if sent. Both still greedy under `temperature=0`, so this is a speed/path difference, not an accuracy one. |
| **Transport** | FastAPI handler → cross-process `multiprocessing.Queue` (`TTQueue`) → device worker subprocess. Extra hop. | Direct in-process call into `AsyncLLMEngine` from vLLM's own OpenAI-compatible server. |
| **Disconnect/cancel** | Separate `cancel_queue`/`cancel_listener()` (lines 119-143) cancels a task on client disconnect — the exact mechanism the tt-xla#5664 fix note warns can "force-clear" a wedged scheduler if the client is killed. | vLLM/uvicorn's own built-in disconnect handling; no custom cancel path. |
| **Scheduler** | `AscendScheduler`, forced by `platform.py` regardless of entry point. | **Same.** Not a source of divergence. |
| **Engine config** (dtypes, gmu, chunk size, `num_hidden_layers`, etc.) | Confirmed byte-identical (above). | **Same.** Not a source of divergence. |

`burst_falcon3_7b_forge.py` reimplements the tt-media-server admission shape (queue → block-for-first + greedy nowait-drain up to `--batch-size` → fire-and-forget `asyncio.create_task` per item → loop immediately, no await) against stock `vllm serve`, to test whether the *shape* of admission — not just the concurrency number — is what triggers the hang. Verified it actually produces overlap (a 3-per-batch/6-total smoke test hit `peak_in_flight=6`, i.e. two batches running concurrently).

Two bugs found and fixed in the driver itself before it could be trusted at `--continuous` duration (both in the initial 2026-07-27 version): (1) an unbounded `asyncio.Queue`'s `put()` never truly suspends while there's room, so the `--continuous` producer ran as a non-yielding busy loop that starved the feeder entirely — zero requests ever sent, while RSS grew to 16.4GB in 90s (killed before it caused real system memory pressure); (2) even after bounding the queue, nothing capped *total* concurrent in-flight requests, so `--continuous` synthesized unbounded demand and `in_flight` exploded past 28,000 with thousands of client-side failures — not a faithful reproduction, since the real driver's overlap is bounded by actual client arrival rate, not infinite. Fixed with a `--max-in-flight` cap (default `2×batch-size`) gating the feeder before each batch pull.

### Burst-admission driver run: clean, no hang (2026-07-27)

`./burst_falcon3_7b_forge.py --continuous` (defaults: `batch_size=32`, `max_in_flight=64`, `max_tokens=128`) run for **37+ minutes** against `serve_falcon3_7b_forge.sh` (single layer), manually stopped (Ctrl-C) once it cleared the stopping-criteria bar (≥30-45min / ≥10-15k requests, agreed in advance to make a real "did not reproduce" call rather than an arbitrary one):

- **20,832 fired / 20,736 completed / 0 failed** across the entire run (445 status lines, every one `failed=0`).
- `in_flight` oscillated 64↔128 throughout (128 = the expected batch-scheduling overshoot above the 64 cap, not runaway) — never got stuck at a constant value, which is what a hang would look like in this output.
- **No hang under the bursty fire-and-forget admission pattern either**, at meaningfully greater cumulative concurrency-depth and duration than the single ~30-concurrent burst observed in the suspicious `tt-inference-server-2` log below.

Throughput did show a **mild downward trend**, same shape as (but much weaker than) the smooth-concurrency slowdown found below: completion rate over 500s windows was 10.5/s (100-500s) → 9.7/s (500-1000s) → 9.0/s (1000-1500s) → 8.6/s (1500-2000s) → 8.1/s (2000-2225s), a ~23% decline start-to-end. Weaker than the ~2.6x smooth-concurrency decline over a shorter run, but the same *qualitative* shape (gradual, monotonic, no errors) — suggestive that whatever's accumulating is a function of server uptime/total requests served rather than of the specific admission pattern.

**Escalated pass** (`--batch-size 64 --max-in-flight 256 --continuous`, ~15 min, manually stopped): **7,360 fired / 6,956+ completed / 0 failed**, `peak_in_flight=566` (some overshoot above the `2×max_in_flight=512` expectation, consistent with the batch-scheduling-overshoot mechanism scaling up with batch size — not runaway), `in_flight` draining steadily downward (468→436→404) at time of kill, not stuck. **No hang at ~4.4x the default burst depth either.**

### Corroborating evidence: bursty-admission signature caught in an existing log

`~/tt-inference-server-2/falcon3_single_layer_server.log` (2026-07-22, tt-media-server path, same config as above) ends with a pattern consistent with a live hang: at `17:01:07,03x` it logs 30 `Device 0: Starting non-streaming batch generation for 1 requests` lines immediately followed by 30 `Device 0: Starting non-streaming generation` lines (a `get_many(32)`-shaped burst of fire-and-forget tasks). **After that: zero further generation/completion log lines** — no error, no traceback, no shutdown message — just `Worker health check: 0 dead workers found` every 30s for the next 10.5 minutes until the file stops growing at `17:11:47`. This is not proof (no py-spy/gdb capture was taken at the time, and the PID doesn't match either of the two occurrences captured earlier in this doc — this may be a separate, earlier instance of the same class of hang), but it's a second independent sighting of "requests submitted, then nothing ever completes, no error" specifically on the bursty-admission path.

### Clean-run baseline: 10x eval-loop, smooth concurrency, single layer (2026-07-27)

`./run_falcon3_7b_evals_loop.sh --loops 10 --limit 0.75 --dir falcon3_loop_single_layer_0.75` completed **10/10 passed, 0 failed, 0 timed out** against `serve_falcon3_7b_forge.sh` (stock `vllm serve`, smooth lm_eval concurrency=32) — no hang under this admission pattern in this run.

However, per-iteration wall time **grew monotonically and substantially** despite every iteration running the identical fixed workload (same `--limit 0.75`, same tasks, same prompts pattern):

| iter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| elapsed (s) | 158 | 162 | 181 | 211 | 244 | 277 | 311 | 346 | 388 | 417 |

~2.6x slowdown from iteration 1 to 10, roughly linear (~+29s/iteration average after iter 2). No errors, no OOM messages, no explicit hang — just steadily increasing latency across independent server-uptime, not within a single growing-context request. This is a candidate precursor/related phenomenon to the hang: consistent with a per-server-lifetime resource accumulating (e.g. prefix-cache bookkeeping, request-id/state tables, KV-pool fragmentation) rather than anything tied to a single request's context depth. Not yet known whether this trend continues to an eventual full stall (10 iterations / ~45 min wasn't enough to find out) or plateaus. Worth extending this exact run to more iterations before concluding stock `vllm serve` is clean.

### Standalone tt-xla verdict AT SINGLE LAYER: does not reproduce (as of 2026-07-27) — ⚠️ superseded at full depth, see occurrence 5 below

Three tested admission patterns — smooth lm_eval concurrency=32 (~45min, 10 iterations, thousands of requests), bursty fire-and-forget up to 128 in-flight (~37min, 20,736 completions), and an escalated bursty pass up to 566 in-flight (~15min, 6,956+ completions) — all ran **clean, zero failures, no hang** against `serve_falcon3_7b_forge.sh` (stock `vllm serve`, single layer, byte-identical config to a real tt-media-server run). Per the handoff doc's own framing, **a clean standalone run is evidence, not proof** — it does not clear tt-media-server's actual driver/process/transport stack, only vLLM's native entry point under the admission shapes tested here.

**This verdict was single-layer only and did not hold at full model depth** — see occurrence 5 far below: the identical standalone `vllm serve` + real `run.py` client combination, at full depth instead of single-layer, hung on the very first attempt. So the "clean" result here was masking a depth-dependence, not clearing the standalone path in general. Combined with occurrence 4 (tt-media-server, full depth, also hangs) and this file's original single-layer tt-media-server occurrences (1-3), the pattern that best fits all five data points is: **hangs at full depth regardless of server (tt-media-server or pure `vllm serve`); clean at single-layer regardless of server or admission pattern tested so far.** Depth, not server/admission shape, now looks like the operative variable — though single-layer was never tested anywhere near as adversarially (burst/escalated) as it was at smooth concurrency, so that's not airtight either.

### Real `run.py` eval client against the standalone server (2026-07-27)

Before fully pivoting back to a tt-media-server-launched server, tried an intermediate, better-controlled datapoint: `~/scripts/model_servers/run_evals_forge.sh` (the *actual* tt-inference-server `run.py --workflow evals` orchestration, same client CI/production use) never launches its own server — it only preflight-checks `/health`/`/v1/models` then execs `run.py` against `--service-port`. So it can point at our standalone `serve_falcon3_7b_forge.sh` process directly, with zero modification:

```bash
~/scripts/model_servers/run_evals_forge.sh --model Falcon3-7B-Instruct --port 8019 --task ifeval --samples 2
```

Validated this end-to-end (`rc=0`). Bonus finding: the `lm_eval` command `run.py` actually built —
`--model_args model=tiiuae/Falcon3-7B-Instruct,base_url=http://127.0.0.1:8019/v1/completions,tokenizer_backend=huggingface,num_concurrent=32,max_retries=1 --gen_kwargs stream=False,seed=42 ...` —
matches what `run_falcon3_7b_evals.sh` was hand-built to replicate, confirming that script is a faithful stand-in for the real client rather than an approximation that merely looks similar.

This isolates **same real client, different server** as a variant distinct from everything above (which was: different client (hand-rolled/burst) + different server (standalone) vs. the original different client (`run.py`) + different server (tt-media-server)). Result: **clean, 10/10, no hang** — see below.

### `run_evals_forge.sh --loops 10` against the standalone server: clean, no hang (2026-07-27)

`~/scripts/model_servers/run_evals_forge.sh --model Falcon3-7B-Instruct --port 8019 --loops 10 --dir falcon3_ttis_run_evals_forge_loop_single_layer_no_ds` ran the real `run.py --workflow evals` orchestration against the standalone server 10 times, total 61m36s. All 10 iterations completed normally with `rc=1` — confirmed this is **not a hang**: `rc=1` is `run.py`'s "Acceptance: FAIL (2 blocker(s))" exit code, from the ifeval/gpqa scores missing their accuracy-tolerance vs. published reference (`tolerance: 0.05`) — exactly the expected, already-documented single-layer garbage-accuracy behavior (§3 above), not a crash/timeout/error. So: **same real client (`run.py`), different server (standalone `vllm serve`) → still clean.**

Bonus: the same monotonic per-iteration slowdown trend showed up a *third* time, now under the real client: elapsed per iteration was 205, 200, 215, 266, 324, 379, 437, 509, 554, 607s across the 10 iterations (~3x growth, roughly linear), same qualitative shape as both earlier variants. Increasingly looks like a property of the server process itself (any client, any admission pattern) rather than anything client-side.

**Combined with the standalone-server results above, all four tested (client × admission-pattern) combinations against `vllm serve` are now clean.** Per the handoff doc's framing, this doesn't clear tt-media-server's actual process/driver — moved to that next (see below).

### Restarting against the real tt-media-server: full-model pass (2026-07-27)

Backgrounded a watcher for the above loop's completion, which then cleanly tore down the standalone server (`pkill -9` sequence, verified port/device free) and relaunched the **real** tt-media-server (`~/scripts/model_servers/launch_falcon3_7b_instruct_uvicorn.sh`, `TT_INFERENCE_SERVER_ROOT=~/tt-inference-server-2`) — this time targeting the **full (non-single-layer) model**, to test the "does it reproduce at full depth" open question.

Hit a real bug in the launcher script during this: **`launch_falcon3_7b_instruct_uvicorn.sh` line 44 unconditionally hardcoded `export NUM_HIDDEN_LAYERS=1`**, ignoring any caller override entirely (unlike `serve_falcon3_7b_forge.sh`, which correctly uses `${NUM_HIDDEN_LAYERS-1}`). This silently launched a *second* single-layer server despite explicit attempts to unset/clear the var beforehand — caught only by checking `/proc/<pid>/environ` on the launched process and noticing `NUM_HIDDEN_LAYERS=1` was still there. Fixed by changing it to `export NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS-1}` (same convention as the sibling script: unset → defaults to 1, explicitly empty (`NUM_HIDDEN_LAYERS=`) → full depth). Re-launched with `NUM_HIDDEN_LAYERS=` explicit-empty; confirmed via the server's own `additional_config` log line that `num_hidden_layers` is now absent from the dict entirely (full depth in effect), vs. `'num_hidden_layers': 1` in every prior log.

Also worth noting for anyone repeating this: `pkill -f <pattern>` was unexpectedly and consistently blocked/failing in this session's interactive tool calls (silent, no output, regardless of pattern or whether anything matched) even though the identical `pkill -9 -f ...` sequence worked fine when run from inside an already-backgrounded script. Killing by explicit PID (`kill -9 <pid>`) worked reliably both times: worth defaulting to PID-based kills over `pkill -f` for interactive teardown going forward.

## LIVE HANG CAUGHT — full model, real tt-media-server (2026-07-28)

The very first eval run against the freshly-launched full-model tt-media-server hung. This is the **first live catch** in this entire investigation — every prior occurrence (table above) was found already-wedged or already-ended. Full py-spy/gdb/tt-smi capture was taken while it was still stuck, per the runbook, without touching/killing anything.

**Setup**: `~/scripts/model_servers/run_evals_forge.sh --model Falcon3-7B-Instruct --port 8019 --loops 10 --dir falcon3_ttis_run_evals_forge_loop_full_no_ds` against the just-restarted full-model tt-media-server (EngineCore pid 875071, `TT_INFERENCE_SERVER_ROOT=~/tt-inference-server-2`).

**Where it happened, precisely** (correlating the eval client's own log with the server log): `ifeval` (541 docs) ran to completion cleanly — 100%, 541/541, 7m44s, no issues. Immediately after, `gpqa_diamond_generative_n_shot` (198 docs, `num_concurrent=32`) started at 00:37:51. Its **first** doc completed at 00:38:57 (`1/198, 55.90s/it` — already fairly slow). At that exact moment (`00:38:57,923`, server log), tt-media-server's admission driver logged **12 simultaneous** `Starting non-streaming batch generation for 1 requests` immediately followed by 12 `Starting non-streaming generation` lines — a single `get_many()` round pulling everything the lm_eval client had queued to refill its `num_concurrent=32` window after the first doc freed a slot. **None of those 12 ever completed.** From `00:38:57` to when captured (`01:07`+, still ongoing at time of writing) — 28+ minutes and counting, zero further generation/completion log lines, just periodic `Worker health check: 0 dead workers found`. This is a clean, direct hit of the exact admission-shape theory from earlier in this doc: a burst of concurrent fire-and-forget dispatches, right at a moment of ramping concurrency between eval tasks.

### Evidence gathered live

**Rule out the boring causes**: no `TT_FATAL`/OOM/traceback anywhere near the hang (the only `TT_FATAL` lines are benign startup-time eth-core warnings, present in every run). Device holder check: only PID 875071 holds `/dev/tenstorrent/0` — no contention.

**py-spy dump --pid 875071** (MainThread):
```
__torch_function__ (tt_torch/torch_overrides.py:149)
sample_tokens (vllm_tt/model_runner.py:2010)
decorate_context (torch/utils/_contextlib.py:124)
sample_tokens (vllm_tt/worker.py:346)
run_method (vllm/v1/serial_utils.py:510)
collective_rpc (vllm/v1/executor/uniproc_executor.py:97)
sample_tokens (vllm/v1/executor/uniproc_executor.py:125)
step_with_batch_queue (vllm/v1/engine/core.py:515)
_process_engine_step (vllm/v1/engine/core.py:1238)
run_busy_loop (vllm/v1/engine/core.py:1199)
```
Identical to occurrences 1 & 2 — genuinely inside a live `sample_tokens` call, mid-step, not vLLM's outer empty-schedule idle loop. Every other thread (`_report_usage_worker`, `tqdm_monitor` x3, `process_input_sockets`, `process_output_sockets`, `signal-callback`) shown as idle.

**Per-thread CPU-delta sampling** (2-3s windows, checked twice ~1 min apart, both times identical): exactly one native thread (LWP 875225, unnamed by py-spy since it's pure-native — no Python frames) accumulates ~100% of one core (313 ticks/3s, 200 ticks/2s); every other thread of ~60 total flat at 0.

**gdb -p 875071 -batch -ex "thread apply all bt"**, the hot thread (LWP 875225):
```
#0  tt::umd::ClusterDescriptor::is_chip_mmio_capable(int) const
#1  tt::Cluster::read_sysmem(void*, unsigned int, unsigned long, int, unsigned short) const
#2  tt::tt_metal::read_cq_host_ptr<true>(tt::tt_metal::SystemMemoryManager const&, int, unsigned char, unsigned int, unsigned int)
#3  tt::tt_metal::SystemMemoryManager::completion_queue_wait_front(unsigned char, std::atomic<bool>&) const
#4  tt::tt_metal::buffer_dispatch::copy_completion_queue_data_into_user_space(...)
#5  std::_Function_handler<...>::_M_invoke(...)
#6  tt::tt_metal::distributed::FDMeshCommandQueue::copy_buffer_data_to_user_space(...)
#7  tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue()
```
Same core signature as documented in occurrences 1 & 2 (`completion_queue_wait_front` → `copy_completion_queue_data_into_user_space` → `copy_buffer_data_to_user_space` → `read_completion_queue`), now with two additional inner frames resolved (`read_cq_host_ptr`, `Cluster::read_sysmem`, `is_chip_mmio_capable`) — the thread is actively polling a host-memory pointer via `sysmem` reads, checking MMIO-capability on every iteration, waiting for a completion doorbell that never arrives. Every other of the ~60 threads is idle (compiler threadpools, `gomp_barrier_wait_end`, `RealtimeProfilerManager` nanosleep, etc.) — confirms this is not a compile stall.

**tt-smi -s**: chip reports healthy — `ARCCLK: 0x320` (ticking), `DDR_STATUS: 0x55555555` (healthy pattern), `TIMER_HEARTBEAT` advancing across samples (`0x97779` → `0x9777b`). **The chip firmware itself is alive and responsive** — this is not a crashed/dead chip; a specific pending buffer-read transaction's completion just never gets signaled.

**EngineCore RSS**: ~21GB, unremarkable for a loaded full 7B model + KV cache — no obvious runaway leak.

**#4521 fix presence**: confirmed still present in this install (`scheduled_req_ids.discard(...)` at 4 call sites in `ascend_scheduler.py`, matching the documented fix).

### Verdict: this is NOT the #4521 scheduler deadlock

Applying the exact same discriminator table from earlier in this doc:

| | #4521 (scheduler deadlock) | This hang (all 4 occurrences) |
|---|---|---|
| Completion-queue thread CPU | idle/blocked, ~0% | **actively spinning, ~100% of one core** |
| MainThread Python stack | outer idle loop, empty schedule | **inside a live `sample_tokens` call, mid-step** |
| Chip firmware | n/a (was never actually the issue) | **alive and healthy** (heartbeat advancing, DDR OK) |
| Root cause location | Python scheduler bookkeeping | **native code — no scheduler frames anywhere in either stack** |

Three independent live-evidence points all agree: (1) py-spy shows genuine in-progress device work, not idle scheduling; (2) exactly one native thread spins at 100% specifically inside the completion-queue read path; (3) the chip itself is healthy per tt-smi, ruling out a crashed/hung ASIC. **This is a genuine native/device-side completion stall**: something about a specific buffer-read dispatch, most likely from among the batch of 12 requests fired simultaneously via the fire-and-forget burst admission pattern, never gets marked complete by firmware/dispatch, and the host-side polling thread spins forever waiting for it. The admission-pattern theory from earlier in this doc now has its most direct evidence yet — the hang triggered at literally the first moment this session exercised a multi-request burst-fire admission event against a warm, previously-healthy server.

**State preserved, not touched**: the server was deliberately left running and wedged (PID 875071 / EngineCore, port 8019) for further investigation rather than killed — per the "next moves" below and the general principle that destroying a live wedge destroys the evidence.

### Updated open questions

- **Confirmed: NOT the #4521 scheduler deadlock** (see verdict above) — this line of inquiry is closed.
- **Confirmed: reproduces at full model depth**, not just single-layer — resolves that open question. (Single-layer-specific re-confirmation via tt-media-server was skipped in favor of going straight to full depth; historically occurrences 1-3 were single-layer, so both depths appear affected.)
- **Which specific request among the 12 stalls, and why?** Not yet isolated — the per-op sync + forced-full-sync-and-log methodology from the original "Next steps" section (never yet executed) is now the clear next move, with a live, still-wedged process to instrument against if it can be done without disturbing it, or as a repro script for the *next* occurrence.
- **Does the hang correlate specifically with burst depth (12+ simultaneous dispatches), or would it eventually trigger even under smooth/lower concurrency given enough total volume?** The standalone `vllm serve` burst driver pushed peak_in_flight to 566 with zero hangs — but that was a *different* server (stock `vllm serve`, not tt-media-server) and a *different* client (synthetic, not real lm_eval/gpqa prompts). Worth trying `burst_falcon3_7b_forge.py` directly against tt-media-server next, now that it's known to be the side that actually reproduces.
- **Is the monotonic throughput-decay trend** (seen under three independent variants prior to this) **related to this stall** (e.g., some resource whose fragmentation/exhaustion state makes a stall more likely over server lifetime), or unrelated? This hang occurred relatively early — right after the first task transition — so a long-uptime precondition seems less likely, but not ruled out.
- **py-spy/gdb dumps and tt-smi output** for this occurrence are captured above in full; worth also trying the per-op forced-sync-and-log instrumentation (documented in "Next steps" near the top of this file) against a future occurrence, or right now if it can be attached without disturbing the current wedge.

---

## LIVE HANG CAUGHT #2 — full model, PURE standalone `vllm serve`, no tt-media-server (2026-07-28)

**This reproduces with zero tt-inference-server/tt-media-server involvement — pure tt-xla `vllm serve` + a real eval client hangs on its own.** This is the single most important new data point in the investigation: it rules out tt-media-server's process, transport, and admission driver as necessary conditions entirely. Whatever's wrong lives in the tt-xla vLLM plugin / tt-metal stack itself.

**Setup**: the standalone tt-xla server already running for the tt-triage healthy-server experiments above (PID 278 `vllm serve` / PID 303 `VLLM::EngineCore`, full model — no `NUM_HIDDEN_LAYERS` in `additional_config`, confirmed healthy immediately beforehand via CPU-delta and `/v1/models`) was driven with the exact same real-client command used against tt-media-server previously, just pointed at this server instead:
```bash
~/scripts/model_servers/run_evals_forge.sh --model Falcon3-7B-Instruct --port 8019 --loops 10 --dir falcon3_xla_run_evals_forge_loop_full_no_ds |& tee falcon3_xla_run_evals_forge_master_full_no_ds.log
```

**Where it happened — same shape as occurrence 4, slightly further in**: `ifeval` (541 docs) completed cleanly, 7m22s. `gpqa_diamond_generative_n_shot` (198 docs) started immediately after; this time it got to **17/198** (vs. stalling right after doc 1 in occurrence 4) before completely stopping — last progress line at `02:24:37`, first checked-and-confirmed-stuck at `02:41:39`, **~17 minutes with zero progress** at time of capture.

### Evidence gathered live

Identical signature to occurrence 4 in every respect:

- **Device holder**: only PID 303 holds `/dev/tenstorrent/0`, no contention.
- **CPU-delta**: exactly one thread (LWP 401) spinning at ~100% of a core (309 ticks/3s) — everything else flat.
- **py-spy**: MainThread identical to every prior occurrence — genuinely inside `sample_tokens` → `step_with_batch_queue` → `run_busy_loop`, not the idle loop.
- **gdb** on LWP 401 — same core signature, this capture's outermost frame is `tt::tt_metal::MetalContext::get_cluster()` (a trivial variation in exactly which instant the spin was sampled) over the identical chain: `read_cq_host_ptr` → `completion_queue_wait_front` → `copy_completion_queue_data_into_user_space` → `copy_buffer_data_to_user_space` → `read_completion_queue`.
- **tt-smi**: chip healthy — `ARCCLK: 0x320`, `DDR_STATUS: 0x55555555`, heartbeat present. Not a dead chip.
- **RSS**: ~21GB, unremarkable.

Same verdict as occurrence 4 applies without modification: **not the #4521 scheduler deadlock** (active spin + live mid-step stack + healthy chip, no scheduler frames anywhere).

### What this changes

- **The tt-media-server-specific admission-driver theory (fire-and-forget bursts via `device_worker_dynamic_batch.py`) is no longer a necessary condition.** This server was driven purely by `lm_eval`'s own steady `num_concurrent=32` client pool (the "smooth" pattern) — the same pattern that ran clean for 61 minutes against the *single-layer* standalone server. The burst-admission investigation earlier in this doc wasn't wrong (tt-media-server's driver is still a real structural difference, and may still matter under some conditions), but it's evidently not required to trigger this — full depth alone was sufficient here.
- **Model depth now looks like the best-supported single variable**: every full-depth attempt (2 for 2: tt-media-server and standalone) has hung; every single-layer attempt across multiple client/admission combinations (smooth, bursty, escalated-bursty, real client) has stayed clean. Not proven — single-layer was never pushed as hard adversarially as full-depth was tested by accident (first try, smooth client) — but it's the simplest hypothesis consistent with all five occurrences.
- **State preserved again**: server left running and wedged (PID 303, port 8019) — not killed. This is the live specimen for the next tt-triage attempt (with guardrails) below.

## tt-triage attempt → host crash (2026-07-28): see `TT_TRIAGE_NOTES.md`

Tried `tt-triage` against the still-live wedge above (PID 875071) to get on-device RISC-V callstacks as a complement to the py-spy/gdb evidence. Got some partial results (config dump, a data point about MMIO contention from the spinning thread), but **the attempt crashed the host machine** — confirmed via host kernel logs (`tenstorrent 0000:01:00.0: Failed to set initial power state: -22` followed by a ~4-minute kernel-workqueue death spiral and a total freeze), matching a prior crash the last time tt-triage was used on this rig. Root-caused to an unconditional `set_power_state(true)` call in tt-umd's `TopologyDiscovery`, racing against the already-open, actively-spinning EngineCore.

**Full tt-triage usage notes, the live-attempt results, the complete incident writeup, root-cause analysis, and recommendations now live in `TT_TRIAGE_NOTES.md`** (moved there to keep tt-triage-specific material out of this hang-specific doc). Net effect on *this* investigation: the crash destroyed the live wedge from occurrence 4 — everything gathered before the crash (py-spy, gdb, tt-smi, the #4521-discriminator verdict above) stands as recorded; nothing further could be collected from that occurrence. **Do not run tt-triage against a live/hung server again without reading the recommendation in `TT_TRIAGE_NOTES.md` first** — py-spy + gdb remain safe and should stay the default tools for any future live catch.

## LIVE HANG CAUGHT #6 — full model, `[SAMPLING-FLAGS]` probe active, ifeval downsampled to 0.5 (2026-07-28)

**Setup**: baseline config (`cpu_sampling=false`, `enable_trace=true`, `gmu=0.35`, `MIN_NUM_SEQS=1`, `PREFILL_BATCH_THRESHOLD=16` — i.e. b1-prefill *enabled*), launched via `serve_falcon3_7b_forge.sh` with `TTXLA_DUMP_SAMPLING_FLAGS=1 TTXLA_LOGGER_LEVEL=INFO`, log at `falcon_serve_full_model.log`. Evals run with `ifeval --limit 0.5` (271/541 docs) → `gpqa_diamond_generative_n_shot` full (198 docs), via `run_falcon3_7b_evals.sh`.

**Where it happened**: `ifeval` finished cleanly at `16:05:08` (4m14s for 271 docs). `gpqa` started immediately after (`16:05:09`–`16:05:14`), began requesting at `16:05:14`. Progress in the eval client log:
```
2026-07-28 16:06:12,496 [INFO] ... Requesting API:   1%|          | 1/198 [00:50<2:44:43, 50.17s/it]
2026-07-28 16:06:17,100 [INFO] ... Requesting API:   9%|▊         | 17/198 [00:57<07:41,  2.55s/it]
```
— apparently nothing after that, **but the client log file actually got one more update, ~29 minutes later**:
```
2026-07-28 16:35:15,777 [INFO] ... Requesting API:  14%|█▍ | 28/198 [01:02<04:17, 1.52s/it]2026-07-28:16:35:15 ERROR [models.api_models:552] Exception:TimeoutError(), undefined, retrying.
(same TimeoutError line repeated 5x at the identical timestamp)
```
**This is not new post-freeze progress — it's a delayed flush of something that already happened.** The `28/198` line's own elapsed counter (`01:02`, 62s) matches ~16:06:16 (62s after the gpqa loop started at `16:05:14`), not 16:35:15 — so those extra 11 docs (18-27) genuinely completed for real, in the brief healthy window right after the `Running: 31` log line and before the true freeze, just before or as several of that batch's decodes finished, hit EOS, and returned. The *log line* announcing it simply sat buffered (tqdm output over a non-TTY pipe) until something else forced a flush. That something else is `aiohttp`'s `ClientTimeout(total=1800)` (lm_eval's default request timeout, `lm_eval/models/api_models.py:139`) finally expiring for the handful of requests that never got a response — 1800s after they were sent lines up almost exactly with 16:35:15. So: most of the 31-request running batch actually finished fine; it was specifically the stragglers (including the one freshly re-admitted single-request prefill from the scheduler dump above) that never returned — which sharpens, rather than changes, the b1-prefill lead in the running list. Server-side throughput logger shows the exact freeze point:
```
INFO 07-28 16:06:07 ... Running: 20 reqs, Waiting: 12 reqs, Deferred: 12 reqs, ...  (still admitting the gpqa burst)
INFO 07-28 16:06:17 ... Avg prompt throughput: 3458.5 tokens/s, ... Running: 31 reqs, Waiting: 0 reqs, ...  (burst finished admitting, prompt-throughput spike)
INFO 07-28 16:06:27 ... Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 31 reqs, Waiting: 1 reqs, Deferred: 1 reqs, ...  (FROZEN — last line in the log)
```
Server confirmed still alive and holding the device ~3m later (`date` showed `16:09:42`, log mtime frozen at `16:06:27`).

### Evidence gathered live

**py-spy** (MainThread, PID 1335 EngineCore) — genuinely mid-step, not idle:
```
sample_tokens (vllm_tt/model_runner.py:2037)   # selected_token_ids = selected_token_ids.cpu()[:num_reqs]
decorate_context (torch/utils/_contextlib.py:124)
sample_tokens (vllm_tt/worker.py:346)
run_method → collective_rpc → sample_tokens (vllm/v1/executor/uniproc_executor.py)
step_with_batch_queue → _process_engine_step → run_busy_loop (vllm/v1/engine/core.py)
```
Blocked in the `.cpu()` transfer at the very end of `sample_tokens`.

**gdb** on all 130 threads — every thread cleanly idle (futex/poll/condvar wait) **except one**: the async device-execution thread, mid-work:
```
Thread 48 (LWP 1433) "VLLM::EngineCor":
#0 tt::umd::LocalChip::get_tt_device()
#1 tt::umd::Chip::advance_device_execution()
#2 tt::tt_metal::SystemMemoryManager::completion_queue_wait_front(unsigned char, std::atomic<bool>&) const
#3 tt::tt_metal::buffer_dispatch::copy_completion_queue_data_into_user_space(...)
#4 FDMeshCommandQueue::copy_buffer_data_to_user_space(...)::$_0::_M_invoke(...)
#5 FDMeshCommandQueue::copy_buffer_data_to_user_space(...)
#6 FDMeshCommandQueue::read_completion_queue()
```
**Identical low-level signature to occurrences 4 and 5** — `completion_queue_wait_front` / `read_completion_queue`, the device has stopped posting completions. Main thread's Python-level block (`torch::lazy::LazyGraphExecutor::DeviceLocker::Barrier()`, waiting on a condition variable for this async thread to finish) is consistent with, not contradictory to, the py-spy capture. Not the #4521 scheduler deadlock, for the fourth time running.

**`[SAMPLING-FLAGS]` evidence** — the last ~25 steps before the freeze, and every step during the ifeval run, all show **identical sampling metadata**: `all_greedy=True all_random=False logprobs=False no_penalties=True no_min_tokens=True no_logit_bias=True no_bad_words=True no_allowed_token_ids=True no_generators=True apply_grammar=False`, for both tasks, at every step. This rules out any sampling-metadata-driven branch/recompile (e.g. `sample_from_logits`'s greedy-fast-path guard) as a contributor — ifeval and gpqa hit the exact same sampling code path.

What *does* stand out in `[SAMPLING-FLAGS]` right before the freeze: a burst of **single-request prefill steps** (`num_reqs=1`, `decode=False`) cycling through multiple buckets back-to-back as gpqa's variable-length 5-shot docs get admitted one at a time:
```
[SAMPLING-FLAGS] num_reqs=1 target_num_reqs=1 total_sched_tokens=1024 input_ids_shape=(1, 1024) decode=False ...
[SAMPLING-FLAGS] num_reqs=1 target_num_reqs=1 total_sched_tokens=564  input_ids_shape=(1, 1024) decode=False ...
[SAMPLING-FLAGS] num_reqs=1 target_num_reqs=1 total_sched_tokens=1024 input_ids_shape=(1, 1024) decode=False ...
[SAMPLING-FLAGS] num_reqs=1 target_num_reqs=1 total_sched_tokens=410  input_ids_shape=(1, 512)  decode=False ...
[SAMPLING-FLAGS] num_reqs=1 target_num_reqs=1 total_sched_tokens=159  input_ids_shape=(1, 256)  decode=False ...
```
— then decode resumes briefly at full batch (`num_reqs=32`, then `31`), then freezes 10s later. The very last `[SAMPLING-FLAGS]` line before the freeze is another lone `num_reqs=1` prefill (`input_ids_shape=(1, 256)`, `total_sched_tokens=138`).

### Deeper dive: scheduler state + proof decode traffic reached the device (~25 min into the freeze)

Went back in with `py-spy dump --locals` (all threads) and gdb, specifically to answer "is this another #4521 scheduler deadlock, and did we actually send traffic to the device."

**Scheduler is not deadlocked — it produced real, non-empty work and handed it off.** `py-spy --locals` on the MainThread shows `sample_tokens (vllm_tt/model_runner.py:2037)` with a live, populated `scheduler_output: <SchedulerOutput at 0x7e888b079340>` local — not the empty-schedule idle case. The frozen call is specifically executing `num_reqs=1, target_num_reqs=1, start_index=0, end_index=1` with `combined_selected_tokens=[]` still empty — i.e. this is a **lone fresh-prefill admission** (the same `input_ids_shape=(1,256)` step the last `[SAMPLING-FLAGS]` line recorded), scheduled *alongside* the already-running 31-request decode batch, not a case where scheduling stalled. `step_with_batch_queue`'s locals confirm `model_executed: True`. The two ZMQ I/O threads (`process_input_sockets`, `process_output_sockets`) are both cleanly idle on their own queues/pollers — no backlog, no separate deadlock there either. This directly contradicts the #4521 signature (MainThread idling in the outer `run_busy_loop` with an empty schedule); here the scheduler already did its job for this step.

**Decode/prefill traffic was absolutely sent to the device — for the whole run, and for this exact frozen step.** Two independent lines of evidence:
- Throughput logs show ~21 minutes of successful serving (tens of thousands of tokens, hundreds of engine steps) before the freeze — this is not a server that never got traffic.
- For the specific frozen step: `sample_tokens()` had already run the forward pass + sampling (lazily traced) and reached the final `.cpu()` pull (`model_runner.py:2037`) — a `.cpu()` call only blocks on ops *already issued*. Confirmed natively: of all 130 threads, **exactly one** (LWP 1433, the async device-execution thread) is burning CPU — `utime` advanced 209 ticks over a 2s window (pegged at ~100% of a core) while all 129 others show a zero delta. It's spinning inside `SystemMemoryManager::completion_queue_wait_front` → `read_from_sysmem`, i.e. actively polling for a completion queue entry that only gets written *after* a command was dispatched to hardware. A completion-queue wait cannot happen without a prior dispatch — this is direct proof the step's work reached the device.

**New: the chip firmware itself is alive, not dead.** `tt-smi -s` (read-only telemetry, no Inspector/tt-triage RPC involved) shows `ARCCLK=0x320`, `DDR_STATUS=0x55555555` (healthy pattern), and **`TIMER_HEARTBEAT` actively incrementing** (`0x1f9b6` → `0x1f9d6` over 3s). So this isn't a fully-wedged/unresponsive ASIC — the ARC management firmware continues ticking normally while specifically the completion-queue/dispatch-firmware path has stopped advancing. That's a more precise characterization than "the chip is dead," and narrows where in the stack to look next (dispatch firmware / command-queue state, not ARC/power management).

**Limitation hit**: `libtt_metal.so`/`libtt-umd.so` are release builds with no debug symbols beyond function names — gdb's `info args`/`info locals` return nothing inside those frames (only libc frames like `start_thread` have locals). Getting further (actual completion-queue read/write cursor values, in-flight command/buffer IDs) would need a debug build of tt-metal/tt-umd. Also tried gdb's CPython pretty-printer (`py-bt`) as an alternative Python-level introspection path — not available for this interpreter build (`Undefined command: "py-bt"`), so `py-spy --locals` remains the reliable tool for Python-level state.

### Compile-timing check: undercuts the "in-flight recompile" theory

Checked every `Compiling graph for config=...` line in the server log (68 total): **all of them occurred during startup warmup** (server log lines 111–346, `15:43:42`–`15:51:22`, well before evals started at `15:51:37`), covering every bucket combination that later showed up in the gpqa burst (`num_tokens` ∈ {1, 128, 256, 512, 1024}, both `all_greedy` states, both `apply_grammar` states, both `prefix_chunk` states — `Compilation finished in 464.84 [secs]`). **Zero new compiles occurred during the actual eval run** — not at the ifeval→gpqa transition, not during the admission burst, not at the freeze. So in this occurrence, an in-flight recompile corrupting a trace region cannot be the trigger — there was no recompile happening at all. What's still suspicious is the *replay* pattern: many different already-compiled prefill buckets getting exercised back-to-back in a tight window, immediately followed by the freeze.

### What this changes

- **Fourth occurrence with the identical low-level signature** (`completion_queue_wait_front`/`read_completion_queue`), across two different server processes (tt-media-server, standalone), two different eval clients, and now two different `TTXLA_LOGGER_LEVEL`/probe configurations. The mechanism is extremely consistent; the trigger condition is still unknown.
- **Sampling-metadata differences are ruled out** as a cause — identical flags throughout both tasks.
- **"Recompile-during-2nd-eval" as stated is weakened**: this occurrence hung with zero new compiles anywhere near the transition (everything was pre-warmed). The timing correlation with the ifeval→gpqa transition itself remains strong and unexplained — but if it's not about a fresh compile, the leading candidate shifts to **replay of many different prefill buckets in quick succession** and/or the **b1-prefill (`num_reqs=1`, serial single-request admission) code path**, which is exactly what's running in every capture immediately before a freeze (occurrence 4, 5, and now 6 all show single-request-at-a-time prefill activity right before going silent — worth re-checking 4 and 5's captured evidence for this pattern too).
- **State preserved**: PID 1310 (`APIServer`)/1335 (`EngineCore`) left running and wedged, holding `/dev/tenstorrent/0`. Not killed as of this writing.

### Next steps / faster repro

Folded into the master running list — see **"Running list: debug ideas, knobs, and time-to-repro improvements"** near the top of this doc. Key line references from this occurrence's analysis: b1-prefill axis at `platform.py:71-75` / `model_runner.py:2578-2580`; token-padding ladder at `model_runner.py:3938`; `prefix_chunk`/`_chunked_sdpa_active` gating at `model_runner.py:433`; warmup cost confirmed at **464.84s** (`falcon_serve_full_model.log:1-347`).

## LIVE HANG CAUGHT #7 — untouched baseline, scheduler debug prints restored (2026-07-28)

Re-confirmation run (`falcon_serve_full_model_debug_scheduler_prints.log`) with the reverted scheduler debug logging (commit `08fbdbe7706eb05dd38623f47fcb0f9f3ccf37da` un-reverted) active, ifeval downsampled to 0.25. Hung again at the ifeval→gpqa transition, same as occurrence 6.

**py-spy + gdb**: identical signature to every prior occurrence — MainThread parked in `sample_tokens` → `.cpu()` pull; the one actively-spinning thread (found instantly via the CPU-delta sweep, 209 ticks/2s) is in `completion_queue_wait_front → read_cq_host_ptr → is_dram_backed → rtoptions` / `FDMeshCommandQueue::read_completion_queue`.

**Scheduler debug prints (`[TT-SCHED]`) right up to the freeze — completely clean, no anomaly**:
```
[TT-SCHED] prefill admitted 1 req(s) [...] | running=31/32 kv_usage=25.2% waiting=0 skipped=1
[TT-SCHED] decode batch -> 32 req(s) [...] | running=32/32 kv_usage=25.4% waiting=0 skipped=0
[TT-SCHED] decode batch -> 31 req(s) [...] | running=31/32 kv_usage=24.5% waiting=0 skipped=0
[TT-SCHED] UNDER-FED: 1/32 slots free but no requests queued (engine not being fed) | running=31/32 ...
[TT-SCHED] prefill admitted 1 req(s) ['...-a2306d1b'] | running=31/32 kv_usage=25.0% waiting=0 skipped=1
```
— followed by that admitted request's `[SAMPLING-FLAGS]` prefill line (bucket 256), then silence. This is a perfectly ordinary "one request finished, one replacement admitted" cycle — the same pattern every other successful cycle in the run took — with no error, no odd state, nothing scheduler-side distinguishing this one from the hundreds that succeeded before it. **Fifth occurrence confirming the scheduler is healthy; the fault is purely device-side**, now additionally confirmed with the scheduler's own internal request-accounting visible and unremarkable.

## LIVE HANG CAUGHT #8 — `[NEW-REQUEST]` per-request sampling params added (2026-07-28)

Same baseline config; client again displayed `17/198` at the point it went silent (later found to be a stale/buffered artifact, not a real fixed stall point — see the running list). py-spy + gdb: identical signature (CPU-delta sweep found the spinning thread instantly, same `completion_queue_wait_front` chain via `LocalChip::get_tt_device`).

**First run with the new `[NEW-REQUEST]` probe — clean negative result at full per-request granularity.** The last-admitted request before the freeze is now identifiable by name: `cmpl-85c5f58a1bad8326-0-94e43631` (1386 prompt tokens), admitted `19:52:19.370`; its prefill (`total_sched_tokens=1024`, first of two chunks since 1386 > the 1024 bucket) fires 2ms later, one more chunk-continuation step follows (`total_sched_tokens=138`, bucket 256), then silence. Every one of the last 10 `[NEW-REQUEST]` admissions has **byte-for-byte identical sampling params** (`temperature=0.0 top_p=1.0 top_k=0 repetition_penalty=1.0 presence_penalty=0.0 frequency_penalty=0.0 seed=42 max_tokens=256 min_tokens=0 logprobs=None structured_outputs=False`) — only `prompt_tokens` varies (1343-1676 in this window). Rules out per-request sampling-config anomalies definitively (not just in aggregate as before — now confirmed per-request); prompt length / the resulting bucket-cycling pattern remains the only thing that varies and correlates with every hang.

## LIVE HANG CAUGHT #9 — overnight session, independently reproduced via `run_evals_forge.sh` (2026-07-29)

Overnight unattended session (kmabee asleep). Killed the prior wedged server (PID 163747/163714 from the earlier live-catch session), `tt-smi -r`, relaunched via `serve_falcon3_7b_forge.sh` (untouched baseline: full model depth `NUM_HIDDEN_LAYERS=""`, default gmu=0.35/bfp8/opt=1/trace=1/b1-prefill-on, `TTXLA_DUMP_SAMPLING_FLAGS=1 TTXLA_DUMP_NEW_REQUESTS=1 TTXLA_LOGGER_LEVEL=INFO TTMLIR_RUNTIME_LOGGER_LEVEL=WARNING` to avoid the verbose-logging confound), warmup 467.35s / 54 compiled graphs (confirmed via `Add new N compiled XLA graphs` line sum). Then ran the exact requested command, to a fresh log:

```bash
~/scripts/model_servers/run_evals_forge.sh --model Falcon3-7B-Instruct --port 8019 \
  > falcon3_evals_baseline_ds_0.10_1.0_overnight_repro1.log 2>&1
```

**Hung at the identical location**: ifeval completed cleanly (55/55 docs, downsampled via `run.py`'s own `eval_config.py`-driven CI-nightly limit), gpqa started, client displayed the familiar `17/198` at `05:02:26`, then silent. Confirmed via a CPU-delta thread sweep ~80s later: exactly one thread (LWP 170982) spinning at ~100% of one core (308 ticks/3s), all ~130 other threads flat. `gdb -p 170982 -batch -ex "thread apply all bt"` — byte-for-byte identical chain to every prior occurrence (`read_from_sysmem → read_cq_host_ptr → completion_queue_wait_front → copy_completion_queue_data_into_user_space → copy_buffer_data_to_user_space → read_completion_queue`, via `ClusterDescriptor::is_chip_mmio_capable`). `py-spy dump --pid 170877` MainThread: parked in `sample_tokens` (`model_runner.py:2092`) → `worker.py:346` → `collective_rpc` → `step_with_batch_queue` → `run_busy_loop`, identical to every prior capture. `tt-smi -s` confirmed `TIMER_HEARTBEAT` still incrementing (chip alive); RSS ~19.9GB unremarkable. **Ninth independent live catch, tenth-plus occurrence overall, zero deviation from the established signature.**

This run also confirms the "untouched baseline reliably hangs" result holds across a server restart/reset cycle done by an agent working autonomously, not just interactively.

## LIVE HANG CAUGHT #10 — overnight session, standalone `run_falcon3_7b_evals.sh` client — the tt-xla-only repro (2026-07-29)

Immediately after #9, killed the server (`kill -9`, PID-based per the established lesson), `tt-smi -r`, relaunched an identical untouched-baseline server, then — instead of the tt-inference-server `run_evals_forge.sh` wrapper — used tt-xla's own standalone client script with matching downsampling (ifeval 0.10, gpqa full):

```bash
NUM_HIDDEN_LAYERS="" ./serve_falcon3_7b_forge.sh   # wait for WARMUP COMPLETE (466.11s, 54 graphs)
./run_falcon3_7b_evals.sh --ifeval-limit 0.10 --gpqa-limit 1.0     # MISTAKE — see below
```

**Gotcha rediscovered**: `--gpqa-limit 1.0` hit the exact documented trap from earlier in this doc — `lm_eval`'s `get_sample_size()` treats exactly `1.0` as the `int(limit)` branch (absolute count = 1 doc), not "100%". Result: `gpqa_diamond_generative_n_shot (n=1)`, completed in under 2 minutes, no hang (correctly so — 1 doc is nowhere near enough sustained load). **Fixed per the doc's own prior note**: re-ran with `--gpqa-limit 100000` (large integer, clamps to the full 198-doc set). This second attempt (fresh `ifeval --ifeval-limit 0.10` → full `gpqa`) **also completed cleanly** — real n=198 gpqa results, full summary printed, no hang. Two clean completions in a row against the same warm server.

**Third attempt (via `run_falcon3_7b_evals_loop.sh --loops 8 --ifeval-limit 0.10 --gpqa-limit 100000`, iteration 1/8, same still-warm server) hung** — first iteration of the loop, same ifeval→gpqa transition, client log silent after `Requesting API: 0%|...` for gpqa. Confirmed via CPU-delta sweep (two samples 3s apart): exactly one thread (LWP 176464) spinning at ~100% (308 ticks/3s), all others flat. `gdb -p 176464 -batch -ex "thread apply all bt"` — identical chain (`read_cq_host_ptr → completion_queue_wait_front → copy_completion_queue_data_into_user_space → copy_buffer_data_to_user_space → read_completion_queue`, outermost frame this capture `Cluster::get_cluster_description()`). `py-spy dump --pid 176359` MainThread: parked in `sample_tokens` (`model_runner.py:2092`) exactly as every prior capture. `tt-smi -s` heartbeat still incrementing (`0x3477→0x3497`); RSS ~20.6GB.

**This is the tt-xla-standalone repro the session's goal called for** — no tt-inference-server, no tt-media-server, pure `vllm serve` (`serve_falcon3_7b_forge.sh`) + tt-xla's own eval client script (`run_falcon3_7b_evals.sh`), reproduced independently by an agent working unattended.

**New data point on non-determinism**: this is the *first* time in the whole investigation that the *same warm server* saw multiple back-to-back ifeval→gpqa transitions where some were clean and a later one hung (previously, "clean" vs "hangs" was compared across different server launches, not repeated attempts against one continuously-running server). Two clean full-dataset passes, then a hang on the third attempt, against a server whose resident graph count/compiled state never changed between attempts — strong direct evidence this is a genuine timing-sensitive race (probability-of-hit per attempt, not a deterministic function of static server state), consistent with the confidence assessment's existing caveat about the verbose-logging confound.

## LIVE HANG CAUGHT #11 — `TT_RUNTIME_TRACE_REGION_SIZE=1GiB`, refuting pool-isolation as a fix (2026-07-29)

See the dedicated "`TT_RUNTIME_TRACE_REGION_SIZE=1GiB` retry" section above for full details — untouched baseline config plus a 1 GiB dedicated trace region (isolated from the shared weights/const-eval pool). Compiled clean (465.56s), but hung at the identical ifeval→gpqa transition with the identical native signature (`completion_queue_wait_front` chain, LWP 193531, `MetalContext::get_cluster()` outermost frame this capture). This is the result that showed pool isolation alone doesn't help — the race is trace-vs-trace, not trace-vs-other-pool.

## LIVE HANG CAUGHT #12 — `TT_RUNTIME_TRACE_ALLOC_DEBUG=1` instrumentation captures eviction clustering pre-freeze (2026-07-29)

Rebuilt the plugin with two new local (uncommitted, tt-mlir/tt-metal submodule) patches: (1) `TT_RUNTIME_SYNC_BEFORE_TRACE_RELEASE` surgical-fix hook in `trace_cache.cpp` (not yet enabled for this run — this run is the "does the freshly-rebuilt binary still hang unmodified" sanity check, doubling as the alloc-debug capture run), and (2) `TT_RUNTIME_TRACE_ALLOC_DEBUG` raw-stderr instrumentation in `buffer.cpp` + `capture_or_execute_trace.cpp` (see the "Status" list above for exact log format). Ran untouched-baseline config with `TT_RUNTIME_TRACE_ALLOC_DEBUG=1` only. **Hung at the identical transition, identical signature** (LWP 197926 spinning, `read_cq_host_ptr → completion_queue_wait_front → ...`, `py-spy` MainThread parked in `sample_tokens`, `tt-smi` heartbeat alive). This confirms the newly-rebuilt binary (with both new opt-in-gated patches present but the sync fix not yet enabled) reproduces the baseline hang identically — the patches don't change default behavior, as intended.

**New evidence from the instrumentation**: 75 total `EVICT_TRACE` events logged over the run; **the last 5 (distinct trace IDs 12, 85, 78, 98, 91) all fired in a tight cluster immediately before the freeze, all at generation 53** (of ~54 total graphs) — the first direct, instrumented confirmation of the "many trace evictions/recaptures in a tight window right before the freeze" pattern, previously only inferred from `[SAMPLING-FLAGS]` bucket-cycling timing. Also observed the same DRAM address (`2781755008`) freed and immediately reallocated at varying sizes (6MB/3MB/1.5MB — consistent with different prefill-bucket activation sizes) dozens of times in the final seconds before the freeze, back-to-back with no visible gap. Not yet a byte-for-byte proven collision (the `EVICT_TRACE` log point doesn't carry the freed buffer's address — see the "Status" list above for the exact gap and proposed next step), but the strongest circumstantial evidence gathered so far.

## LIVE HANG CAUGHT #13 — `TT_RUNTIME_SYNC_BEFORE_TRACE_RELEASE=1` (surgical fix), attempt 1 — **hung** (2026-07-29)

Tested the actual surgical fix: `Synchronize(device, std::nullopt)` immediately before `release_trace()` in both `TraceCache::erase()` overloads (`trace_cache.cpp`), targeting only the actual eviction transition rather than every replay. **First attempt hung**, at the identical transition, identical signature (LWP 202240 spinning at `read_cq_host_ptr → completion_queue_wait_front → ...` via `ClusterDescriptor::get_closest_mmio_capable_chip`; `tt-smi` heartbeat alive).

**Important methodology note — a red herring investigated and ruled out**: checked `/proc/<EngineCore-pid>/environ` to confirm `TT_RUNTIME_SYNC_BEFORE_TRACE_RELEASE=1` actually reached the runtime, and it was *absent* — alongside several other vars (`TTXLA_LOGGER_LEVEL`, `TTXLA_DUMP_NEW_REQUESTS`) that were confirmed, separately, to have actually taken effect (116 `[NEW-REQUEST]` lines present in this exact run's log, despite `TTXLA_DUMP_NEW_REQUESTS` also being "missing" from that same `/proc/.../environ` snapshot). **`/proc/PID/environ` only reflects the environment at the initial `execve()` snapshot** — it does not reflect anything vLLM's multiprocessing/env-forwarding machinery does afterward, so it is not a valid way to check whether a spawned EngineCore subprocess's C++ runtime actually saw a given env var. Confirmed via this cross-check that the hang is a genuine result, not an artifact of the flag failing to propagate.

## LIVE HANG CAUGHT #14 — `TT_RUNTIME_SYNC_BEFORE_TRACE_RELEASE=1` (surgical fix), attempt 2 — **also hung** — surgical fix confirmed ineffective, theory refined (2026-07-29)

Repeated the exact same test (kill, `tt-smi -r`, relaunch, warmup 465.55s, `run_falcon3_7b_evals_loop.sh` at the same downsampling) with a completely fresh server. **Hung again, on the first iteration, at the identical transition.** CPU-delta sweep found LWP 206643 spinning at ~100% (308 ticks/3s); `gdb` confirmed the identical chain (`LocalChip::get_tt_device → Chip::advance_device_execution → completion_queue_wait_front → ... → read_completion_queue`); `tt-smi` heartbeat alive (`0x205c`).

**n=2/2 — the surgical fix is confirmed NOT to prevent the hang, at least not reliably at the same rate the untouched baseline hangs.** Combined with occurrence #11 (full pool isolation, also didn't help), this settles (not just tentatively suggests) the theory refinement first raised at occurrence #13: **the hazard is not really "DRAM address reuse specifically at trace-eviction time."** Both of the two structurally-targeted interventions tried — isolating trace buffers into their own pool, and synchronizing exactly at the moment a trace buffer's memory is released — fail to prevent the hang. The one intervention that *does* work, `TT_RUNTIME_SYNC_AFTER_TRACE`, is also the only one that syncs on **every single trace replay**, unconditionally, regardless of whether an eviction is happening at all. The common thread across all evidence to date is best explained by: **something about two (or more) trace replays being enqueued/in-flight concurrently — without any eviction necessarily involved — is the actual hazard**, and forcing a full drain after every replay is what removes it (by construction, no two replays can ever overlap). The DRAM-pool-sharing and generation-id-eviction mechanism documented in the ROOT CAUSE section above remains real and verified against source, and graph count/DRAM footprint remains a genuine causal factor (more resident graphs → more distinct traces → more chances of two being replayed close together in a busy window) — but eviction-time synchronization specifically is not where the fix belongs. The next most promising lead: something in the trace **replay/dispatch enqueue path** itself (not the eviction/release path) lacks a needed serialization when two different traces are replayed back-to-back without an intervening full sync — this reframes "where to add the minimal fix" from `TraceCache::erase()` toward wherever `ExecuteTraceOp`/`replay_mesh_trace` enqueues work, which is exactly the code path `TT_RUNTIME_SYNC_AFTER_TRACE` already instruments (just unconditionally, every time, rather than only when actually needed).

## BREAKTHROUGH — minimal, pure-`ttnn` reproducer (no vLLM/tt-xla/tt-mlir-runtime) + exact source-level root cause found (2026-07-29)

Acting on the replay-concurrency refinement above, wrote three standalone Python scripts using only `ttnn`'s own low-level trace API (`ttnn.begin_trace_capture` / `end_trace_capture` / `execute_trace` / `release_trace`) directly against a single-device `ttnn.open_device(device_id=0, trace_region_size=0)` — **zero tt-mlir runtime, zero tt-xla, zero vLLM anywhere in the process**. Repo root: `ttnn_trace_race_repro.py`, `ttnn_trace_race_repro_singlethread.py`, `ttnn_trace_race_repro_replay_only.py`.

**Result — three variants, cleanly isolating the exact necessary condition:**

| Variant | What it does | Result |
|---|---|---|
| Sequential (`_singlethread.py`) | 4 traces, 3 rounds of release+recapture, **no concurrency at all** | ✅ Clean, no crash |
| Replay-only (`_replay_only.py`) | 4 traces captured once, then **4 threads replay them concurrently forever** (940k+ replays over 20s), **no eviction/recapture ever** | ✅ Clean, no crash |
| Concurrent replay + recapture (`ttnn_trace_race_repro.py`) | 4 traces; 1 background thread continuously non-blocking-replays 2 of them; main thread **simultaneously** releases+recaptures the other 2 | ❌ **Crashes immediately, 2/2 runs** |

This is a complete, minimal, deterministic-enough isolation: **the bug requires a trace being released+recaptured (`end_trace_capture`) while a *different* trace's replay (`execute_trace(blocking=False)`) is concurrently in flight on another thread.** Neither ingredient alone is sufficient. At just 4 tiny traces (256×256 tensors) this reproduces essentially immediately — no need for 54 graphs, DRAM pressure, or a busy vLLM serving loop; those just raise the *odds* of hitting the same race in production.

**Crash signature (both runs, identical)**:
```
Segmentation fault (11), Signal code: Address not mapped
  RingbufferCacheManager::add_manager_entry_no_evict(...)
  RingbufferCacheManager::get_cache_offset(...)
  FDMeshCommandQueue::record_end()
  MeshDeviceImpl::end_mesh_trace(...)
  ttnn::operations::trace::end_trace_capture(...)
```
(Second run additionally hit a glibc `malloc(): invalid size`/heap-corruption abort before the segfault — direct confirmation of genuine memory corruption, not just a null-pointer read.)

### Exact source-level root cause (verified line-by-line, not inferred)

`FDMeshCommandQueue` (`tt_metal/distributed/fd_mesh_command_queue.cpp`) owns a single shared `std::unique_ptr<RingbufferCacheManager> prefetcher_cache_manager_` (plus a `dummy_prefetcher_cache_manager_` used to stash state during capture) — this is tt-metal's **prefetcher L1 cache for dispatched kernel binaries**, implemented as a ring buffer (`tt_metal/impl/dispatch/ringbuffer_cache.{hpp,cpp}`), a completely different subsystem from the DRAM buffer allocator this investigation spent most of the night on.

- `FDMeshCommandQueue::enqueue_mesh_workload()` (regular dispatch, and dispatch during trace *capture*) reads/mutates it via `query_prefetcher_cache()` → `prefetcher_cache_manager_->get_cache_offset(...)` (`fd_mesh_command_queue.cpp:486,1638-1639`).
- `FDMeshCommandQueue::record_begin()` (`:1268`) swaps in a dummy manager before capture, `record_end()` (`:1307`) resets and swaps the real one back (`:1627-1630`) after capture.
- `FDMeshCommandQueue::enqueue_trace()` (`:1225`) — **the replay path**, invoked by `ttnn.execute_trace` — also calls `this->reset_prefetcher_cache_manager()` (`:1252`), directly mutating the very same shared object.
- **Every one of these methods takes `auto lock = lock_api_function_();` as its first line — `enqueue_mesh_workload` (`:385`), `record_begin` (`:1269`), `enqueue_trace` (`:1226`) — except `record_end()` (`:1307`), which does not.** Confirmed by direct grep: `lock_api_function_()` appears at exactly the top of every other mutator of this class's shared state, and is conspicuously absent from `record_end()`.

**This is the bug.** `record_end()` — called when a trace capture finishes (`end_trace_capture`) — freely mutates `prefetcher_cache_manager_`/`dummy_prefetcher_cache_manager_` with no lock held, while `enqueue_trace()` (replay, on a different host thread, e.g. vLLM's async device-execution thread vs. its main/capture thread) legitimately holds the *same* mutex when it touches the *same* object. A replay's lock-holding access can interleave with `record_end()`'s unlocked swap/reset of the identical shared `RingbufferCacheManager`, corrupting its internal ring-buffer bookkeeping (block offsets, entry ages) — which explains both the immediate segfault/heap-corruption in the minimal repro (small scale, easy to hit fast) and the production hang (`FDMeshCommandQueue::read_completion_queue()` spinning forever, consistent with the dispatcher/prefetcher ending up in a corrupted state where a completion is never posted).

### Candidate fix applied and verified in this session — **fix confirmed effective**

Added `auto lock = lock_api_function_();` as the first line of `FDMeshCommandQueue::record_end()` (`fd_mesh_command_queue.cpp`), matching every sibling mutator (one-line change, plus a comment). Rebuilt (`ninja` in `third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release`, then the top-level `cmake --build build` to propagate into `third_party/tt-mlir/install/` — this is the install tree tt-xla's own `python_package/ttnn` symlinks to; a `build_Release`-only rebuild is *not* sufficient on its own, learned the hard way).

**Retest results — the exact command that crashed 2/2 pre-fix**:
- `ttnn_trace_race_repro.py --num-traces 4 --recapture-rounds 2` (the original crashing config): **5/5 clean runs post-fix** (one during initial verification, four more back-to-back).
- Stress test at higher scale: `--num-traces 20 --recapture-rounds 100` (10 concurrently-replaying traces, 100 release+recapture rounds on the other 10, **1.24 million background replays** over the run): **clean, no crash, no hang.**

This is now a **verified, working, one-line fix for the specific race the minimal `ttnn` reproducer isolates**, confirmed via a fast-iterating (~10s per run, no vLLM warmup needed) test. **Important, honest caveat found on end-to-end retest (see next section): this fix alone does NOT resolve the original full vLLM/Falcon3-7B production hang.** The `record_end()` missing-lock bug is real, verified, and worth fixing/reporting on its own merits — but the production hang most likely has at least one additional or different concurrency bug in the same general area (trace dispatch/replay), not fully explained by this one fix. Treat this as "a confirmed bug found and fixed," not "the confirmed sole cause of the Falcon3-7B hang."

## END-TO-END VERIFICATION: fix applied to production config — **hang still occurs** (2026-07-29)

Ran the exact untouched-baseline production repro (`serve_falcon3_7b_forge.sh`, full model depth, no env-var workarounds — just the `record_end()` lock fix baked permanently into the rebuilt binary) through the identical eval loop that has hung repeatedly all night. Warmup: 465.47s (matches baseline, confirming the fix adds no meaningful overhead). **Hung on the first iteration, at the identical ifeval→gpqa transition.**

CPU-delta sweep found the same single spinning thread (TID 213588, 307 ticks/3s). `gdb` confirmed the **identical signature**: `is_dram_backed() → read_cq_host_ptr → completion_queue_wait_front → copy_completion_queue_data_into_user_space → copy_buffer_data_to_user_space → read_completion_queue`. `py-spy` MainThread parked in `sample_tokens` exactly as every prior capture. `tt-smi` heartbeat alive (`0x7b2a`). Byte-for-byte the same hang as every occurrence before the fix.

**What this means**: the `FDMeshCommandQueue::record_end()` missing-lock bug found via the minimal `ttnn` reproducer is a real, confirmed, independently-verified bug (crash 2/2 before the fix, clean 5/5 + a 1.24M-replay stress test after) — but it is evidently **not the (sole) cause of the Falcon3-7B production hang**, since fixing it did not stop the production hang from recurring. Plausible explanations, roughly in order of likelihood:
1. **Multiple distinct concurrency bugs exist in this dispatch/trace-replay area.** The `prefetcher_cache_manager_` race is one real, confirmed one; the production hang may be hitting a *different* unsynchronized shared-state race elsewhere in `FDMeshCommandQueue` or the broader dispatch/completion-queue path — plausible given how much shared mutable state (`config_buffer_mgr_`, `expected_num_workers_completed_`, `worker_launch_message_buffer_state`, `sysmem_manager()`, and more) this class manages across capture/replay/end paths, any one of which could have a similar missing-lock bug that a 4-tiny-trace repro wouldn't exercise (different code paths get hit depending on trace size, sub-device count, chunked-prefill state, etc., none of which the minimal repro exercises).
2. **The production workload's specific access pattern** (54 real graphs, KV-cache-backed tensors, real sampling readbacks competing for the same command queue, chunked prefill, sub-device management) may simply be a much richer/different trigger than the minimal repro's synthetic 256×256 tensors and toy op chain, exercising code paths the repro never touches at all (e.g. multi-sub-device synchronization, `config_buffer_mgr_` wraparound, real KV-cache buffer lifetime interactions).
3. Lower likelihood, but not ruled out: the original DRAM-pool/generation-id eviction mechanism documented earlier in this doc (independently verified against source, real, and never actually disproven — only two *specific* interventions targeting it were shown insufficient) could still be a contributing or alternate cause under the production workload's specific conditions, even though it's now clearly not required to reproduce a crash via the minimal repro.

**This is still very real progress, not a wasted result**: one genuine, confirmed, well-isolated upstream bug was found and fixed this session, with a small reusable reproducer that makes it trivial for the tt-metal team to verify independently. The Falcon3-7B production hang needs at least one more root cause found before it's fully resolved — the minimal reproducer and the `record_end()` fix are a strong foundation (and a proven-good methodology: isolate to a tiny pure-`ttnn` script) for whoever continues this investigation.

### Why this reframes (but doesn't contradict) the whole night's DRAM-pool investigation

Every observation from earlier in this document is still consistent with this root cause:
- **Graph-count/DRAM-footprint threshold** — more resident graphs means more distinct traces that can legitimately be mid-replay when some *other* trace's capture happens to be ending; more graphs → more opportunities for the race window to be hit, without graph count *itself* being causal.
- **`TT_RUNTIME_SYNC_AFTER_TRACE` working** — a full device sync after every replay serializes host-side dispatch enough (in practice) to sharply reduce the odds that `record_end()`'s unlocked window and a replay's locked window ever overlap — a side-effect fix, exactly as this doc speculated before this section.
- **`TT_RUNTIME_TRACE_REGION_SIZE` and the eviction-time surgical sync not helping** — entirely expected once the actual shared, unsynchronized state is the *prefetcher cache manager* (host-side dispatch bookkeeping), not the DRAM trace-buffer allocator at all.
- **The DRAM-pool-sharing/generation-id mechanism** documented earlier in this doc is real, verified, and probably still a *legitimate*, independent hazard class (worth its own fix) — it just isn't *this* bug.

## OVERNIGHT SESSION SUMMARY (2026-07-29) — final synthesis, workaround recommendation, confidence verdict

**Superseded in part by the "DECISIVE FINDING" section further below, written after this summary — read that section first.** Short version: this summary (written mid-session) recommended `TT_RUNTIME_SYNC_AFTER_TRACE` as the primary workaround and left the root cause as "somewhere in `FDMeshCommandQueue`'s host-side locking." A later, more direct experiment (live lock-trace instrumentation + the `TT_METAL_OPERATION_TIMEOUT_SECONDS` test) conclusively proved this is a **device-side dispatch/completion-signaling stall**, not a host thread-safety bug at all — see the "DECISIVE FINDING" section for the full evidence and the updated, corrected recommendation (`TT_METAL_OPERATION_TIMEOUT_SECONDS=30` as the new primary production workaround, with essentially zero steady-state cost).

Kyle asked for this session to: reproduce independently, get a tt-xla-standalone repro, draft an upstream issue, dig into the confirmed/open items, build a minimal testcase, and land on a workaround + confidence verdict. Status on each (as understood mid-session; see the correction above):

### What got done
- **Independent repro, twice** (occurrences #9, #10): both `run_evals_forge.sh` (tt-inference-server client) and tt-xla's own `run_falcon3_7b_evals.sh` (the pure tt-xla-standalone repro goal) hit the identical hang, confirmed via CPU-delta/gdb/py-spy/tt-smi each time.
- **`TT_RUNTIME_TRACE_REGION_SIZE` tested and ruled out** as a workaround (occurrence #11) — isolating trace buffers into their own DRAM region does not prevent the hang.
- **A new surgical fix (`TT_RUNTIME_SYNC_BEFORE_TRACE_RELEASE`) written, tested twice, and ruled out** (occurrences #13, #14) — syncing exactly at trace eviction doesn't prevent the hang either.
- **Live-capture instrumentation (`TT_RUNTIME_TRACE_ALLOC_DEBUG`) added and used** to catch a hang with real alloc/dealloc/eviction timestamps (occurrence #12) — didn't produce a byte-for-byte address-collision proof, but did directly confirm eviction clustering right before a freeze.
- **A minimal, pure-`ttnn` standalone reproducer was built — this is the single biggest result of the night.** It isolates the exact necessary condition (concurrent trace replay + a different trace's release/recapture) down to 4 tiny traces, crashing in seconds with no vLLM/tt-xla/tt-mlir-runtime involved at all.
- **Two real, confirmed, independently-verified upstream bugs were found and fixed**: `FDMeshCommandQueue::record_end()` and `FDMeshCommandQueue::wait_for_completion()`, both missing `lock_api_function_()`. The first verified via before/after testing on the minimal reproducer (2/2 crash → 5/5 clean, stress-tested to 1.24M replays); the second found via a systematic source audit of every shared member of the class, verified not to regress anything.
- **Three draft issues written**: focused, complete, ready-to-file reports for each confirmed bug (`ISSUE_DRAFT_fd_mesh_command_queue_record_end_missing_lock.md`, `ISSUE_DRAFT_fd_mesh_command_queue_wait_for_completion_missing_lock.md`), and an honestly-updated Falcon3-7B hang report (`ISSUE_DRAFT_falcon3_7b_completion_queue_hang.md`) that accurately states neither fix resolves the production hang.
- **End-to-end verification against the real production workload was run twice** (once per fix, then combined) — the crucial, sobering result: **neither fix, nor both together, stops the Falcon3-7B hang.** The actual bug causing this specific hang is still unfound.

### Confidence verdict

- **High confidence, fully verified**: both missing-lock bugs are real, exactly located, and independently confirmed. File both upstream regardless of anything else — they're genuine, independent contributions, with zero known downside.
- **High confidence**: neither bug, nor both together, is sufficient to explain the Falcon3-7B production hang. Direct end-to-end retests with each fix (and both together) still hang, identical signature every time.
- **Moderate confidence**: the hazard class is "unsynchronized shared mutable state somewhere in `FDMeshCommandQueue`/the trace dispatch path" — two real bugs of exactly this shape were found via source audit alone, without even needing a live capture. A third such bug, not yet found, remains the most likely explanation.
- **Lower confidence, not disproven**: the original DRAM-pool/generation-id eviction mechanism (documented earlier in this doc, verified against source) could still be a genuine contributing factor, possibly needing to combine with a `FDMeshCommandQueue` fix rather than being an alternative to it. Two specific interventions against it (pool isolation, eviction-time sync) were shown insufficient *alone* — that's weaker evidence against the mechanism entirely than it might first appear, since a partial fix to a multi-part bug wouldn't show a clean result either.

### Workaround recommendation for production Falcon3-7B-Instruct TODAY

**Use `TT_RUNTIME_SYNC_AFTER_TRACE=1`** if the hang is actively blocking a release or CI signal. It is the only intervention proven, repeatedly, to prevent the hang. Cost: a real throughput hit (full device sync after every trace replay, i.e. roughly every decode step) — benchmark this against your latency/throughput SLA before deciding it's acceptable; it is a stopgap, not a long-term setting.

**If throughput matters more than eliminating the hang risk entirely**, reduce resident graph count below the empirically-observed danger zone (`TTXLA_WARMUP_GREEDY_ONLY=1 TTXLA_WARMUP_NO_GRAMMAR=1` costs nothing if the workload is always-greedy/never-grammar, which both ifeval and gpqa are) — this reduces *risk* (fewer chances to hit whichever race(s) are involved) but does not eliminate it, and the exact safe threshold isn't pinned down precisely (somewhere between 36 and 54 graphs).

**Do not use `TT_RUNTIME_TRACE_REGION_SIZE`** — tested, doesn't help, and an oversized value can introduce a new DRAM-OOM failure mode.

**Land both confirmed lock fixes regardless** — zero downside, real bugs, and it's possible a workload needs these plus a still-undiscovered third fix together to be fully safe (per the "lower confidence" note above).

### Immediate next steps for whoever picks this back up

1. **Add live-capture instrumentation** (thread ID + timestamp logging at every `lock_api_function_()` acquire/release across *every* method of `FDMeshCommandQueue`, not just the two already fixed) and capture it during an actual production hang. Two rounds of "audit source, guess a candidate, patch, rebuild, retest end-to-end" (~25 min per cycle) have each found a real bug but not the one causing this hang — direct observation of which two operations are actually racing is now higher-value than continuing to guess.
2. In parallel or as a fallback, audit sibling/related classes (not just `FDMeshCommandQueue`) in the same dispatch stack for the same pattern.
3. File all three draft issues upstream once satisfied (currently left as local drafts per this session's instructions — not submitted).
4. Consider whether the DRAM-pool/eviction mechanism documented earlier needs its own fix in addition to whatever further `FDMeshCommandQueue`-class fix is eventually found — don't assume it's fully ruled out.

## Second candidate found and tested: `FDMeshCommandQueue::wait_for_completion()` missing lock

Follow-up source audit (background agent, launched right after the `record_end()` end-to-end retest showed the hang persists) enumerated every shared mutable member of `FDMeshCommandQueue` and which methods lock vs. don't. Found a second, real, same-shaped bug:

**`FDMeshCommandQueue::wait_for_completion()`** (`fd_mesh_command_queue.cpp:1662`) — reached via `MeshDeviceImpl::quiesce_internal()` → `quiesce_devices()`, exposed to Python as `ttnn.MeshDevice.quiesce_devices()` ("call before closing a mesh that has carved submeshes") — mutates `expected_num_workers_completed_`, `config_buffer_mgr_`, `cq_shared_state_->sub_device_cq_owner`, and `cq_shared_state_->worker_launch_message_buffer_state` with **zero locking anywhere in its call chain**, while `enqueue_mesh_workload()`/`enqueue_trace()`/`record_begin()`/`record_end()` mutate the exact same shared state **under** `lock_api_function_()`. This is a near-duplicate of `reset_worker_state()` (same file, `:1139`), which touches the same four members and is safe only because its one caller holds the lock — `wait_for_completion()`'s caller (`quiesce_internal`) never does.

**Applied and verified in isolation**: added `auto lock = lock_api_function_();` as the first line of `wait_for_completion()`, replaced its internal `finish()` call with `finish_nolock()` to avoid self-deadlocking on the non-recursive `api_mutex_` (`finish()` itself takes the same lock). Rebuilt (`build_Release` then top-level, same two-step process as the first fix). Minimal `ttnn` reproducer re-run post-fix: still clean (expected — this fix targets a different code path the minimal repro never exercises).

**Important caveat on relevance, found before testing**: `quiesce_devices`/`quiesce_internal`/`wait_for_completion` do not appear anywhere in tt-mlir's `runtime/` tree or tt-xla's `python_package/` (confirmed via grep, zero matches) — this API is specifically for tearing down/reconfiguring carved submeshes (multi-device tensor-parallel setups), which our single-P150-chip Falcon3-7B config never does. **This makes it unlikely, though not impossible, that this second fix is the (or a) cause of the Falcon3-7B hang** — the buggy function may simply never be called in this workload's code path. Testing anyway since it's cheap and the fix is unambiguously correct regardless.

**End-to-end production retest with BOTH fixes applied — hang still occurs, exactly as the reachability analysis predicted.** Full untouched-baseline production repro, warmup 465.09s (matches baseline). Hung on iteration 1/8, identical transition. CPU-delta sweep found TID 219582 spinning at ~100% (308 ticks/3s); `gdb` confirmed the same chain (`get_cq_completion_wr_ptr → completion_queue_wait_front → ... → read_completion_queue`, this capture's outermost frame `MetalContext::instance()`); `tt-smi` heartbeat alive (`0x321c`). Byte-for-byte the same hang as every prior occurrence, with **both** confirmed missing-lock fixes now permanently in the binary.

**Conclusion**: two real, independently-verified upstream bugs were found and fixed this session (`record_end()` and `wait_for_completion()`, both missing `lock_api_function_()` on shared `FDMeshCommandQueue` state) — neither resolves the Falcon3-7B hang, and the second was correctly predicted in advance to be unlikely to matter (its only call path, `quiesce_devices()`, is never invoked anywhere in tt-mlir's runtime or tt-xla's plugin code). **The actual bug causing this specific hang has still not been found.** Given each "audit a class, find a candidate, patch, rebuild (~15min round trip), retest end-to-end (~10min)" cycle is expensive relative to the search space (an entire class with many shared members and call sites), continuing to guess at individual candidates one at a time has diminishing returns. The highest-value next step is almost certainly **direct live-capture instrumentation during an actual production hang** — e.g. thread-ID + timestamp logging at every `lock_api_function_()` acquire/release across all of `FDMeshCommandQueue`'s methods (not just the two already found), to directly observe which two operations are racing in the moment, rather than continuing to infer candidates from source reading alone.

## MAJOR REDIRECTING FINDING: live lock-trace instrumentation proves zero `api_mutex_` concurrency in the real hang — the two fixed bugs are real but almost certainly NOT this hang's cause

Implemented `TT_RUNTIME_LOCK_TRACE_DEBUG=1`: instrumented **every one of the 12 `lock_api_function_()` call sites** in `FDMeshCommandQueue` (`enqueue_mesh_workload`, `enqueue_write_shard_to_core`, `enqueue_read_shard_from_core`, `finish`, `enqueue_record_event`, `enqueue_record_event_to_host`, `enqueue_wait_for_event`, `enqueue_trace`, `record_begin`, `record_end`, `wait_for_completion`, `finish_and_reset_in_use`) with raw-`fprintf` ACQUIRE/RELEASE logging (thread-id hash + timestamp), gated by env var, bypassing the log-level system to avoid the confound. Ran the untouched-baseline production repro with this active — **it hung again** (not masked this time), giving a full lock-acquisition timeline right up to a genuine freeze.

**Result: every single one of 255,758 ACQUIRE events across the whole run, cleanly matched by 255,758 RELEASE events, came from exactly ONE thread ID.** Only 4 of the 12 instrumented methods were ever called at all in this workload (`enqueue_mesh_workload` ×252,229, `enqueue_trace` ×3,273, `record_begin`/`record_end` ×128 each, matched pairs) — the other 8 (`enqueue_write_shard_to_core`, `enqueue_read_shard_from_core`, `finish`, both `enqueue_record_event*`, `enqueue_wait_for_event`, `wait_for_completion`, `finish_and_reset_in_use`) were never invoked.

**This proves there is zero concurrency at the `FDMeshCommandQueue::api_mutex_` level in this real vLLM/Falcon3-7B workload** — a single application thread does all trace capture/replay/dispatch sequentially through this lock. This directly contradicts the assumption behind both bugs fixed earlier tonight (`record_end()` and `wait_for_completion()` missing locks): those bugs are real and independently confirmed via a Python script that *explicitly* spawns multiple threads calling `ttnn`'s trace API concurrently — a pattern vLLM's actual engine core does not use. **Both fixes were correctly predicted (in the `wait_for_completion` case) or empirically shown (in both cases, via end-to-end retest) to not resolve this hang — this lock-trace result now explains *why*: the race those fixes address structurally cannot occur in this workload's actual threading pattern.**

### Where the real race almost certainly lives instead

The MainThread (per every py-spy capture) blocks synchronously inside `sample_tokens`'s `.cpu()` pull — this is the single dispatch thread, and it took `api_mutex_` cleanly and released it every time (per the trace above) right up until the point it's waiting on a result. The thread that's actually spinning forever in every hang capture (`FDMeshCommandQueue::read_completion_queue()`) is a **different, tt-metal-internal background thread** (no Python frames, not the one taking `api_mutex_`) — almost certainly the completion-queue reader thread, synchronized via its **own separate primitives** (`reader_thread_cv_mutex_`, `reads_processed_cv_mutex_`, and the raw completion-queue read/write pointers/atomics), independent of `api_mutex_` entirely.

**The actual race is therefore most likely between**: (a) the single application thread enqueuing a new buffer read (`copy_completion_queue_data_into_user_space`'s caller path, which does *not* appear to be one of the 12 `api_mutex_`-guarded methods — none of `enqueue_write_shard_to_core`/`enqueue_read_shard_from_core` were even called, so the real read path must be something else not yet identified/instrumented) and (b) the internal reader thread processing completion-queue entries, using whatever CV/mutex/atomic mechanism governs `reader_thread_cv_mutex_`/`reads_processed_cv_mutex_`/`populate_read_descriptor_queue`/`get_read_descriptor_queue`/`increment_num_entries_in_completion_queue`.

### Immediate next step (highest priority, not yet started)

Instrument this **actual** synchronization domain instead: every acquire/release of `reader_thread_cv_mutex_` and `reads_processed_cv_mutex_`, plus every call to `populate_read_descriptor_queue`/`get_read_descriptor_queue`/`increment_num_entries_in_completion_queue`/`read_completion_queue`, with the same thread-id+timestamp approach — this is where the real concurrency (single dispatch thread vs. tt-metal's own reader thread) actually lives, and where the missing/incorrect synchronization causing this specific hang is most likely to be found.

## DECISIVE FINDING: `TT_METAL_OPERATION_TIMEOUT_SECONDS` converts the silent hang into a clean, catchable exception — proves this is a genuine DEVICE-side dispatch stall, not a host synchronization bug

Following the lock-trace result above, investigated where the actual spin (`SystemMemoryManager::completion_queue_wait_front()`) lives and how it's bounded. Found: `loop_and_wait_with_timeout()` (`system_memory_manager.cpp`) only has a bounded-wait code path if `MetalContext`'s configured operation timeout is nonzero; the default is **0.0** (no timeout at all, `rtoptions.cpp`), controlled by the existing, already-implemented `TT_METAL_OPERATION_TIMEOUT_SECONDS` env var (`export TT_METAL_OPERATION_TIMEOUT_SECONDS=30.0`, per its own usage comment). A source audit of the reader-thread CV/mutex mechanism (`reader_thread_cv_mutex_`/`reads_processed_cv_mutex_`) found it to be correctly synchronized — the producer (host dispatch thread) and consumer (reader thread) always lock the same mutex around the matching increment/wait pair, ruling out a classic missed-wakeup bug there. This pointed squarely at the busy-poll itself being unbounded by default as the mechanism that turns a genuine device stall into a silent, permanent hang instead of a detectable error.

**Tested immediately (zero rebuild needed — pure env var, no code change)**: relaunched the untouched-baseline production repro with `TT_METAL_OPERATION_TIMEOUT_SECONDS=30` added, otherwise unmodified. Warmup unaffected (465.76s, matches baseline — no premature timeouts during normal operation). Ran the eval loop — **hit the identical hang location, but this time it surfaced as a clean, catchable exception ~30 seconds later**:

```
2026-07-29 08:47:30.144 | error    |           Metal | Timeout detected (metal_context.cpp:783)
2026-07-29 08:47:30.234 | critical |          Always | TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable (assert.hpp:104)
RuntimeError: TT_THROW @ .../system_memory_manager.cpp:765: tt::exception
info: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable
backtrace:
 --- ... copy_completion_queue_data_into_user_space(...)
 --- ... FDMeshCommandQueue::copy_buffer_data_to_user_space(...)
 --- ... FDMeshCommandQueue::read_completion_queue()
```
— the exact same stack every silent hang has shown all night, now with tt-metal's own diagnostic explicitly stating **"the device is unrecoverable."** vLLM's engine core correctly caught this as a fatal error, killed the EngineCore, and returned `500 Internal Server Error` to in-flight client requests (`EngineDeadError`), instead of hanging forever. `tt-smi` afterward showed the chip firmware/heartbeat still healthy — it's specifically this dispatch/completion-queue session that's unrecoverable, not the physical ASIC (consistent with every prior occurrence's `tt-smi` heartbeat check).

### What this conclusively proves

1. **This is a genuine device-side dispatch/synchronization stall**, not a host-side thread-safety bug in `FDMeshCommandQueue`'s API surface. Combined with the lock-trace result (zero `api_mutex_` concurrency in this workload), this rules out the entire class of bug the two fixes earlier tonight (`record_end()`, `wait_for_completion()`) address — those are real, confirmed bugs, but structurally cannot be the cause of *this* hang, since they require multi-threaded contention on a lock this workload never contests.
2. **The actual defect is almost certainly on the device/dispatcher side**: something causes the on-device dispatch firmware to stop advancing its completion-queue write pointer for one specific read, permanently. Candidates (not yet investigated in this session): incorrect `expected_num_workers_completed_`/go-signal sequencing written into the command stream by a *prior* operation (trace replay, recapture, or otherwise) corrupting state the dispatcher relies on; a genuine device-firmware bug triggered by some specific op/trace/sequence pattern; or a lost/dropped semaphore or event signal somewhere in the dispatch pipeline that the host never sees because it's purely on-device state.
3. **This also fully explains why every host-side intervention tried tonight targeting `FDMeshCommandQueue`'s lock surface (both confirmed bugs' fixes) failed to resolve the hang** — none of them touch the actual on-device dispatch/completion-signaling mechanism that's actually breaking.

### Why `TT_RUNTIME_SYNC_AFTER_TRACE` (the one previously-proven mitigation) still fits this picture

A full device `Synchronize()`/`Finish()` after every trace replay forces the host to fully drain and confirm every prior op's completion before issuing the next one — this would mask/avoid whatever device-side sequencing issue causes the stall (by never having two "batches" of dispatched work whose completion tracking could become inconsistent with each other), without fixing the underlying device-side defect. Fully consistent with tonight's new understanding.

## UPDATED, FINAL WORKAROUND RECOMMENDATION FOR PRODUCTION

**Two workarounds are now known, with different tradeoffs:**

1. **`TT_METAL_OPERATION_TIMEOUT_SECONDS=<N>`** (e.g. 30) — **recommended as the primary production workaround from this point forward.** Does **not** prevent the underlying device stall from occurring, but converts a silent, permanent, undetectable hang into a fast (~N seconds), loud, catchable failure that vLLM already handles by killing the EngineCore and returning `500` to clients — enabling automated monitoring/alerting/restart instead of an indefinite silent wedge. **Zero steady-state performance cost** (the timeout only matters once a stall has already begun) — this is a meaningfully better default than `TT_RUNTIME_SYNC_AFTER_TRACE` for any deployment that already has crash/restart automation, since it doesn't sacrifice any throughput during normal healthy operation. **Caveat**: does not reduce how often the underlying stall occurs, only how quickly/gracefully it's detected — the server still needs to restart after every occurrence, and in-flight requests at the time of the stall are lost (surfaced as 500s).
2. **`TT_RUNTIME_SYNC_AFTER_TRACE=1`** — still the only known intervention that (in testing so far) prevents the stall from happening at all, at a real, ongoing throughput cost every decode step. Use if avoiding the stall entirely is worth the throughput tradeoff and a hard crash/restart cycle is not acceptable operationally.

**Combine them** for a defense-in-depth production posture if uncertain: `TT_RUNTIME_SYNC_AFTER_TRACE=1` to reduce the *odds* of the stall, plus `TT_METAL_OPERATION_TIMEOUT_SECONDS=30` as a safety net in case it still occurs.

**Do not use `TT_RUNTIME_TRACE_REGION_SIZE`** — confirmed not to help, can introduce an unrelated OOM.

### Immediate next step for whoever picks this up

The search should now focus entirely on **device-side dispatch/completion-signaling correctness**, not host-side `FDMeshCommandQueue` locking (which is now well-explored and shown not to be the cause). Concretely: instrument (or find existing instrumentation for) the on-device dispatcher's go-signal/worker-completion-count sequencing, and/or engage tt-metal's dispatch-firmware owners directly with this exact repro and the `TT_METAL_OPERATION_TIMEOUT_SECONDS` finding — this is likely to get a much faster, more targeted response than continuing to guess from the host side, since it now points at a specific, well-characterized subsystem (on-device dispatch/completion signaling) rather than the broad "somewhere in trace replay" starting point this investigation began with.
