# Falcon3-7B-Instruct single-layer hang — debug session notes

Working notes for an eventual issue. Not yet root-caused; this is the evidence gathered so far and the plan for what's next.

## Symptom

A Falcon3-7B-Instruct server (`~/scripts/model_servers/launch_falcon3_7b_instruct_uvicorn.sh`, bare-metal uvicorn, tt-xla-2 venv) hangs during serving. Originally only seen single-layer (`NUM_HIDDEN_LAYERS=1`); **confirmed 2026-07-28 to also reproduce at full model depth** (see the live-caught occurrence below) — so this is NOT specific to the single-layer debug config, resolving one of the long-standing open questions.

| Occurrence | PID (EngineCore) | Config | Caught live? |
|---|---|---|---|
| 1 | 3794128 | single layer, default launch script settings (bfp8 weights + bfp8 KV, gmu=0.35) | no, found already-wedged |
| 2 | 410948 | single layer, bf16 weights (`WEIGHT_DTYPE=`), bf16 KV (`KV_CACHE_DTYPE=none`), gmu=0.20 | no, found already-wedged |
| 3 (`tt-inference-server-2` log, 2026-07-22) | unknown | single layer, default settings | no, found already-ended (log just stops) |
| **4** | **875071** | **full model** (no `NUM_HIDDEN_LAYERS`), default settings, real tt-media-server | **yes — first live catch, 2026-07-28, full py-spy+gdb+tt-smi capture below** |

Same native signature across all four (see below) — the hang does not appear tied to weight/KV dtype, memory-utilization, or model depth.

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

## Next steps (planned, not yet run)

- **Cherry-picked vLLM scheduler debug prints** (already pulled into this tt-xla-2 checkout): expected to mainly *corroborate* rather than find root cause here — the evidence above (active spin + live mid-step stack, no scheduler frames anywhere) already points away from the scheduler. Still worth having for cheap confirmation that the scheduler is behaving normally leading up to the hang.
- **Per-op sync + debug logging** (the #4521 methodology: disable trace, instrument every device op with a forced full sync + log after each one, to catch which op is stuck mid-execution): expected to be the more useful next step for *this* hang. In #4521 this technique famously caught nothing stuck (because there was no real device stall, just scheduler idling) — but here we already have direct evidence of a genuine active spin inside a device completion-queue read, so per-op instrumentation should be able to identify exactly which op/buffer-read never completes, unlike in the #4521 case.

## Open questions

- Does this reproduce on the full (non-single-layer) model, or is it specific to the truncated `NUM_HIDDEN_LAYERS=1` debug config?
- Reproducible on demand, or only after some period of serving (both captured occurrences were found already-hung, not caught at the moment of onset)?
- Any correlation with a specific request pattern (e.g. the `attack_falcon3_7b_instruct_lite.sh`/`_loop.sh` scripts written this session), or does it also occur under normal/light serving? — **see the admission-driver comparison below; this is now the leading theory.**

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

### Standalone tt-xla verdict: does not reproduce (as of 2026-07-27)

Three tested admission patterns — smooth lm_eval concurrency=32 (~45min, 10 iterations, thousands of requests), bursty fire-and-forget up to 128 in-flight (~37min, 20,736 completions), and an escalated bursty pass up to 566 in-flight (~15min, 6,956+ completions) — all ran **clean, zero failures, no hang** against `serve_falcon3_7b_forge.sh` (stock `vllm serve`, single layer, byte-identical config to a real tt-media-server run). Per the handoff doc's own framing, **a clean standalone run is evidence, not proof** — it does not clear tt-media-server's actual driver/process/transport stack, only vLLM's native entry point under the admission shapes tested here.

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
