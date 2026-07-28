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

**Current leading theory — AND-gate, fits all 4 data points so far (first theory that hasn't been weakened by the next run):**

| Config | Non-decode buckets | b1-prefill | Result |
|---|---|---|---|
| Baseline (occurrences 4,5,6) | 4 (`128,256,512,1024`) | ON | **HANGS, 3/3** |
| `MIN_NUM_SEQS=32 THRESH=0` alone | 4 (full ladder) | OFF | Clean |
| `MIN_NUM_SEQS=32 THRESH=0 MIN_CONTEXT_LEN=512` | 2 (`512,1024`) | OFF | Clean |
| `PREFILL_CHUNK_SIZE=256` (b1 on) | 2 (`128,256`) | ON | Clean |

Neither "b1-prefill alone" nor "bucket-count alone" explains all four cells — but **both being present simultaneously** does: the hang needs *both* b1-prefill-driven single-request admission bursts (the recurring `num_reqs=1` pattern in every gdb/py-spy capture) *and* enough bucket diversity (≥3-4 distinct prefill sizes cycling in the same burst) at once. Remove either ingredient and it doesn't manifest. All three non-hanging runs each removed at least one of the two factors.

- `✅` **Re-confirm the untouched baseline still hangs reliably** — reproduced again at ifeval downsampled to `0.25` (136 docs, 2m33s) and again at `0.10` (55 docs, 1m30s). Baseline reproducibility holds at 6/6 now; loop is much faster (ifeval down to ~1.5min from ~4-11min). **Correction**: the client always *displaying* `17/198` at the point it goes silent is **not** a genuine "always the same request count" signal — see the buffered-client-output finding below. Ignore the specific number; it's an artifact of buffering, not a real stall point.
- `✅` **`17/198` was a stale/buffered display artifact, not the true count — confirmed via server-side counting.** The eval client's tqdm progress line is delayed reaching the log file (same buffering mechanism first spotted in occurrence 6's 29-minute-delayed `TimeoutError`), so whatever's last visible when it "goes silent" is not the true progress. **Reliable check**: count `POST /v1/completions 200 OK` in the *server's* log after the eval-transition point — uvicorn writes this synchronously when a response is actually sent, unaffected by the client's buffering. Done for the `ds_0.10` run (`falcon_serve_full_model_debug_scheduler_prints_more2.log`): split at line 750 (gpqa's first `[NEW-REQUEST]`) → 56 completions before (ifeval), **28 after (gpqa)** — not the displayed `17`. No invasive action needed (no killing the client, no waiting ~30min for its timeout) — just grep the server log.
- `🔲` If baseline still hangs: the AND-gate theory predicts a **partial** fix (e.g. 3 buckets instead of 2, or a milder threshold) should land *between* clean and hung — a good discriminating test once the fast baseline loop is in hand.
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
