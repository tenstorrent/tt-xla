<!--
SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# TT vLLM serving observability — design & roadmap

Tools to *see* what a TT vLLM server is doing — live and after the fact:
per-request/per-slot state, prefill vs decode, decode rate, batch occupancy, and
how new requests perturb in-flight ones. Educational, and an A/B rig for
comparing optimizations.

This file is the high-level overview + the running list of follow-up ideas.
For how to *run* the tools, see [README.md](README.md).

## Architecture

```
              ENGINE (vllm_tt, EngineCore process)
  add_request / remove_request / execute_model
                     │  (env-gated hooks, TT_INSTRUMENT=1)
                     ▼
        vllm_tt/instrumentation.py   ── the emitter (stdlib only, hot-path safe)
                     │  writes
        ┌────────────┴─────────────┐
        ▼                          ▼
  snapshot.json              events.jsonl
  (latest step, overwritten) (append-only: admit/complete [+ snapshots])
        │                          │
        │   the SCHEMA + file layout is the contract (not the UI)
        ▼                          ▼
  live_dashboard.py          analyze_run.py            (+ any downstream,
  (live TUI, 4 sources)      (offline summary + HTML)   e.g. tt-inference-server)
```

**Producer/consumer split:** the emitter is stdlib-only and never imports UI
deps; the consumers carry the optional deps (textual for the dashboard; the
analyzer is stdlib). The schema is the only coupling.

## Two approaches, four sources

- **Approach A — client (zero server changes).** Fire OpenAI requests, infer
  state from SSE token cadence. Sees the interference *symptom* (decode hitch)
  but no internals. Works against any OpenAI server.
- **Approach B — telemetry (ground truth).** The engine reports real per-slot
  state. Needs `TT_INSTRUMENT=1`.

`live_dashboard.py --source`:
| source | what | needs |
|---|---|---|
| `demo` | synthetic, exercises the UI | nothing |
| `client` | drive + infer (Approach A) | a server |
| `snapshot` | watch ground truth, read-only | instrumented server |
| `interactive` | drive (client) + show ground truth (snapshot) | instrumented server |

`analyze_run.py`: offline markdown summary + zoomable HTML slot timeline from
`events.jsonl`.

## Components
- `vllm_tt/instrumentation.py` — emitter + schema + env gate. 3 events
  (`request_admitted` / `step_snapshot` / `request_completed`), 2 sinks.
- 3 hooks: `input_batch.add_request`, `input_batch.remove_request`,
  `model_runner.execute_model`.
- `tools/live_dashboard.py` — live TUI (textual; stdlib `--plain` fallback).
- `tools/analyze_run.py` — offline analysis + HTML timeline (stdlib).
- `examples/vllm/serve_instrumented.sh` — convenience launcher (env-gating means
  it's optional; any launch path works by exporting the vars).
- Tests: `test_instrumentation.py`, `test_analyze_run.py` (stdlib, no hardware).

## Design principles
- **Off by default, cheap when on.** `TT_INSTRUMENT` gate is a cached bool;
  every hook is a guarded no-op when off.
- **O(active_slots) per step, never O(output_len)** (the #4278 lesson). Snapshot
  is throttled (`TT_INSTRUMENT_THROTTLE_MS`).
- **The schema is the interface.** Versioned (`SCHEMA_VERSION`); consumers read
  fields, downstreams can consume the files without the UI.
- **Renderer-agnostic model.** One `Model`/`StreamState`; sources and renderers
  are swappable.
- **Be honest about fidelity** (see below) — in code comments, the report, and
  the UI footer.

## Known limitations (by design, today)
- **Client mode infers**: can't tell prefill from queue-wait; no slot IDs / KV /
  preemption. Symptom, not cause.
- **Snapshots are sampled** (throttle): the state timeline and time-in-state are
  reconstructions; very short stalls can fall between samples. `scheduled` is
  authoritative only for captured steps.
- **`finish_reason` is heuristic** (`length` vs `stop` from `out_len` vs
  `max_tokens`) — the engine's real reason isn't available at `remove_request`.
- **No token *text* in telemetry** — counts only (`out_len`). Streamed text shows
  only in `--source client`.
- **Interactive can't correlate** a launched request to a snapshot row (OpenAI
  API hides the engine `req_id`); `k` cancels your newest launched request.
- **Slot grid / real STALLED need internals** — snapshot or demo only.
- **Same-filesystem** assumption for snapshot/analyzer (server + reader share the
  telemetry dir).
- **admit/complete fire per input-batch add/remove, not per request lifecycle.**
  Under b1-prefill / partial scheduling a request is evicted and re-added across
  steps, so its `req_id` recurs (a real log showed 93 admit events for 46
  requests). Consumers compensate: the dashboard guards on first-seen, and
  `analyze_run.py` dedups to first-admit / last-complete. True arrival/finish
  semantics belong in the emitter (see roadmap).
- **`num_slots` is a lower bound without snapshots** — inferred from slot indices
  seen in events (`≥N (inferred)`); the true batch size needs snapshot logging.

## Follow-up / improvement ideas

### Emitter / telemetry
- **True arrival/finish semantics**: admit/complete fire on every input-batch
  add/remove (a request recurs as it's evicted + re-added). Emit a distinct
  `request_arrived` (first add) and `request_finished` (real `finished_req_ids`)
  so consumers don't have to dedup; fold re-adds/removes into a transient event
  (or drop them). This is the top accuracy gap.
- **Exact `finish_reason`**: pass finished-vs-preempted into the completion hook
  (emit from `model_runner` where `finished_req_ids` is known) instead of the
  heuristic.
- **Preemption events**: distinguish preempt from complete (today both fire
  `remove_request`).
- **Per-slot KV/memory usage** in snapshots if the engine exposes it.
- **Accurate `num_waiting`**: model_runner only sees `scheduler_output`; surface
  the real scheduler queue depth.
- **Optional token text** behind a separate flag (needs detokenization; keep off
  the default hot path) for text replay.
- **Perf-guard test** (the #4278 lesson): assert decode tok/s is unchanged with
  `TT_INSTRUMENT` on vs off in a `tests/benchmark` sweep.
- **Lower-overhead sink** for very high throughput (batch writes / binary) if
  `events.jsonl` append shows up.

### Live dashboard
- **`--client-isl N`** (+ jitter): synthesize ~N-token prompts (repeat `"word "`,
  like the benchmark) so client/interactive can actually exercise prefill /
  chunked-prefill / the b1-prefill routing, not just tiny prompts.
- **req_id correlation** in interactive if vLLM can echo an `x-request-id`.
- **Remote/over-socket** snapshot transport (unix socket / tail) for servers not
  on the local filesystem.

### Offline analyzer
- **Textual TUI timeline** (phase 4) for in-terminal scroll/zoom.
- **HTML enhancements**: time-axis ticks/labels, per-request hover detail
  (ISL/OSL/TTFT), slot-lane vs request-lane toggle, zoom-to-fit, PNG export.
- **Aggregate charts** in the HTML: concurrency-over-time, tok/s-over-time, TTFT
  histogram.
- **Stall attribution**: link each decoder stall to the prefill that caused it.
- **Schema-version check** + migration notes for older logs.

### Integration
- **tt-inference-server** as a documented downstream consumer of the same files
  (reports / its own viewer) — no UI code needed there.
- **Upstreamability**: the emitter is plausibly upstreamable to the vLLM TPU
  plugin; the TUI/analyzer stay tt-xla dev tools.

## Status
Built and unit/headless-verified (emitter contract, stall detection, restart
attach, interactive split, analyzer metrics + HTML). **Not yet exercised against
a live instrumented TT server** — the end-to-end runtime behavior (snapshot
cadence, what a real prefill step's slot list contains, perf overhead) needs a
real run to confirm.
