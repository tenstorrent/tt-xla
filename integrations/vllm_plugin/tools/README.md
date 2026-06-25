<!--
SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Live inference dashboard for TT vLLM serving

Two complementary ways to *see* what a TT vLLM server is doing while it serves:
per-request/per-slot state, decode rate, prefill, and how new requests perturb
in-flight streams.

| | Approach A — client | Approach B — engine telemetry |
|---|---|---|
| `--source` | `client` | `snapshot` |
| Server changes | **none** | env-gated hooks in `vllm_tt` |
| State | *inferred* from SSE token cadence | *true* slots / PREFILL vs DECODE / waiting |
| Interference | inferred from rate hitches | read from the scheduler batch |

Both feed the **same** renderer and in-memory model, so switching approaches is
one `--source` flag. There is also a `--source demo` that synthesizes data with
no server at all — start there.

## Quick start (no server needed)

```bash
# Synthetic data: watch new arrivals' prefill stall in-flight decode.
python3 integrations/vllm_plugin/tools/live_dashboard.py --source demo
```

In a tty you get keys: `n` launch one, `b` burst, `1`..`9` launch that many,
`k` cancel newest, `t` counter/text, `p` prefill highlight, `q` quit. Fire a
burst while streams decode and watch every running stream's tok/s dip during the
burst's prefill window — that hitch *is* the visualization.

Renderer: uses [`rich`](https://github.com/Textualize/rich) if installed,
otherwise a stdlib ANSI fallback (or force it with `--plain`). No `textual`
dependency.

## Approach A — play against a running server (zero server changes)

```bash
# Start any example server (defaults to port 8000):
bash examples/vllm/Llama-3.1-8B-Instruct/service.sh

# Then, interactively fire requests and watch interference:
python3 integrations/vllm_plugin/tools/live_dashboard.py \
    --source client --port 8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --temperature 0.7 --seed 0 --repetition-penalty 1.0 --max-tokens 200
```

Each launch uses a distinct nonce-prefixed prompt to defeat prefix caching (else
TTFT is fake on repeats). Set sampling explicitly so A/B comparisons compare what
you intend. `--completions` switches to `/v1/completions`; default is
`/v1/chat/completions`. `--start N` auto-launches N at startup; `--ignore-eos`
forces the full output length.

**Honest limits (Approach A):** cannot distinguish prefill from queue-wait (both
= "no token yet", labelled `PREFILL`); no true slot IDs, KV usage, or explicit
preemption. Interference is *inferred* from rate, not read from the server. Use
Approach B for ground truth.

## Approach B — true engine state (telemetry)

Launch the server with telemetry on (the wrapper sets an absolute
`TT_INSTRUMENT_DIR` so the EngineCore subprocess writes where the dashboard
reads):

```bash
examples/vllm/serve_instrumented.sh Llama-3.1-8B-Instruct        # batch_size=1
# prints the telemetry dir + the exact dashboard command, e.g.:
python3 integrations/vllm_plugin/tools/live_dashboard.py \
    --source snapshot --dir /tmp/tt_instrument/Llama-3.1-8B-Instruct
```

This shows real slot indices, real `PREFILL`/`DECODE` per slot, ISL and output
length straight from `InputBatch`, the engine-computed per-slot decode rate, and
(when available) `num_waiting`.

The emitter lives in `vllm_tt/instrumentation.py` and is **off unless
`TT_INSTRUMENT=1`**. When off, every hook is a single cached-bool no-op. It is
stdlib-only and strictly `O(active_slots)` per step (snapshots are throttled by
`TT_INSTRUMENT_THROTTLE_MS`, default 100 ms) — telemetry must never reintroduce
a per-step host regression (cf. #4278).

### Env vars
| var | default | meaning |
|---|---|---|
| `TT_INSTRUMENT` | unset | `1`/`true`/`yes`/`on` enables emission |
| `TT_INSTRUMENT_DIR` | `./.tt_instrument` | sink directory (use absolute for servers) |
| `TT_INSTRUMENT_THROTTLE_MS` | `50` | min gap between step snapshots |
| `TT_INSTRUMENT_EVENTS` | `1` | also append admit/complete to `events.jsonl` |
| `TT_INSTRUMENT_SNAPSHOTS_JSONL` | `0` | also append each snapshot to `events.jsonl` |

## The schema (the A↔B contract)

`schema` version is `instrumentation.SCHEMA_VERSION`. Two sinks in
`TT_INSTRUMENT_DIR`:

- **`snapshot.json`** — latest `step_snapshot` only, atomically overwritten. A
  live viewer polls this for *current* state.
- **`events.jsonl`** — append-only `request_admitted` + `request_completed`
  (truncated at each fresh run). For offline analysis / request-mix audits.

```jsonc
// request_admitted  (events.jsonl)
{"schema":1,"event":"request_admitted","ts":<float>,"req_id":"...","slot_idx":3,
 "isl":512,"sampling":{"temperature":0.7,"top_k":20,"top_p":0.95,"min_p":0.0,
   "repetition_penalty":1.1,"presence_penalty":0.0,"frequency_penalty":0.0,
   "seed":7,"max_tokens":128,"n":1}}

// step_snapshot  (snapshot.json; one per step, throttled)
{"schema":1,"event":"step_snapshot","ts":<float>,"step_idx":42,"num_running":3,
 "num_waiting":1,"agg_rate":63.2,
 "slots":[{"slot_idx":0,"req_id":"...","state":"DECODE","num_prompt_tokens":512,
   "num_computed_tokens":512,"out_len":37,"inst_rate":21.1}, ...]}

// request_completed  (events.jsonl)
{"schema":1,"event":"request_completed","ts":<float>,"req_id":"...","slot_idx":0,
 "isl":512,"out_len":200}
```

`step_snapshot.slots[*]` maps directly onto the dashboard's `StreamState`, which
is why one renderer serves all three sources. Note `request_completed` has no
`finish_reason` (not available at `InputBatch.remove_request`); a viewer derives
TTFT/mean-rate from the admit event + the snapshots it already saw.

`tt-inference-server` (or any downstream) can consume these files directly — the
data stream is the shared interface, not the TUI.

## Tests

```bash
# Emitter contract — no hardware/server (runs anywhere):
pytest -q integrations/vllm_plugin/test_instrumentation.py

# Dashboard renders headless (any source) via --exit-after:
python3 integrations/vllm_plugin/tools/live_dashboard.py --source demo --plain --exit-after 8
```

### Morning checklist (needs the rebuilt server)
1. **A, single stream:** start `service.sh`, `--source client`, press `n` a few
   times; confirm `CONNECTING → PREFILL → DECODE → DONE` and that the client
   tok/s looks sane vs the server log.
2. **A, interference:** with one stream mid-decode, `b` a burst → the in-flight
   stream's tok/s should visibly dip during the burst's prefill window.
3. **A, prefix cache:** identical prompts → suspiciously low TTFT; the tool's
   nonce'd prompts should give realistic TTFT.
4. **A, cancel:** `k` aborts the HTTP stream server-side (request count stops).
5. **B, ground truth:** relaunch via `serve_instrumented.sh`, `--source
   snapshot`; confirm real slot indices, PREFILL/DECODE, and that
   `request_completed.out_len` ≈ what the client saw.
6. **B, perf guard:** run a `tests/benchmark` decode sweep with `TT_INSTRUMENT`
   on vs off — decode tok/s must be unchanged (the #4278 lesson). If snapshot
   writes show up, raise `TT_INSTRUMENT_THROTTLE_MS`.
