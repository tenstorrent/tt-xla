# vLLM Serving Telemetry

Env-gated instrumentation for the vLLM TT serving path
([issue #5715](https://github.com/tenstorrent/tt-xla/issues/5715)). It exposes
the scheduling/runtime signals that are otherwise invisible when debugging
serving performance (KV thrashing, prefill stalling decode, batch under-fill),
as JSON-lines you can inspect or visualize.

Two layers are instrumented, because they answer different questions:

| Layer | Process | Answers |
|---|---|---|
| **Scheduler** (`AscendScheduler`) | EngineCore | *intent* — what each step decided: prefill vs decode, preemption, queue depth, KV/batch utilization |
| **Runner** (`TTModelRunnerV2` / `TTModelRunner`) | Worker | *reality* — slot occupancy, executed prefill/decode pass split, actual decode rate, per-request lifecycle |

The two run in separate processes and each writes its own sink; records carry
`request_id` (+ a monotonic `step`) so you can join them offline.

Telemetry is **off by default** and **zero-cost when off** (a single boolean
check on the hot path). When on, it never does per-step disk I/O — records
buffer in memory and flush on an interval, at request completion, and at exit.

## Enabling it

Two equivalent ways; **environment variables take precedence** over
`additional_config` (same pattern as `prefill_kv_watermark`).

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `TTXLA_TELEMETRY` | unset (off) | truthy (`1`/`true`/`yes`/`on`) enables emission |
| `TTXLA_TELEMETRY_DIR` | `./tt_telemetry` | output directory for the sinks |
| `TTXLA_TELEMETRY_FLUSH_MS` | `1000` | minimum gap between disk flushes |

```bash
TTXLA_TELEMETRY=1 TTXLA_TELEMETRY_DIR=./tt_telemetry \
  vllm serve Qwen/Qwen3-0.6B \
    --max-num-seqs 4 --max-model-len 1024 --max-num-batched-tokens 4096
```

### `additional_config` knobs

```python
llm = vllm.LLM(
    model="Qwen/Qwen3-0.6B",
    additional_config={
        "telemetry_enabled": True,
        "telemetry_dir": "tt_telemetry",
        "telemetry_flush_ms": 200,
    },
)
```

### Requirements

- **Either generative runner.** The collector lives in both
  `TTModelRunnerV2` and `TTModelRunner`, so no routing knob is needed;
  `use_v2_model_runner` selects between them and defaults to v2. Pooling models
  use `pooling_runner.py` and emit no runner telemetry.
- Scheduler-side telemetry is emitted for any non-pooling model (which use
  `AscendScheduler`), independent of the runner.

### Reading slot output

The two runners model slots differently, and it shows up in the sinks.

v2's `TTRequestState` is a persistent slot table: a request holds one slot for
its lifetime, so `slot` is a stable identity and a per-row timeline reads as one
request per row. Re-admission happens only on abort + resubmit.

v1's `InputBatch` condenses: when a request leaves, later rows compact down to
fill the gap, so a row index identifies a position rather than a request.
Occupancy, utilization, and the pass split stay exact; a per-row timeline shows
row **reuse**, with consecutive segments of one row belonging to different
requests. Join on `request_id` when following a single request.

Two consequences show up in v1 output. From a 4-prompt run on n150:

```
adm 0-94dd32b8 slot 0 step 0 prompt 8 cached 0 hit False readmission False
adm 1-a7a7ee6c slot 0 step 1 ...
adm 0-94dd32b8 slot 3 step 2 prompt 8 cached 8 hit True  readmission True
```

- **A request can be admitted more than once.** Unscheduling a request removes
  it from the persistent batch; it is re-added later, typically at a different
  row (here `0-94dd32b8` moves row 0 -> row 3). So `request_admitted` records
  outnumber requests — 5 records for 4 requests above. Counting requests served
  means counting distinct `request_id`, or filtering `readmission == false`.
  These records *are* the re-admission signal, useful in their own right.
- **A re-admitted request reports `prefix_cache_hit`** against its own
  already-computed prefix (`cached 8` of an 8-token prompt), which is a true
  cache hit but not the cross-request hit you are usually hunting. Filter on
  `readmission` when looking for the latter.

## Output files

Written under the telemetry directory:

| File | Contents |
|---|---|
| `scheduler.jsonl` | one record per `schedule()` step: `num_running`, `num_waiting`, `batch_util`, `kv_util`, prefill/decode counts, `preempted`, `decode_gated` (+ `decodes_displaced`), `watermark_rejects`, `b1_cap_hit`, and `cum_*` run totals |
| `runner.jsonl` | per-step records (`slots_occupied/free`, `num_prefilling/decoding`, `prefill_passes`, `decode_passes`, `emitted_tokens`, `decode_rate_toks_per_s`) plus `request_admitted` / `request_completed` events |
| `runner_snapshot.json` | latest per-slot state, atomically overwritten — a cheap "current state" view for live monitoring |

Decode rate is defined as **accepted tokens / step**, so it stays meaningful if
speculative decode lands.

## Quick start

```bash
source venv/activate
python examples/vllm/telemetry/run_telemetry_demo.py
```

Each demo writes to its own subfolder under `tt_telemetry/` by default
(`tt_telemetry/demo`, `.../oversubscribed`, `.../memory_pressure`); the whole
`tt_telemetry/` tree is git-ignored. Override with `--dir`.

This runs four prompts through Qwen3-0.6B with `max_num_seqs=4` (so all four
share the batch and decode concurrently), then prints a summary confirming the
sinks were written — e.g. peak slots occupied, decode rate, and how many steps a
prefill stalled in-flight decode.

Flags: `--dir` (output dir), `--max-num-seqs` (batch capacity), `--max-tokens`.

### Oversubscribed variant

[`run_telemetry_oversubscribed.py`](run_telemetry_oversubscribed.py) submits
**more requests than the batch has slots** (8 prompts, `max_num_seqs=4`), so the
scheduler admits four and the rest queue, getting admitted as running requests
finish:

```bash
python examples/vllm/telemetry/run_telemetry_oversubscribed.py
```

This is what exercises the queueing signals a full-batch run cannot. On an n150
it reports the batch pinned at 4 slots with a **max queue depth of 4** and
**7 of 8 requests waiting** for a slot before admission. (Preemption stays 0
here — that needs KV-cache pressure, not just a full batch; drive longer
sequences or a higher `gpu_memory_utilization` to provoke it.)

Flags: `--dir`, `--max-num-seqs`, `--num-prompts`, `--max-tokens`.

### Memory-pressure variant

[`run_telemetry_memory_pressure.py`](run_telemetry_memory_pressure.py) submits
requests with a **large ISL** (16 requests x 2048-token prompts by default), so
KV cache — not the slot count — becomes the binding constraint. Prompts are
built to an exact length and made distinct per request so prefix caching cannot
share their blocks.

```bash
python examples/vllm/telemetry/run_telemetry_memory_pressure.py
```

On an n150 (KV pool ~22.3k tokens, ~2.2k tokens per request) this shows the
fresh-prefill watermark at work: the batch **self-limits to 8 of 16 slots**,
**peak `kv_util` caps at ~75%** — exactly the default `prefill_kv_watermark`
reserve of 25% — and the scheduler logs hundreds of **`watermark_rejects`**
(declining to admit a fresh prefill so in-flight decodes keep their KV), with
**0 preemptions**. The watermark is trading admission latency for decode
stability; set `additional_config['prefill_kv_watermark']=0` to disable it and
watch rejects turn into preemptions instead.

Flags: `--dir`, `--isl`, `--max-num-seqs`, `--num-prompts`, `--max-tokens`,
`--gpu-mem-util` (lower shrinks the KV pool for more pressure).

## Visualizing

[`scripts/telemetry/telemetry_viz.py`](../../../scripts/telemetry/telemetry_viz.py) turns the sinks
into a single self-contained, dependency-free (stdlib-only), theme-aware
interactive HTML dashboard — slot-occupancy timeline, decode-rate line with a
prefill-stall event rug, KV/batch utilization, a per-request Gantt, the current
slot table, and run-total event chips.

```bash
# Post-hoc: write a standalone HTML report from a finished run
python scripts/telemetry/telemetry_viz.py report --dir tt_telemetry/memory_pressure
#   -> tt_telemetry/memory_pressure/report.html  (open in a browser; data embedded)

# Live: serve a dashboard that polls the sinks while a model is serving
python scripts/telemetry/telemetry_viz.py live --dir tt_telemetry/memory_pressure --port 8009
#   -> http://127.0.0.1:8009/     (slot state / decode rate update in place)
```

### Live monitoring

Live mode runs a small local web server that re-reads the sinks every
`--interval-ms` and re-renders in place, so you point it at whatever directory a
*running* serving process is writing to. It shines against the **online server**
(continuous traffic), not a one-shot `generate()` that finishes in seconds.

**Terminal 1 — serve with telemetry on.** The env vars are the gate; a low flush
keeps the view responsive (the viewer can't show data faster than the producer
flushes it — the 1000 ms default feels laggy):

```bash
source venv/activate
TTXLA_TELEMETRY=1 \
TTXLA_TELEMETRY_DIR=tt_telemetry/serve \
TTXLA_TELEMETRY_FLUSH_MS=200 \
  vllm serve Qwen/Qwen3-0.6B \
    --max-num-seqs 16 --max-model-len 2048 --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.2
```

**Terminal 2 — the viewer**, pointed at the *same* directory:

```bash
python scripts/telemetry/telemetry_viz.py live --dir tt_telemetry/serve --port 8009
#   -> open http://127.0.0.1:8009/  (over Remote-SSH, the port is auto-forwarded)
```

**Terminal 3 — send overlapping requests** with the load generator below (or
curl the OpenAI-compatible endpoint / any load tool). As traffic flows, the
dashboard shows slots filling, the queue forming, and the decode rate moving —
where you catch KV thrashing or prefill stalling decode as it happens. The
"live" badge flips to "stalled" once the files stop changing.

Notes: the serving process's telemetry dir and the viewer's `--dir` must match;
flags are `--dir`, `--port` (default 8009), `--interval-ms` (default 500).

### API load test

[`api_load_test.py`](api_load_test.py) drives a running server over the HTTP
API with many long, concurrent requests (stdlib only — no extra deps). Because
requests arrive over the network and overlap, it produces the queueing and
TTFT-under-load that offline `generate()` cannot.

```bash
python examples/vllm/telemetry/api_load_test.py \
    --num-requests 48 --concurrency 16 --isl 1200 --max-tokens 96
```

It streams (for TTFT) and prints a load summary — throughput, output token rate,
and e2e / TTFT p50/p95. On an n150 against a `max_num_seqs=16` server, 48
concurrent ~1200-token requests drive **peak `kv_util` ~80%**, **queue depth
14**, and **hundreds of `watermark_rejects`**, with **TTFT p95 ~100 s** — the
long TTFT is almost entirely admission wait behind the KV watermark, which the
telemetry makes legible. Prompts are made distinct per request so prefix caching
does not collapse their KV.

To model mixed traffic, randomize input/output lengths per request: `--isl-min`
draws each input length from `[isl-min, --isl]` and `--osl-min` draws each output
length from `[osl-min, --max-tokens]` (deterministic per `--seed`). Preview the
plan without sending anything via `--dry-run`:

```bash
python examples/vllm/telemetry/api_load_test.py \
    --num-requests 48 --concurrency 16 \
    --isl 2000 --isl-min 500 --max-tokens 128 --osl-min 16 --max-model-len 2048
```

Each request's `input + output` is capped to fit `--max-model-len` (default
2048; set it to match your server) — the server rejects a request that exceeds
its context window, and the prompt builder only approximates the token count, so
a slack margin is reserved. If a length is capped the tool prints a note.

Flags: `--url`, `--model`, `--num-requests`, `--concurrency`, `--isl`,
`--isl-min`, `--max-tokens`, `--osl-min`, `--max-model-len`, `--seed`,
`--stagger-ms`, `--timeout`, `--no-stream`, `--dry-run`.
