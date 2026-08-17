# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""API load test for a live vLLM TT server.

A client-side load generator for the OpenAI-compatible HTTP API. Point it at a
running ``vllm serve`` (with telemetry enabled) to drive real serving load --
long prompts, many in flight -- while you watch the scheduler/runner telemetry
update live. Unlike the offline `generate()` demos, requests arrive over the
network and overlap, so this is what surfaces queueing, TTFT-under-load, and
KV pressure the way production traffic does.

Input and output lengths can be randomized per request (``--isl-min`` /
``--osl-min``) to model mixed traffic; use ``--dry-run`` to preview the plan.

    # terminal 1 -- serve with telemetry on (low flush = responsive live view)
    TTXLA_TELEMETRY=1 TTXLA_TELEMETRY_DIR=tt_telemetry/serve \
    TTXLA_TELEMETRY_FLUSH_MS=200 \
      vllm serve Qwen/Qwen3-0.6B --max-num-seqs 16 --max-model-len 2048 \
        --gpu-memory-utilization 0.2

    # terminal 2 -- live dashboard
    python scripts/telemetry/telemetry_viz.py live --dir tt_telemetry/serve

    # terminal 3 -- run the load test
    python examples/vllm/telemetry/api_load_test.py \
        --num-requests 64 --concurrency 16 --isl 1500 --max-tokens 128

Stdlib only (urllib + threads); no extra dependencies.
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Unique preamble + rotation per prompt, so prefix caching cannot collapse them
# into one shared prefix and erase the per-request KV cost.
SENTENCES = [
    "The turbine spun in the coastal wind under a bright and cloudless sky.",
    "A cartographer folded the old map along its worn and softened creases.",
    "Migrating cranes traced a long arc above the flooded rice paddies.",
    "The archivist catalogued each letter by date, sender, and faded postmark.",
    "Basalt columns stepped down to the sea in perfect hexagonal tiles.",
    "A luthier sanded the maple back until the grain caught the lamplight.",
    "The glacier calved with a report that rolled across the still fjord.",
    "Fireflies pulsed in slow waves along the edge of the summer meadow.",
]


def build_prompt(i, approx_tokens):
    """A distinct prompt of roughly `approx_tokens` tokens (~1.3 tokens/word)."""
    words_needed = max(1, int(approx_tokens / 1.3))
    base = (
        f"Passage {i}. "
        + " ".join(SENTENCES[(i + j) % len(SENTENCES)] for j in range(len(SENTENCES)))
    ).split()
    reps = -(-words_needed // len(base))  # ceil
    return " ".join((base * reps)[:words_needed])


def build_plan(rng, n, isl, isl_min, osl, osl_min, max_model_len):
    """Per-request (input_len, output_len) pairs, kept within the context window.

    When ``isl_min`` / ``osl_min`` are set and below the max, each request draws
    its length uniformly from [min, max] to model mixed traffic; otherwise the
    length is fixed at the max. Deterministic for a given rng seed.

    ``input + output`` is capped at ``max_model_len`` minus a slack margin: the
    server rejects a request whose prompt + max_tokens exceed the context
    window, and build_prompt only approximates the token count (BOS + tokenizer
    variance can push the real prompt a little past its target). Returns
    ``(plan, clamped)`` where ``clamped`` is True if any input length was capped.
    """
    slack = max(64, int(0.03 * max_model_len))
    budget = max_model_len - slack
    clamped = False
    plan = []
    for _ in range(n):
        o = rng.randint(osl_min, osl) if (osl_min and osl_min < osl) else osl
        o = max(1, min(o, budget - 1))  # always leave room for >=1 input token
        hi = min(isl, budget - o)
        if hi < isl:
            clamped = True
        lo = isl_min if (isl_min and isl_min < hi) else hi
        lo = max(1, min(lo, hi))
        i = rng.randint(lo, hi) if lo < hi else hi
        plan.append((max(1, i), o))
    return plan, clamped


def one_request(args, idx, isl, osl):
    """Fire one completion request; return timing + token counts (or an error).

    ``isl`` / ``osl`` are this request's input / output lengths (see build_plan).
    """
    url = args.url.rstrip("/") + "/completions"
    body = {
        "model": args.model,
        "prompt": build_prompt(idx, isl),
        "max_tokens": osl,
        "temperature": 0.0,
        "stream": args.stream,
    }
    if args.stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttft = None
    out_tokens = 0
    prompt_tokens = None
    try:
        resp = urllib.request.urlopen(req, timeout=args.timeout)
        if args.stream:
            chunks = 0
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    chunks += 1
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    out_tokens = usage.get("completion_tokens", out_tokens)
            if not out_tokens:  # usage not reported: approximate by chunk count
                out_tokens = chunks
        else:
            payload = json.loads(resp.read().decode())
            usage = payload.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            out_tokens = usage.get("completion_tokens", 0)
        return {
            "ok": True,
            "e2e": time.monotonic() - t0,
            "ttft": ttft,
            "out_tokens": out_tokens,
            "prompt_tokens": prompt_tokens,
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        return {"ok": False, "e2e": time.monotonic() - t0, "error": str(e)}


def _pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(results, wall):
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    e2e = [r["e2e"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
    out_total = sum(r["out_tokens"] or 0 for r in ok)
    in_toks = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
    out_toks = [r["out_tokens"] for r in ok if r.get("out_tokens")]

    print("\n=== load summary ===", flush=True)
    print(f"  completed / failed   : {len(ok)} / {len(fail)}")
    print(f"  wall time            : {wall:.1f} s")
    print(f"  request throughput   : {len(ok) / wall:.2f} req/s")
    print(f"  output token rate    : {out_total / wall:.1f} tok/s")
    if in_toks:
        print(f"  input tokens min/max : {min(in_toks)} / {max(in_toks)}")
    if out_toks:
        print(f"  output tokens min/max: {min(out_toks)} / {max(out_toks)}")
    if e2e:
        print(f"  e2e latency p50/p95  : {_pct(e2e, .5):.2f} / {_pct(e2e, .95):.2f} s")
    if ttfts:
        print(
            f"  TTFT p50/p95         : {_pct(ttfts, .5):.2f} / {_pct(ttfts, .95):.2f} s"
        )
    if fail:
        print(f"  first error          : {fail[0]['error']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL"
    )
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="served model name")
    ap.add_argument(
        "--num-requests", type=int, default=64, help="total requests to send"
    )
    ap.add_argument(
        "--concurrency", type=int, default=16, help="max requests in flight"
    )
    ap.add_argument(
        "--isl", type=int, default=1500, help="prompt tokens per request (the max)"
    )
    ap.add_argument(
        "--isl-min",
        type=int,
        default=None,
        help="if set (< --isl), draw each request's input length from [isl-min, isl]",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="tokens to generate per request (the max)",
    )
    ap.add_argument(
        "--osl-min",
        type=int,
        default=None,
        help="if set (< --max-tokens), draw each request's output length from "
        "[osl-min, max-tokens]",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="server context window; input+output per request is capped to fit it",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="RNG seed for ISL/OSL randomization"
    )
    ap.add_argument(
        "--stagger-ms", type=float, default=0.0, help="delay between launching requests"
    )
    ap.add_argument(
        "--timeout", type=float, default=600.0, help="per-request timeout (s)"
    )
    ap.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="disable streaming (no TTFT)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the ISL/OSL plan and exit without sending requests",
    )
    args = ap.parse_args()

    # Seeded on purpose: --seed must reproduce the same ISL/OSL plan, which is
    # what --dry-run previews. These draws are token lengths, never secrets.
    rng = random.Random(args.seed)
    plan, clamped = build_plan(
        rng,
        args.num_requests,
        args.isl,
        args.isl_min,
        args.max_tokens,
        args.osl_min,
        args.max_model_len,
    )
    isl_desc = f"{args.isl_min}-{args.isl}" if args.isl_min else str(args.isl)
    osl_desc = (
        f"{args.osl_min}-{args.max_tokens}" if args.osl_min else str(args.max_tokens)
    )
    if clamped:
        print(
            f"note: input length capped so input+output fits --max-model-len "
            f"{args.max_model_len} (pass a larger --max-model-len if the server "
            "allows it).",
            flush=True,
        )

    if args.dry_run:
        print(f"ISL/OSL plan (seed {args.seed}), ISL {isl_desc}, OSL {osl_desc}:")
        for i, (isl, osl) in enumerate(plan):
            print(f"  req {i:>3}: isl={isl:>5}  osl={osl:>4}")
        return

    print(
        f"Load testing {args.url} with {args.num_requests} requests "
        f"(ISL {isl_desc} tok, OSL {osl_desc} tok, concurrency "
        f"{args.concurrency}, stream={args.stream})",
        flush=True,
    )
    results = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = []
        for i in range(args.num_requests):
            futures.append(ex.submit(one_request, args, i, plan[i][0], plan[i][1]))
            if args.stagger_ms:
                time.sleep(args.stagger_ms / 1000.0)
        done = 0
        for f in as_completed(futures):
            results.append(f.result())
            done += 1
            if done % max(1, args.num_requests // 10) == 0:
                print(f"  {done}/{args.num_requests} done", flush=True)
    summarize(results, time.monotonic() - start)


if __name__ == "__main__":
    main()
