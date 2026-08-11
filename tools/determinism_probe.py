# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Direct determinism probe against a running OpenAI-compatible server.

Isolates questions that a full eval run conflates:

  1. SEQUENTIAL determinism -- same prompt, one at a time, temp 0. Greedy argmax
     must give byte-identical output every time.
  2. BATCH-COMPOSITION dependence -- the same prompt issued alone vs alongside N
     other in-flight requests. If output changes with batch composition, that is
     a different defect from (1).
  3. SWEEP -- (2) as a function of concurrency, reporting *where* the output
     diverges. A step change at a particular N names a padding bucket;
     divergence at any N >= 2 points at a per-slot reduction or KV block layout.
  4. PHASE -- (2) with the load split by phase: filler that is prefill-only
     (long prompt, max_tokens=1) vs filler that is decode-heavy (short prompt,
     long generation). Separates "a prefill got mixed into my decode batch"
     from "more decode slots in my batch".

Usage:  python determinism_probe.py [--port 8019] [--mode all]
"""

import argparse
import concurrent.futures as cf
import json
import threading
import urllib.request

PROMPTS = [
    "Write a short paragraph about the ocean.",
    "List exactly 3 bullet points about trees. Use markdown bullets.",
    "Explain photosynthesis in two sentences.",
    "Write a JSON object describing a book. Output only JSON.",
]
FILLER = [f"Count from 1 to {n} and then stop." for n in range(5, 29)]

# Decode-heavy: trivial prefill, long generation -- adds decode slots only.
FILLER_DECODE = [
    f"Write a long story about a {a}. Keep writing for many paragraphs."
    for a in ("dragon", "sailor", "clockmaker", "gardener", "pilot", "chemist")
]
# Prefill-heavy: long prompt, one output token -- forces prefill work to be
# scheduled alongside the target's decode steps.
FILLER_PREFILL = [
    (
        "Consider the following notes and reply with one word.\n"
        + f"Note {i}: "
        + ("the quick brown fox jumps over the lazy dog. " * 30)
    )
    for i in range(6)
]
SWEEP_LEVELS = [0, 1, 3, 7, 15, 23, 31]
# Finer levels either side of 16 in-flight, where PREFILL_BATCH_THRESHOLD sits.
SWEEP_LEVELS_FINE = [13, 14, 15, 16, 17, 19, 21]


def complete(port, prompt, max_tokens=160):
    body = json.dumps(
        {
            "model": "tiiuae/Falcon3-7B-Instruct",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer your-secret-key",
        },
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["text"]


def divergence(a, b):
    """First differing character index, or None if identical."""
    if a == b:
        return None
    for k in range(min(len(a), len(b))):
        if a[k] != b[k]:
            return k
    return min(len(a), len(b))


def under_load(port, prompt, n_filler, pool, filler_tokens):
    """Issue `prompt` while n_filler workers keep the server busy.

    Filler is re-issued in a loop rather than fired once, so the load is still
    present when the target reaches its later decode steps.
    """
    if n_filler == 0:
        return complete(port, prompt)
    stop = threading.Event()

    def churn(i):
        while not stop.is_set():
            try:
                complete(port, pool[i % len(pool)], filler_tokens)
            except Exception:
                return

    ex = cf.ThreadPoolExecutor(max_workers=n_filler + 1)
    try:
        # Submit the target FIRST so it is admitted to the batch. Fast-turnaround
        # filler (prefill-only, max_tokens=1) otherwise saturates the queue and
        # starves a target submitted behind it.
        target = ex.submit(complete, port, prompt)
        for i in range(n_filler):
            ex.submit(churn, i)
        return target.result()
    finally:
        # Don't block on in-flight filler: a heavy prefill arm can hold the
        # server for minutes and would otherwise stall or fail the whole probe.
        stop.set()
        ex.shutdown(wait=False)


def probe_sequential(port, reps):
    print("=" * 78)
    print("1. SEQUENTIAL determinism (same prompt, one at a time, temp 0)")
    print("=" * 78)
    base = {}
    for p in PROMPTS:
        outs = [complete(port, p) for _ in range(reps)]
        base[p] = outs[0]
        uniq = len(set(outs))
        print(
            f"  reps={reps}  unique={uniq}  {'OK' if uniq == 1 else 'NON-DETERMINISTIC'}"
            f"   | {p[:44]}"
        )
    return base


def probe_batch(port, base):
    print()
    print("=" * 78)
    print(f"2. BATCH-COMPOSITION dependence (alone vs {len(FILLER)} concurrent)")
    print("=" * 78)
    for p in PROMPTS:
        batched = under_load(port, p, len(FILLER), FILLER, 120)
        d = divergence(base[p], batched)
        print(
            f"  alone == batched: {str(d is None):>5}  "
            f"{'OK' if d is None else 'BATCH-DEPENDENT'}   | {p[:44]}"
        )
        if d is not None:
            print(f"      diverges at char {d}: alone={base[p][d:d+40]!r}")
            print(f"                            batched={batched[d:d+40]!r}")


def probe_sweep(port, base, trials, levels=SWEEP_LEVELS):
    print()
    print("=" * 78)
    print("3. CONCURRENCY SWEEP (divergence position vs filler count)")
    print("   step change at one level -> padding bucket; any N>=2 -> per-slot")
    print("=" * 78)
    print(f"  {'prompt':<34} " + " ".join(f"N={n:<5}" for n in levels))
    for p in PROMPTS[:2]:
        cells = []
        for n in levels:
            ds = [
                divergence(base[p], under_load(port, p, n, FILLER, 120))
                for _ in range(trials)
            ]
            hits = [d for d in ds if d is not None]
            cells.append("same " if not hits else f"@{min(hits):<4}")
        print(f"  {p[:34]:<34} " + " ".join(f"{c:<6}" for c in cells))


def probe_phase(port, base, n_filler, trials):
    print()
    print("=" * 78)
    print(f"4. LOAD PHASE (n_filler={n_filler}): prefill-only vs decode-heavy")
    print("   prefill-only diverges but decode-heavy does not -> mixed-batch")
    print("   chunked prefill; the reverse -> decode batch width")
    print("=" * 78)
    for label, pool, toks in (
        ("decode-heavy", FILLER_DECODE, 400),
        ("prefill-only", FILLER_PREFILL, 1),
        ("mixed", FILLER, 120),
    ):
        for p in PROMPTS[:2]:
            ds = []
            for _ in range(trials):
                try:
                    ds.append(
                        divergence(base[p], under_load(port, p, n_filler, pool, toks))
                    )
                except Exception as e:  # keep the remaining arms alive
                    print(f"  {label:<13} trial failed: {type(e).__name__}")
            hits = [d for d in ds if d is not None]
            verdict = (
                "no data"
                if not ds
                else "same" if not hits else f"DIVERGES @{min(hits)}"
            )
            print(f"  {label:<13} {verdict:<16} | {p[:44]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8019)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--trials", type=int, default=2, help="repeats per load level")
    ap.add_argument(
        "--n-filler", type=int, default=15, help="load level for --mode phase"
    )
    ap.add_argument(
        "--mode",
        default="all",
        choices=["all", "sequential", "batch", "sweep", "fine", "phase"],
    )
    args = ap.parse_args()

    base = probe_sequential(args.port, args.reps)
    if args.mode in ("all", "batch"):
        probe_batch(args.port, base)
    if args.mode in ("all", "sweep"):
        probe_sweep(args.port, base, args.trials)
    if args.mode == "fine":
        probe_sweep(args.port, base, args.trials, SWEEP_LEVELS_FINE)
    if args.mode in ("all", "phase"):
        probe_phase(args.port, base, args.n_filler, args.trials)


if __name__ == "__main__":
    main()
