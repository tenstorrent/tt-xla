# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
"""Direct determinism probe against a running OpenAI-compatible server.

Isolates two questions that a full eval run conflates:

  1. SEQUENTIAL determinism -- same prompt, one at a time, temp 0. Greedy argmax
     must give byte-identical output every time.
  2. BATCH-COMPOSITION dependence -- the same prompt issued alone vs alongside N
     other in-flight requests. If output changes with batch composition, that is
     a different defect from (1).

Usage:  python determinism_probe.py [--port 8019] [--reps 5]
"""

import argparse
import concurrent.futures as cf
import json
import urllib.request

PROMPTS = [
    "Write a short paragraph about the ocean.",
    "List exactly 3 bullet points about trees. Use markdown bullets.",
    "Explain photosynthesis in two sentences.",
    "Write a JSON object describing a book. Output only JSON.",
]
FILLER = [f"Count from 1 to {n} and then stop." for n in range(5, 29)]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8019)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    print("=" * 78)
    print("1. SEQUENTIAL determinism (same prompt, one at a time, temp 0)")
    print("=" * 78)
    seq = {}
    for p in PROMPTS:
        outs = [complete(args.port, p) for _ in range(args.reps)]
        uniq = len(set(outs))
        seq[p] = outs[0]
        print(
            f"  reps={args.reps}  unique={uniq}  {'OK' if uniq == 1 else 'NON-DETERMINISTIC'}"
            f"   | {p[:44]}"
        )

    print()
    print("=" * 78)
    print(
        f"2. BATCH-COMPOSITION dependence (same prompt alone vs with {len(FILLER)} concurrent)"
    )
    print("=" * 78)
    for p in PROMPTS:
        with cf.ThreadPoolExecutor(max_workers=len(FILLER) + 1) as ex:
            target = ex.submit(complete, args.port, p)
            for f in FILLER:
                ex.submit(complete, args.port, f, 120)
            batched = target.result()
        same = batched == seq[p]
        print(
            f"  alone == batched: {str(same):>5}  {'OK' if same else 'BATCH-DEPENDENT'}"
            f"   | {p[:44]}"
        )
        if not same:
            a, b = seq[p], batched
            i = next(
                (k for k in range(min(len(a), len(b))) if a[k] != b[k]),
                min(len(a), len(b)),
            )
            print(f"      diverges at char {i}: alone={a[i:i+40]!r}")
            print(f"                            batched={b[i:i+40]!r}")


if __name__ == "__main__":
    main()
