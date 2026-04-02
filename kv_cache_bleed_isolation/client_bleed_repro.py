#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Client for server-side KV cache bleed reproduction (#3899).

Sends 4 topic-specific prompts to a running vLLM server with staggered timing
(simulating real users arriving ~1 second apart). Checks responses for
cross-topic keyword contamination.

The bleed typically appears on the SECOND round of requests — the first round
is clean because cache blocks are fresh. When prefix caching is enabled, the
second round reuses cached prefix blocks that contain stale data from the
first round, causing contamination.

Usage:
  1. Start server:  ./server_bleed_repro.sh
  2. Run client:    python3 client_bleed_repro.py [--rounds N] [--port PORT]

Expected results:
  - With prefix caching (default server):  Round 2+ may BLEED
  - With --no-prefix-cache server flag:    All rounds CLEAN
"""

import argparse
import concurrent.futures
import sys
import time

import requests

KEYWORDS = ["penguin", "submarine", "dinosaur", "chocolate"]
PROMPTS = [
    "Explain how penguins survive in Antarctica. Always use the word penguin in every sentence.",
    "Describe how submarine sonar works underwater. Always use the word submarine in every sentence.",
    "Explain how dinosaurs went extinct millions of years ago. Always use the word dinosaur in every sentence.",
    "Describe how chocolate is made from cacao beans. Always use the word chocolate in every sentence.",
]


def send_request(url, model, prompt, idx):
    """Send a chat completion request and check for cross-topic bleed."""
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.6,
        },
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    foreign = [
        kw for j, kw in enumerate(KEYWORDS) if j != idx and kw.lower() in text.lower()
    ]
    return idx, text[:100], foreign


def run_round(url, model, delay=0.8):
    """Send 4 prompts with staggered timing, return list of (idx, text, foreign)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, prompt in enumerate(PROMPTS):
            futures.append(executor.submit(send_request, url, model, prompt, i))
            time.sleep(delay)
        return [f.result() for f in futures]


def main():
    parser = argparse.ArgumentParser(description="KV cache bleed repro client")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Seconds between requests within a round",
    )
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"

    # Check server is up
    try:
        requests.get(f"{url}/health", timeout=5)
    except requests.ConnectionError:
        print(f"Server not running at {url}. Start it first with server_bleed_repro.sh")
        sys.exit(1)

    total_bleeds = 0
    for round_num in range(1, args.rounds + 1):
        results = run_round(url, args.model, delay=args.delay)
        bleeds = [(idx, text, foreign) for idx, text, foreign in results if foreign]
        if bleeds:
            total_bleeds += 1
            print(f"ROUND {round_num}: BLEED")
            for idx, text, foreign in bleeds:
                print(f"  {KEYWORDS[idx]}: foreign={foreign} text={text[:60]}")
        else:
            print(f"ROUND {round_num}: CLEAN")
        time.sleep(1)

    print(f"\nResults: {total_bleeds}/{args.rounds} rounds had bleed")
    if total_bleeds > 0:
        print("FAIL — cross-user contamination detected")
        sys.exit(1)
    else:
        print("PASS — no contamination")


if __name__ == "__main__":
    main()
