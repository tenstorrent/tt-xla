#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Minimal KV cache bleed reproducer.
#
# One llm.generate() call with 8 prompts on distinct topics.
# Checks if any response contains a keyword from a different topic.
#
# Usage:
#   python3 repro_bleed.py
#   MODEL=facebook/opt-125m python3 repro_bleed.py --num-runs 50

import argparse
import os
import time
from datetime import datetime

import vllm

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.2-1B-Instruct")

# 8 prompts on distinct topics. Selected from known-failing combination
# for Llama-3.2-1B-Instruct (58% failure rate at batch 8, max_model_len 512).
# fmt: off
PROMPTS = [
    "Explain how penguins survive in Antarctica. Always use the word penguin.",                       # slot 0: penguin
    "Explain how volcanoes form islands. Always use the word volcano.",                                # slot 1: volcano
    "Describe how to make an origami crane. Always use the word origami.",                             # slot 2: origami
    "Describe how submarine sonar works. Always use the word submarine.",                              # slot 3: submarine
    "Explain how dinosaurs went extinct. Always use the word dinosaur.",                               # slot 4: dinosaur
    "Describe different types of chocolate and their flavors. Always use the word chocolate.",         # slot 5: chocolate
    "Explain how castles were built in the Middle Ages. Always use the word castle.",                  # slot 6: castle
    "Describe different types of galaxies in the universe. Always use the word galaxy.",               # slot 7: galaxy
]
KEYWORDS = ["penguin", "volcano", "origami", "submarine", "dinosaur", "chocolate", "castle", "galaxy"]
# fmt: on


def check_bleed(responses):
    bleed_found = False
    for i, response in enumerate(responses):
        for j, kw in enumerate(KEYWORDS):
            if i != j and kw.lower() in response.lower():
                print(f"  BLEED: '{kw}' (slot {j}/{KEYWORDS[j]}) found in slot {i}/{KEYWORDS[i]}")
                print(f"    Response: {response[:150]}...")
                bleed_found = True
    return bleed_found


def main():
    parser = argparse.ArgumentParser(description="Minimal KV cache bleed reproducer")
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    batch_size = len(PROMPTS)

    print(f"Model:         {MODEL}")
    print(f"Batch size:    {batch_size}")
    print(f"Runs:          {args.num_runs}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Max tokens:    {args.max_tokens}")
    print()

    llm = vllm.LLM(
        model=MODEL,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len * batch_size,
        max_num_seqs=batch_size,
        gpu_memory_utilization=0.05,
        disable_log_stats=True,
        additional_config={
            "enable_const_eval": True,
            "min_context_len": 32,
            "experimental_weight_dtype": "bfp8",
            "cpu_sampling": True,
        },
    )

    sampling_params = vllm.SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    # Warmup
    llm.generate(["Hello"], vllm.SamplingParams(max_tokens=8, temperature=0.0))
    print("Ready.\n")

    passes = 0
    fails = 0
    start = time.perf_counter()

    for run in range(1, args.num_runs + 1):
        t0 = time.perf_counter()
        outputs = llm.generate(PROMPTS, sampling_params)
        dt = time.perf_counter() - t0
        responses = [o.outputs[0].text for o in outputs]

        ts = datetime.now().strftime("%H:%M:%S")
        if check_bleed(responses):
            fails += 1
            print(f"[{ts}] Run {run}/{args.num_runs} ({dt:.1f}s) FAIL\n")
        else:
            passes += 1
            print(f"[{ts}] Run {run}/{args.num_runs} ({dt:.1f}s) PASS  [{passes}P/{fails}F]")

    elapsed = time.perf_counter() - start
    print()
    print("============================================")
    print(f"Results: {passes} PASS / {fails} FAIL out of {args.num_runs}")
    if fails > 0:
        print(f"Failure rate: {fails / args.num_runs * 100:.1f}%")
    print(f"Total time: {elapsed:.0f}s ({elapsed / args.num_runs:.1f}s/run)")
    print("============================================")


if __name__ == "__main__":
    main()
