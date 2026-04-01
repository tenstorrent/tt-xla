#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Minimal unit-test-style KV cache bleed reproducer.
#
# Uses the known-failing configuration: batch 16, max_model_len 512,
# Llama-3.2-1B-Instruct with bfp8/consteval/cpu_sampling.
# 83% failure rate — dinosaur prompt bleeds into submarine slot.
#
# Can be run as a pytest test or standalone.
#
# Usage:
#   python3 test_minimal_bleed.py           # standalone, 30 runs
#   pytest test_minimal_bleed.py -v         # as pytest test

import os
import sys

import vllm

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.2-1B-Instruct")

# 16 prompts, each about a unique topic with a unique keyword.
# Slots 3 (submarine, 14 tokens) and 4 (dinosaur, 14 tokens) are the
# known-failing pair — dinosaur content bleeds into submarine responses.
# fmt: off
PROMPTS = [
    "Explain how penguins survive in Antarctica. Always use the word penguin.",
    "Explain how volcanoes form islands. Always use the word volcano.",
    "Describe how to make an origami crane. Always use the word origami.",
    "Describe how submarine sonar works. Always use the word submarine.",
    "Explain how dinosaurs went extinct. Always use the word dinosaur.",
    "Describe different types of chocolate and their flavors. Always use the word chocolate.",
    "Explain how castles were built in the Middle Ages. Always use the word castle.",
    "Describe different types of galaxies in the universe. Always use the word galaxy.",
    "Explain how dolphins communicate underwater. Always use the word dolphin.",
    "Describe how earthquakes are measured on the Richter scale. Always use the word earthquake.",
    "Explain the basic rules of cricket. Always use the word cricket.",
    "Describe how lightning forms in thunderstorms. Always use the word lightning.",
    "Explain how pyramids were built in ancient Egypt. Always use the word pyramid.",
    "Describe how telescopes work to observe distant stars. Always use the word telescope.",
    "Explain how glaciers shape mountain landscapes. Always use the word glacier.",
    "Describe how coral reefs form in tropical oceans. Always use the word coral.",
]
KEYWORDS = [
    "penguin", "volcano", "origami", "submarine", "dinosaur", "chocolate",
    "castle", "galaxy", "dolphin", "earthquake", "cricket", "lightning",
    "pyramid", "telescope", "glacier", "coral",
]
# fmt: on

BATCH_SIZE = len(PROMPTS)


def create_engine():
    return vllm.LLM(
        model=MODEL,
        max_model_len=512,
        max_num_batched_tokens=512 * BATCH_SIZE,
        max_num_seqs=BATCH_SIZE,
        gpu_memory_utilization=0.05,
        disable_log_stats=True,
        additional_config={
            "enable_const_eval": True,
            "min_context_len": 32,
            "experimental_weight_dtype": "bfp8",
            "cpu_sampling": True,
        },
    )


def check_bleed(responses):
    """Returns list of (source_slot, victim_slot, keyword) tuples for any bleed found."""
    bleeds = []
    for i, response in enumerate(responses):
        for j, kw in enumerate(KEYWORDS):
            if i != j and kw.lower() in response.lower():
                bleeds.append((j, i, kw))
    return bleeds


def run_once(llm):
    """Run one batched generate and return (responses, bleeds)."""
    sampling_params = vllm.SamplingParams(max_tokens=64, temperature=0.0)
    outputs = llm.generate(PROMPTS, sampling_params)
    responses = [o.outputs[0].text for o in outputs]
    bleeds = check_bleed(responses)
    return responses, bleeds


# ---- Pytest test ----


def test_kv_cache_no_bleed():
    """KV cache bleed test — fails if any cross-topic contamination in 30 batched runs.

    Known issue: ~83% failure rate at batch 16 on TT hardware.
    https://github.com/tenstorrent/tt-xla/issues/3899
    """
    num_runs = int(os.environ.get("NUM_RUNS", 30))
    llm = create_engine()
    # Warmup
    llm.generate(["Hello"], vllm.SamplingParams(max_tokens=8, temperature=0.0))

    all_bleeds = []
    for run in range(num_runs):
        _, bleeds = run_once(llm)
        if bleeds:
            for src, vic, kw in bleeds:
                all_bleeds.append(
                    f"Run {run+1}: '{kw}' (slot {src}/{KEYWORDS[src]}) "
                    f"found in slot {vic}/{KEYWORDS[vic]}"
                )

    assert len(all_bleeds) == 0, (
        f"KV cache bleed detected in {len(all_bleeds)} instances "
        f"across {num_runs} runs:\n" + "\n".join(all_bleeds[:10])
    )


# ---- Standalone mode ----

if __name__ == "__main__":
    import time
    from datetime import datetime

    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    print(f"Model:         {MODEL}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Runs:          {num_runs}")
    print()

    llm = create_engine()
    llm.generate(["Hello"], vllm.SamplingParams(max_tokens=8, temperature=0.0))
    print("Ready.\n")

    passes = 0
    fails = 0
    start = time.perf_counter()

    for run in range(1, num_runs + 1):
        t0 = time.perf_counter()
        responses, bleeds = run_once(llm)
        dt = time.perf_counter() - t0
        ts = datetime.now().strftime("%H:%M:%S")

        if bleeds:
            fails += 1
            for src, vic, kw in bleeds:
                print(
                    f"  BLEED: '{kw}' (slot {src}/{KEYWORDS[src]}) "
                    f"found in slot {vic}/{KEYWORDS[vic]}"
                )
                print(f"    Response: {responses[vic][:150]}...")
            print(f"[{ts}] Run {run}/{num_runs} ({dt:.1f}s) FAIL\n")
        else:
            passes += 1
            print(f"[{ts}] Run {run}/{num_runs} ({dt:.1f}s) PASS  [{passes}P/{fails}F]")

    elapsed = time.perf_counter() - start
    print()
    print("============================================")
    print(f"Results: {passes} PASS / {fails} FAIL out of {num_runs}")
    if fails > 0:
        print(f"Failure rate: {fails / num_runs * 100:.1f}%")
    print(f"Total time: {elapsed:.0f}s ({elapsed / num_runs:.1f}s/run)")
    print("============================================")

    sys.exit(1 if fails > 0 else 0)
