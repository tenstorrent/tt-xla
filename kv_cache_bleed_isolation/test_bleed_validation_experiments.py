#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Validation experiments for KV cache bleed bug (#3899).

Demonstrates three properties of the SDPA decode causal mask leak:
  1. Equal-length prompts don't bleed; variable-length prompts do
  2. Bleed occurs between same-length prompts (not longer->shorter)
  3. Bleed scales with batch size (batch=32 hits massively)

How batching works in these tests:
  llm.generate(PROMPTS, sp) passes ALL prompts in a single call. vLLM's
  engine batches them together: during decode, all users run in one model
  forward call with batch dimension = len(PROMPTS). The SDPA decode kernel
  processes all users simultaneously in a single kernel invocation. The
  outer for-loop just repeats this batched generation to measure failure rate.
  This is NOT sequential single-user processing — it's true multi-user
  batched inference, the same as production serving.

Usage:
  python3 kv_cache_bleed_isolation/test_bleed_validation_experiments.py       # all experiments
  python3 kv_cache_bleed_isolation/test_bleed_validation_experiments.py 1     # experiment 1 only
  python3 kv_cache_bleed_isolation/test_bleed_validation_experiments.py 1 2   # experiments 1 and 2

Requires: vLLM with TT backend, Llama-3.2-1B-Instruct model cached.
"""

import os
import sys
from collections import Counter

import vllm

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

KEYWORDS_8 = [
    "penguin",
    "volcano",
    "origami",
    "submarine",
    "dinosaur",
    "chocolate",
    "castle",
    "galaxy",
]

# Variable-length prompts (14-19 tokens) — known to trigger bleed
PROMPTS_VARIABLE = [
    "Explain how penguins survive in Antarctica. Always use the word penguin.",
    "Explain how volcanoes form islands. Always use the word volcano.",
    "Describe how to make an origami crane. Always use the word origami.",
    "Describe how submarine sonar works. Always use the word submarine.",
    "Explain how dinosaurs went extinct. Always use the word dinosaur.",
    "Describe different types of chocolate and their flavors. Always use the word chocolate.",
    "Explain how castles were built in the Middle Ages. Always use the word castle.",
    "Describe different types of galaxies in the universe. Always use the word galaxy.",
]

# Equal-length prompts (all 17 tokens) — should NOT trigger bleed.
# When all prompts are the same length, there's no differential padding
# between users, so the leaked padding is identical and doesn't cause bleed.
PROMPTS_EQUAL = [
    "Explain how penguins survive in Antarctica. Always use the word penguin.",
    "Explain how volcanoes form the new islands. Always use the word volcano.",
    "Describe how to make an origami crane. Always use the word origami.",
    "Describe how a submarine uses sonar to detect things. Always use word submarine.",
    "Explain how all of the dinosaurs went fully extinct. Always use word dinosaur.",
    "Describe all of the many different types of good chocolate. Always use word chocolate.",
    "Explain how a castle was built long ago. Always use the word castle.",
    "Describe all of the many different types of known galaxies. Always use word galaxy.",
]

# 32 prompts for batch=32 experiment
PROMPTS_32 = PROMPTS_VARIABLE + [
    "Explain how penguins huddle together for warmth. Always use the word penguin.",
    "Explain how volcanoes erupt with hot lava. Always use the word volcano.",
    "Describe the art of origami paper folding. Always use the word origami.",
    "Describe how a submarine dives underwater. Always use the word submarine.",
    "Explain why dinosaurs were so large. Always use the word dinosaur.",
    "Describe how chocolate is made from cacao. Always use the word chocolate.",
    "Explain the architecture of a medieval castle. Always use the word castle.",
    "Describe the Milky Way galaxy and its structure. Always use the word galaxy.",
    "Explain penguin mating rituals in detail. Always use the word penguin.",
    "Explain what happens when a volcano is dormant. Always use the word volcano.",
    "Describe origami techniques for beginners. Always use the word origami.",
    "Describe submarine navigation systems. Always use the word submarine.",
    "Explain the different eras of dinosaur existence. Always use the word dinosaur.",
    "Describe white chocolate versus dark chocolate. Always use the word chocolate.",
    "Explain how castle moats were constructed. Always use the word castle.",
    "Describe spiral galaxies and their formation. Always use the word galaxy.",
    "Explain how penguins feed their young chicks. Always use the word penguin.",
    "Explain volcanic islands like Hawaii. Always use the word volcano.",
    "Describe making an origami flower step by step. Always use the word origami.",
    "Describe modern nuclear submarine technology. Always use the word submarine.",
    "Explain dinosaur fossil excavation methods. Always use the word dinosaur.",
    "Describe the history of chocolate in Europe. Always use the word chocolate.",
    "Explain how castles defended against sieges. Always use the word castle.",
    "Describe elliptical galaxies and their properties. Always use the word galaxy.",
]
KEYWORDS_32 = KEYWORDS_8 * 4


def check_bleed(responses, keywords):
    """Returns list of (source_keyword, source_slot, victim_slot) tuples."""
    bleeds = []
    for i, resp in enumerate(responses):
        for j, kw in enumerate(keywords):
            if j != i and kw.lower() in resp.lower():
                bleeds.append((kw, j, i))
    return bleeds


def create_engine(batch_size, min_context_len=32):
    return vllm.LLM(
        model=MODEL,
        max_model_len=512,
        max_num_batched_tokens=512 * batch_size,
        max_num_seqs=batch_size,
        gpu_memory_utilization=0.05,
        disable_log_stats=True,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": min_context_len,
            "cpu_sampling": True,
        },
    )


def run_experiment_1():
    """Equal-length prompts should not bleed; variable-length should."""
    print("=" * 70)
    print("EXPERIMENT 1: Equal vs variable prompt lengths")
    print("  Config: min_context_len=32, batch=8, Llama-3.2-1B")
    print("=" * 70)

    # Verify prompt lengths
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        eq_lens = [len(tok.encode(p)) for p in PROMPTS_EQUAL]
        var_lens = [len(tok.encode(p)) for p in PROMPTS_VARIABLE]
        print(f"  Equal prompts token lengths:    {eq_lens}")
        print(f"  Variable prompts token lengths: {var_lens}")
    except ImportError:
        print("  (transformers not available, skipping length check)")

    llm = create_engine(batch_size=8, min_context_len=32)
    llm.generate(["Hello"], vllm.SamplingParams(max_tokens=8, temperature=0.0))
    sp = vllm.SamplingParams(max_tokens=32, temperature=0.0)

    # 1a: Equal-length
    fails_eq = 0
    for run in range(20):
        outputs = llm.generate(PROMPTS_EQUAL, sp)
        responses = [o.outputs[0].text for o in outputs]
        if check_bleed(responses, KEYWORDS_8):
            fails_eq += 1
    print(f"  Equal-length:    {fails_eq}/20 failures")

    # 1b: Variable-length
    fails_var = 0
    for run in range(20):
        outputs = llm.generate(PROMPTS_VARIABLE, sp)
        responses = [o.outputs[0].text for o in outputs]
        if check_bleed(responses, KEYWORDS_8):
            fails_var += 1
    print(f"  Variable-length: {fails_var}/20 failures")
    print()


def run_experiment_2():
    """Bleed always occurs between same-length prompts."""
    print("=" * 70)
    print("EXPERIMENT 2: Corruption direction (longer->shorter? same length?)")
    print("  Config: min_context_len=32, batch=8, Llama-3.2-1B")
    print("=" * 70)

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        lengths = {
            kw: len(tok.encode(p)) for kw, p in zip(KEYWORDS_8, PROMPTS_VARIABLE)
        }
        print(f"  Prompt lengths: {lengths}")
    except ImportError:
        lengths = {}

    llm = create_engine(batch_size=8, min_context_len=32)
    llm.generate(["Hello"], vllm.SamplingParams(max_tokens=8, temperature=0.0))
    sp = vllm.SamplingParams(max_tokens=64, temperature=0.0)

    all_bleeds = []
    for run in range(30):
        outputs = llm.generate(PROMPTS_VARIABLE, sp)
        responses = [o.outputs[0].text for o in outputs]
        for src_kw, src_slot, vic_slot in check_bleed(responses, KEYWORDS_8):
            src_len = lengths.get(src_kw, "?")
            vic_len = lengths.get(KEYWORDS_8[vic_slot], "?")
            if isinstance(src_len, int) and isinstance(vic_len, int):
                if src_len > vic_len:
                    direction = "longer->shorter"
                elif src_len < vic_len:
                    direction = "shorter->longer"
                else:
                    direction = "same_length"
            else:
                direction = "unknown"
            all_bleeds.append(
                (src_kw, KEYWORDS_8[vic_slot], src_len, vic_len, direction)
            )

    print(f"  Total bleed instances: {len(all_bleeds)}")
    directions = Counter(b[4] for b in all_bleeds)
    print(f"  Direction counts: {dict(directions)}")
    pairs = Counter((b[0], b[1]) for b in all_bleeds)
    print(f"  Top bleed pairs:")
    for (src, vic), count in pairs.most_common(5):
        src_l = lengths.get(src, "?")
        vic_l = lengths.get(vic, "?")
        print(f"    {src}({src_l}tok) -> {vic}({vic_l}tok): {count} times")
    print()


def run_experiment_3():
    """Bleed scales with batch size — batch=32 hits massively."""
    print("=" * 70)
    print("EXPERIMENT 3: Batch=32")
    print("  Config: min_context_len=32, batch=32, Llama-3.2-1B")
    print("=" * 70)

    llm = create_engine(batch_size=32, min_context_len=32)
    llm.generate(["Hello"], vllm.SamplingParams(max_tokens=8, temperature=0.0))
    sp = vllm.SamplingParams(max_tokens=32, temperature=0.0)

    for run in range(5):
        outputs = llm.generate(PROMPTS_32, sp)
        responses = [o.outputs[0].text for o in outputs]
        bleeds = check_bleed(responses, KEYWORDS_32)
        if bleeds:
            print(f"  Run {run}: FAIL ({len(bleeds)} bleed instances)")
        else:
            print(f"  Run {run}: PASS")
    print()


if __name__ == "__main__":
    experiments = [1, 2, 3]
    if len(sys.argv) > 1:
        experiments = [int(x) for x in sys.argv[1:]]

    for exp in experiments:
        if exp == 1:
            run_experiment_1()
        elif exp == 2:
            run_experiment_2()
        elif exp == 3:
            run_experiment_3()
