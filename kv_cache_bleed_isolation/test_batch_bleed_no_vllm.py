#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Batched LLM inference WITHOUT vLLM — tests if cross-batch contamination
# exists at the torch.compile level with standard (non-paged) attention.
#
# Uses StaticCache + torch.compile(backend="tt"), same as examples/pytorch/llama.py.
# Runs many prefill+decode cycles with different prompt orderings to stress test
# batch slot isolation.
#
# If this test fails: the bug is in compiled graph batching, not paged attention.
# If this passes: the bug is specific to the paged KV cache execution pattern.
#
# Usage:
#   python3 test_batch_bleed_no_vllm.py
#   python3 test_batch_bleed_no_vllm.py --num-runs 50 --batch-size 8 --max-new-tokens 32

import argparse
import os
import random
import sys
import time
from datetime import datetime
from typing import List

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.2-1B-Instruct")

# Same prompts as the vLLM bleed test — distinct topics with unique keywords
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
    "penguin",
    "volcano",
    "origami",
    "submarine",
    "dinosaur",
    "chocolate",
    "castle",
    "galaxy",
    "dolphin",
    "earthquake",
    "cricket",
    "lightning",
    "pyramid",
    "telescope",
    "glacier",
    "coral",
]


def check_bleed(responses, keywords_used):
    """Check for cross-topic keyword contamination."""
    bleeds = []
    for i in range(len(responses)):
        for j, kw in enumerate(keywords_used):
            if i != j and kw.lower() in responses[i].lower():
                bleeds.append((j, i, kw))
    return bleeds


def main():
    parser = argparse.ArgumentParser(
        description="Batched LLM bleed test WITHOUT vLLM (standard StaticCache attention)"
    )
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-cache-len", type=int, default=128)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle prompt order each run for varied cache patterns",
    )
    args = parser.parse_args()

    batch_size = min(args.batch_size, len(PROMPTS))

    print(f"Model:          {MODEL}")
    print(f"Batch size:     {batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Max cache len:  {args.max_cache_len}")
    print(f"Runs:           {args.num_runs}")
    print(f"Shuffle:        {args.shuffle}")
    print(f"Cache type:     StaticCache (non-paged)")
    print()

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, use_cache=True
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = xm.xla_device()
    model = model.to(device)

    compiled_model = torch.compile(model, backend="tt")

    # Do a single-prompt warmup to trigger compilation
    print("Warmup (compiling prefill graph)...")
    warmup_prompts = PROMPTS[:batch_size]
    warmup_inputs = tokenizer(
        warmup_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    prompt_len = warmup_inputs.input_ids.shape[1]

    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=args.max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )
    for layer in cache.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)

    full_mask = torch.ones((batch_size, args.max_cache_len), dtype=torch.long)
    full_mask[:, :prompt_len] = warmup_inputs.attention_mask

    warmup_args = {
        "input_ids": warmup_inputs.input_ids.to(device),
        "past_key_values": cache,
        "cache_position": torch.arange(0, prompt_len).to(device),
        "use_cache": True,
        "attention_mask": full_mask.to(device),
    }

    with torch.no_grad():
        # Prefill
        output = compiled_model(**warmup_args)
        logits = output.logits.to("cpu")
        next_ids = logits[:, -1].argmax(dim=-1)
        print(
            f"  Warmup prefill done. First tokens: {[tokenizer.decode(t) for t in next_ids[:3]]}"
        )

        # One decode step to compile decode graph
        warmup_args["input_ids"] = next_ids.unsqueeze(-1).to(device)
        host_pos = warmup_args["cache_position"].to("cpu")
        warmup_args["cache_position"] = torch.tensor([host_pos[-1:] + 1]).to(device)
        output = compiled_model(**warmup_args)
        print("  Warmup decode done.")

    print("Ready.\n")

    # Main test loop
    passes = 0
    fails = 0
    start = time.perf_counter()

    for run in range(1, args.num_runs + 1):
        # Pick prompts for this run
        if args.shuffle:
            indices = random.sample(range(len(PROMPTS)), batch_size)
        else:
            indices = list(range(batch_size))
        run_prompts = [PROMPTS[i] for i in indices]
        run_keywords = [KEYWORDS[i] for i in indices]

        # Tokenize
        inputs = tokenizer(
            run_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        prompt_len = inputs.input_ids.shape[1]

        # Reset cache
        for layer in cache.layers:
            layer.keys.zero_()
            layer.values.zero_()

        # Build inputs
        full_mask = torch.ones((batch_size, args.max_cache_len), dtype=torch.long)
        full_mask[:, :prompt_len] = inputs.attention_mask

        input_args = {
            "input_ids": inputs.input_ids.to(device),
            "past_key_values": cache,
            "cache_position": torch.arange(0, prompt_len).to(device),
            "use_cache": True,
            "attention_mask": full_mask.to(device),
        }

        all_tokens: List[List[int]] = [[] for _ in range(batch_size)]

        t0 = time.perf_counter()
        with torch.no_grad():
            for step in range(args.max_new_tokens):
                output = compiled_model(**input_args)
                logits = output.logits.to("cpu")
                next_ids = logits[:, -1].argmax(dim=-1)

                for i in range(batch_size):
                    all_tokens[i].append(next_ids[i].item())

                if torch.all(next_ids == tokenizer.eos_token_id):
                    break

                input_args["input_ids"] = next_ids.unsqueeze(-1).to(device)
                host_pos = input_args["cache_position"].to("cpu")
                input_args["cache_position"] = torch.tensor([host_pos[-1:] + 1]).to(
                    device
                )

        dt = time.perf_counter() - t0
        responses = [tokenizer.decode(t, skip_special_tokens=True) for t in all_tokens]

        ts = datetime.now().strftime("%H:%M:%S")
        bleeds = check_bleed(responses, run_keywords)

        if bleeds:
            fails += 1
            for src, vic, kw in bleeds:
                print(
                    f"  BLEED: '{kw}' (slot {src}/{run_keywords[src]}) found in slot {vic}/{run_keywords[vic]}"
                )
                print(f"    Response: {responses[vic][:150]}...")
            print(f"[{ts}] Run {run}/{args.num_runs} ({dt:.1f}s) FAIL\n")
        else:
            passes += 1
            # Print first 2 responses on first and last run for sanity
            if run <= 2 or run == args.num_runs:
                for i in range(min(2, batch_size)):
                    print(f"  Slot {i} ({run_keywords[i]}): {responses[i][:80]}...")
            print(
                f"[{ts}] Run {run}/{args.num_runs} ({dt:.1f}s) PASS  [{passes}P/{fails}F]"
            )

    elapsed = time.perf_counter() - start
    print()
    print("============================================")
    print(f"Results: {passes} PASS / {fails} FAIL out of {args.num_runs}")
    if fails > 0:
        print(f"Failure rate: {fails / args.num_runs * 100:.1f}%")
        print("Bug is in compiled graph batching — not specific to paged attention")
    else:
        print("No bleed — bug is specific to paged KV cache, not general batching")
    print(f"Total time: {elapsed:.0f}s ({elapsed / args.num_runs:.1f}s/run)")
    print("============================================")

    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
