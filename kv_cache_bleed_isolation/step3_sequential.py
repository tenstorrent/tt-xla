#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Step 3: Sequential conversations (batch=1) — temporal contamination test
#
# Runs 4 conversations one at a time through the same vllm.LLM engine.
# Tests whether KV cache residue from one conversation bleeds into the next.
#
# If this fails: cache cleanup bug between requests.
# If this passes but Step 2 fails: concurrent batch isolation bug.
#
# Usage:
#   python3 step3_sequential.py
#   python3 step3_sequential.py --num-runs 50 --num-turns 5

import argparse
import os

import vllm

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.2-1B-Instruct")

TOPICS = [
    {
        "name": "penguins",
        "keyword": "penguin",
        "system": "You are an expert on penguins. Only discuss penguins. Always mention the word penguin in every response.",
        "prompts": [
            "Tell me about emperor penguins",
            "How do penguins survive the cold?",
            "What do penguins eat?",
            "Describe penguin mating rituals",
            "How fast can penguins swim?",
            "Tell me about baby penguins",
            "Where do penguins live?",
            "What are the biggest threats to penguins?",
            "How many species of penguins exist?",
            "Tell me a fun fact about penguins",
        ],
    },
    {
        "name": "volcanoes",
        "keyword": "volcano",
        "system": "You are an expert on volcanoes. Only discuss volcanoes. Always mention the word volcano in every response.",
        "prompts": [
            "Tell me about Mount Vesuvius volcano",
            "How do volcanoes erupt?",
            "What is volcanic lava made of?",
            "Describe the Ring of Fire volcanoes",
            "What was the biggest volcanic eruption?",
            "How do scientists predict volcanic eruptions?",
            "Tell me about underwater volcanoes",
            "What gases do volcanoes emit?",
            "How do volcanoes form islands?",
            "Tell me about dormant volcanoes",
        ],
    },
    {
        "name": "origami",
        "keyword": "origami",
        "system": "You are an expert on origami paper folding. Only discuss origami. Always mention the word origami in every response.",
        "prompts": [
            "How do I make an origami crane?",
            "What paper is best for origami?",
            "Tell me about the history of origami",
            "What is modular origami?",
            "Describe complex origami techniques",
            "Who are famous origami artists?",
            "How is origami used in engineering?",
            "Tell me about origami mathematics",
            "What is wet-folding origami?",
            "Describe origami paper sizes",
        ],
    },
    {
        "name": "submarines",
        "keyword": "submarine",
        "system": "You are an expert on submarines. Only discuss submarines. Always mention the word submarine in every response.",
        "prompts": [
            "How do submarines dive underwater?",
            "Tell me about nuclear submarines",
            "What is life like on a submarine?",
            "How deep can submarines go?",
            "Describe submarine sonar systems",
            "Tell me about famous submarine battles",
            "How do submarines produce oxygen?",
            "What was the first submarine ever built?",
            "How do submarine crews communicate?",
            "Tell me about research submarines",
        ],
    },
]

FOREIGN_KEYWORDS = {}
for t in TOPICS:
    FOREIGN_KEYWORDS[t["name"]] = [
        other["keyword"] for other in TOPICS if other["name"] != t["name"]
    ]


def build_prompt(tokenizer, topic, turn, conversation_history):
    messages = [{"role": "system", "content": topic["system"]}]
    messages.extend(conversation_history)
    prompt_text = topic["prompts"][turn % len(topic["prompts"])]
    messages.append({"role": "user", "content": prompt_text})
    return messages, tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_sequential_test(llm, tokenizer, num_turns):
    """Run 4 conversations sequentially (batch=1) through the same engine."""
    all_responses = [[] for _ in range(len(TOPICS))]
    sampling_params = vllm.SamplingParams(max_tokens=32, temperature=0.0)

    for topic_idx, topic in enumerate(TOPICS):
        history = []
        for turn in range(num_turns):
            msgs, prompt_str = build_prompt(tokenizer, topic, turn, history)

            # Single request, batch=1
            outputs = llm.generate([prompt_str], sampling_params)
            text = outputs[0].outputs[0].text

            all_responses[topic_idx].append(text)
            history.append({"role": "user", "content": msgs[-1]["content"]})
            history.append({"role": "assistant", "content": text})

    return all_responses


def check_bleed(all_responses):
    """Check for cross-topic contamination.

    Only check for keywords from PREVIOUS conversations bleeding into
    later ones (temporal contamination).
    """
    bleed_found = False
    previous_keywords = []

    for i, topic in enumerate(TOPICS):
        for fk in previous_keywords:
            for turn_idx, response in enumerate(all_responses[i]):
                if fk.lower() in response.lower():
                    print(
                        f"  BLEED: '{fk}' found in {topic['name']} turn {turn_idx + 1}"
                    )
                    print(f"    Response: {response[:200]}...")
                    bleed_found = True
        previous_keywords.append(topic["keyword"])

    return bleed_found


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Sequential conversations (batch=1) temporal bleed test"
    )
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--num-turns", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=512)
    args = parser.parse_args()

    print(f"Model:         {MODEL}")
    print(f"Batch size:    1 (sequential)")
    print(f"Turns:         {args.num_turns} per conversation")
    print(f"Conversations: {len(TOPICS)} (run sequentially)")
    print(f"Runs:          {args.num_runs}")
    print(f"Max model len: {args.max_model_len}")
    print()

    print("Loading model...")
    llm = vllm.LLM(
        model=MODEL,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=0.05,
        disable_log_stats=True,
        additional_config={
            "enable_const_eval": True,
            "min_context_len": 32,
            "experimental_weight_dtype": "bfp8",
            "cpu_sampling": True,
        },
    )
    tokenizer = llm.get_tokenizer()

    print("Warmup...")
    llm.generate(["Hello"], vllm.SamplingParams(max_tokens=8, temperature=0.0))
    print("Ready.\n")

    passes = 0
    fails = 0

    for run in range(1, args.num_runs + 1):
        print(f"=== Run {run}/{args.num_runs} ===")
        all_responses = run_sequential_test(llm, tokenizer, args.num_turns)
        bleed = check_bleed(all_responses)
        if bleed:
            fails += 1
            print(f"  FAIL\n")
        else:
            passes += 1
            print(f"  PASS\n")

    print("============================================")
    print(f"Step 3 Results: {passes} PASS / {fails} FAIL out of {args.num_runs}")
    if fails > 0:
        print(f"Failure rate: {fails / args.num_runs * 100:.0f}%")
        print("Temporal contamination confirmed — cache cleanup bug between requests")
    else:
        print(
            "No temporal contamination — bug is specific to concurrent batch isolation"
        )
    print("============================================")


if __name__ == "__main__":
    main()
