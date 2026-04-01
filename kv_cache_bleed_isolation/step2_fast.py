#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Step 2 FAST variant — tuned to hit the KV cache bleed bug more frequently.
#
# Changes vs step2_vllm_direct.py:
#   - Smaller max_model_len (512) to force KV cache page reuse
#   - Fewer turns (5) since bleed appears in turns 3-8
#   - More runs (50) to catch intermittent failures
#   - Batch size 8 for more cache slot boundaries
#
# Usage:
#   python3 step2_fast.py
#   python3 step2_fast.py --batch-size 4 --num-runs 100 --max-model-len 256

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
        ],
    },
    {
        "name": "dinosaurs",
        "keyword": "dinosaur",
        "system": "You are an expert on dinosaurs. Only discuss dinosaurs. Always mention the word dinosaur in every response.",
        "prompts": [
            "Tell me about T-Rex dinosaurs",
            "How did dinosaurs go extinct?",
            "What did dinosaurs eat?",
            "Describe the biggest dinosaur ever",
            "How fast could dinosaurs run?",
        ],
    },
    {
        "name": "chocolate",
        "keyword": "chocolate",
        "system": "You are an expert on chocolate. Only discuss chocolate. Always mention the word chocolate in every response.",
        "prompts": [
            "How is chocolate made?",
            "What is dark chocolate?",
            "Tell me about chocolate history",
            "Describe Belgian chocolate",
            "What makes chocolate taste good?",
        ],
    },
    {
        "name": "castles",
        "keyword": "castle",
        "system": "You are an expert on castles. Only discuss castles. Always mention the word castle in every response.",
        "prompts": [
            "Tell me about medieval castles",
            "How were castles defended?",
            "What is the biggest castle?",
            "Describe castle architecture",
            "Who lived in castles?",
        ],
    },
    {
        "name": "galaxies",
        "keyword": "galaxy",
        "system": "You are an expert on galaxies. Only discuss galaxies. Always mention the word galaxy in every response.",
        "prompts": [
            "Tell me about the Milky Way galaxy",
            "How do galaxies form?",
            "What is a spiral galaxy?",
            "Describe galaxy collisions",
            "How many galaxies exist?",
        ],
    },
    {
        "name": "dolphins",
        "keyword": "dolphin",
        "system": "You are an expert on dolphins. Only discuss dolphins. Always mention the word dolphin in every response.",
        "prompts": [
            "How do dolphins communicate underwater?",
            "Tell me about dolphin intelligence",
            "What do dolphins eat?",
            "Describe dolphin social behavior",
            "How fast can dolphins swim?",
        ],
    },
    {
        "name": "earthquakes",
        "keyword": "earthquake",
        "system": "You are an expert on earthquakes. Only discuss earthquakes. Always mention the word earthquake in every response.",
        "prompts": [
            "How are earthquakes measured?",
            "Tell me about the San Andreas fault earthquake risk",
            "What causes earthquakes?",
            "Describe earthquake early warning systems",
            "What was the biggest earthquake ever recorded?",
        ],
    },
    {
        "name": "pyramids",
        "keyword": "pyramid",
        "system": "You are an expert on pyramids. Only discuss pyramids. Always mention the word pyramid in every response.",
        "prompts": [
            "How were the pyramids built in ancient Egypt?",
            "Tell me about the Great Pyramid of Giza",
            "What is inside the pyramids?",
            "Describe pyramid construction techniques",
            "Who built the pyramids?",
        ],
    },
    {
        "name": "lightning",
        "keyword": "lightning",
        "system": "You are an expert on lightning. Only discuss lightning. Always mention the word lightning in every response.",
        "prompts": [
            "How does lightning form in thunderstorms?",
            "Tell me about ball lightning",
            "What temperature is lightning?",
            "Describe lightning safety tips",
            "How many times does lightning strike the earth daily?",
        ],
    },
    {
        "name": "glaciers",
        "keyword": "glacier",
        "system": "You are an expert on glaciers. Only discuss glaciers. Always mention the word glacier in every response.",
        "prompts": [
            "How do glaciers shape mountain landscapes?",
            "Tell me about glacier retreat and climate change",
            "What is inside a glacier?",
            "Describe the largest glaciers on Earth",
            "How do glaciers move?",
        ],
    },
    {
        "name": "coral",
        "keyword": "coral",
        "system": "You are an expert on coral reefs. Only discuss coral. Always mention the word coral in every response.",
        "prompts": [
            "How do coral reefs form in tropical oceans?",
            "Tell me about coral bleaching",
            "What animals live in coral reefs?",
            "Describe the Great Barrier Reef coral",
            "How can we protect coral reefs?",
        ],
    },
    {
        "name": "telescopes",
        "keyword": "telescope",
        "system": "You are an expert on telescopes. Only discuss telescopes. Always mention the word telescope in every response.",
        "prompts": [
            "How do telescopes work to observe distant stars?",
            "Tell me about the James Webb Space Telescope",
            "What is a radio telescope?",
            "Describe the history of telescopes",
            "How powerful are modern telescopes?",
        ],
    },
    {
        "name": "cricket",
        "keyword": "cricket",
        "system": "You are an expert on cricket. Only discuss cricket. Always mention the word cricket in every response.",
        "prompts": [
            "Explain the basic rules of cricket",
            "Tell me about Test cricket matches",
            "What equipment is used in cricket?",
            "Describe famous cricket players",
            "How is cricket scored?",
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


def run_batch_test(llm, tokenizer, num_turns, batch_size):
    topics = TOPICS[:batch_size]
    histories = [[] for _ in range(batch_size)]
    all_responses = [[] for _ in range(batch_size)]
    sampling_params = vllm.SamplingParams(max_tokens=32, temperature=0.0)

    for turn in range(num_turns):
        prompts = []
        messages_list = []
        for i, topic in enumerate(topics):
            msgs, prompt_str = build_prompt(tokenizer, topic, turn, histories[i])
            prompts.append(prompt_str)
            messages_list.append(msgs)

        outputs = llm.generate(prompts, sampling_params)

        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            all_responses[i].append(text)
            histories[i].append(
                {"role": "user", "content": messages_list[i][-1]["content"]}
            )
            histories[i].append({"role": "assistant", "content": text})

    return all_responses


def check_bleed(all_responses, batch_size):
    topics = TOPICS[:batch_size]
    bleed_found = False

    for i, topic in enumerate(topics):
        foreign = FOREIGN_KEYWORDS[topic["name"]]
        for fk in foreign:
            for turn_idx, response in enumerate(all_responses[i]):
                if fk.lower() in response.lower():
                    print(
                        f"  BLEED: '{fk}' found in {topic['name']} turn {turn_idx + 1}"
                    )
                    print(f"    Response: {response[:200]}...")
                    bleed_found = True

    return bleed_found


def main():
    parser = argparse.ArgumentParser(
        description="Step 2 FAST: Tuned to hit KV cache bleed more frequently"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument("--num-turns", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"Model:         {MODEL}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Turns:         {args.num_turns}")
    print(f"Runs:          {args.num_runs}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Max tokens:    {args.max_tokens}")
    print()

    print("Loading model...")
    llm = vllm.LLM(
        model=MODEL,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len * args.batch_size,
        max_num_seqs=args.batch_size,
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
        all_responses = run_batch_test(llm, tokenizer, args.num_turns, args.batch_size)
        bleed = check_bleed(all_responses, args.batch_size)
        if bleed:
            fails += 1
            print(f"  FAIL\n")
        else:
            passes += 1
            print(f"  PASS\n")

    print("============================================")
    print(f"Results: {passes} PASS / {fails} FAIL out of {args.num_runs}")
    if fails > 0:
        print(f"Failure rate: {fails/args.num_runs*100:.0f}%")
    print("============================================")


if __name__ == "__main__":
    main()
