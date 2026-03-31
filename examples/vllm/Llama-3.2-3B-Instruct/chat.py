# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Interactive single-process chat with Llama-3.2-3B-Instruct using vllm.LLM.

No server required — runs the engine directly in this process.

Usage:
    # Interactive chat (greedy)
    python examples/vllm/Llama-3.2-3B-Instruct/chat.py

    # Benchmark: device sampling, non-greedy (the slow case)
    python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --temperature 0.8

    # Benchmark: CPU sampling, non-greedy
    python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --temperature 0.8 --cpu-sampling

    # Benchmark: device sampling, greedy (baseline)
    python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --temperature 0.0
"""

import argparse
import time

import vllm

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MAX_MODEL_LEN = 2048
MAX_MODEL_LEN_FAST = 128  # Smaller context → faster compilation
GPU_MEMORY_UTILIZATION = 0.05

BENCHMARK_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main differences between Python and Rust?",
    "Describe the process of photosynthesis step by step.",
    "What is the meaning of life according to different philosophies?",
    "Explain how a neural network learns from data.",
    "Write a poem about the ocean at night.",
    "What were the key events of the French Revolution?",
]


def create_engine(cpu_sampling=False, fast=False):
    additional_config = {
        "enable_const_eval": False,
        "min_context_len": 32,
    }
    if cpu_sampling:
        additional_config["cpu_sampling"] = True

    max_len = MAX_MODEL_LEN_FAST if fast else MAX_MODEL_LEN
    sampling_label = "CPU" if cpu_sampling else "device"
    print(f"Loading {MODEL} (sampling: {sampling_label}, max_model_len={max_len}) ...")
    llm = vllm.LLM(
        model=MODEL,
        max_model_len=max_len,
        max_num_batched_tokens=max_len,
        max_num_seqs=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
        additional_config=additional_config,
    )
    print("Engine ready.\n")
    return llm


def warmup(llm, temperature):
    print("Warming up ...")
    params = vllm.SamplingParams(max_tokens=16, temperature=temperature)
    llm.generate(["Hello"], params)
    print("Warmup complete.\n")


def run_benchmark(llm, args):
    tokenizer = llm.get_tokenizer()
    prompts = BENCHMARK_PROMPTS[: args.num_prompts]

    sampling_kwargs = {"max_tokens": args.max_tokens, "temperature": args.temperature}
    if args.temperature > 0:
        sampling_kwargs["top_p"] = args.top_p

    params = vllm.SamplingParams(**sampling_kwargs)
    sampling_label = "CPU" if args.cpu_sampling else "device"
    temp_label = f"temperature={args.temperature}"
    if args.temperature > 0:
        temp_label += f", top_p={args.top_p}"

    print(f"Sampling: {sampling_label}, {temp_label}")
    print(f"Generating {args.max_tokens} tokens x {len(prompts)} prompts")
    print("-" * 70)

    results = []
    for i, prompt_text in enumerate(prompts):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start = time.perf_counter()
        outputs = llm.generate([prompt], params)
        elapsed = time.perf_counter() - start

        output = outputs[0]
        num_tokens = len(output.outputs[0].token_ids)
        prompt_tokens = len(output.prompt_token_ids)
        tok_s = num_tokens / elapsed if elapsed > 0 else 0

        results.append((num_tokens, prompt_tokens, elapsed, tok_s))
        print(
            f"  [{i + 1}/{len(prompts)}] {num_tokens} tokens, "
            f"{prompt_tokens} prompt tokens, {tok_s:.2f} tok/s, {elapsed:.2f}s"
        )

    total_tokens = sum(r[0] for r in results)
    total_time = sum(r[2] for r in results)
    avg_tok_s = sum(r[3] for r in results) / len(results)

    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Model:              {MODEL}")
    print(f"  Sampling:           {sampling_label}")
    print(f"  Temperature:        {args.temperature}")
    if args.temperature > 0:
        print(f"  Top-p:              {args.top_p}")
    print(f"  Prompts:            {len(results)}")
    print(f"  Max tokens:         {args.max_tokens}")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Avg tok/s:          {avg_tok_s:.2f}")
    print(f"  Overall tok/s:      {total_tokens / total_time:.2f}")
    print("=" * 70)


def run_interactive(llm, temperature, max_tokens):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    tokenizer = llm.get_tokenizer()

    while True:
        user_input = input("Enter a message (or 'q' to quit): ")
        if user_input.strip().lower() == "q":
            break

        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        params = vllm.SamplingParams(max_tokens=max_tokens, temperature=temperature)

        start = time.perf_counter()
        outputs = llm.generate([prompt], params)
        elapsed = time.perf_counter() - start

        output = outputs[0]
        text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        prompt_tokens = len(output.prompt_token_ids)
        tok_s = num_tokens / elapsed if elapsed > 0 else 0

        print(text)
        print(
            f"[{num_tokens} tokens, {prompt_tokens} prompt tokens, "
            f"{tok_s:.1f} tok/s, {elapsed:.2f}s]"
        )

        messages.append({"role": "assistant", "content": text})


def main():
    parser = argparse.ArgumentParser(description="Chat or benchmark Llama-3.2-3B")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run automated benchmark"
    )
    parser.add_argument(
        "--cpu-sampling",
        action="store_true",
        help="Use CPU sampling instead of device-compiled sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p for non-greedy sampling"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Max tokens per generation"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of benchmark prompts (max 8)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"Use smaller max_model_len ({MAX_MODEL_LEN_FAST}) for faster compilation",
    )
    args = parser.parse_args()

    llm = create_engine(cpu_sampling=args.cpu_sampling, fast=args.fast)
    warmup(llm, args.temperature)

    if args.benchmark:
        run_benchmark(llm, args)
    else:
        run_interactive(llm, args.temperature, args.max_tokens)


if __name__ == "__main__":
    main()
