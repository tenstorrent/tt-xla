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

    # Batch=16, non-greedy device
    python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --temperature 0.8 --batch-size 16 --gpu-memory-utilization 0.037
"""

import argparse
import time

import vllm

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MAX_MODEL_LEN = 2048
MAX_MODEL_LEN_FAST = 128  # Smaller context → faster compilation
GPU_MEMORY_UTILIZATION = 0.05

BENCHMARK_PROMPT = "Explain the theory of relativity in simple terms."

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


def create_engine(
    cpu_sampling=False,
    fast=False,
    skip_precompile=False,
    max_model_len_override=None,
    batch_size=1,
    gpu_memory_utilization=None,
):
    max_len = max_model_len_override or (MAX_MODEL_LEN_FAST if fast else MAX_MODEL_LEN)
    gpu_mem = (
        gpu_memory_utilization
        if gpu_memory_utilization is not None
        else GPU_MEMORY_UTILIZATION
    )
    additional_config = {
        "enable_const_eval": False,
        "min_context_len": 32,
    }
    if cpu_sampling:
        additional_config["cpu_sampling"] = True

    sampling_label = "CPU" if cpu_sampling else "device"
    print(
        f"Loading {MODEL} (sampling: {sampling_label}, max_model_len={max_len}, batch_size={batch_size}) ..."
    )
    start = time.perf_counter()
    llm = vllm.LLM(
        model=MODEL,
        max_model_len=max_len,
        max_num_batched_tokens=max_len * batch_size,
        max_num_seqs=batch_size,
        gpu_memory_utilization=gpu_mem,
        disable_log_stats=True,
        enforce_eager=skip_precompile,
        additional_config=additional_config,
    )
    elapsed = time.perf_counter() - start
    print(f"Engine ready in {elapsed:.1f}s.\n")
    return llm


def warmup(llm, temperature):
    print("Warming up ...")
    try:
        import tracy

        tracy.signpost("warmup_start")
    except (ImportError, AttributeError):
        pass
    params = vllm.SamplingParams(max_tokens=16, temperature=temperature)
    llm.generate(["Hello"], params)
    try:
        import tracy

        tracy.signpost("warmup_complete")
    except (ImportError, AttributeError):
        pass
    print("Warmup complete.\n")


def run_benchmark(llm, args):
    tokenizer = llm.get_tokenizer()
    sampling_kwargs = {"max_tokens": args.max_tokens, "temperature": args.temperature}
    if args.temperature > 0:
        sampling_kwargs["top_p"] = args.top_p

    params = vllm.SamplingParams(**sampling_kwargs)
    sampling_label = "CPU" if args.cpu_sampling else "device"
    temp_label = f"temperature={args.temperature}"
    if args.temperature > 0:
        temp_label += f", top_p={args.top_p}"

    # Use same prompt repeated for all batch slots (matches vllm_benchmark.py pattern)
    prompt_text = args.prompt or BENCHMARK_PROMPT
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompts = [prompt] * args.batch_size

    print(f"Sampling: {sampling_label}, {temp_label}, batch_size={args.batch_size}")
    print(f"Generating {args.max_tokens} tokens x {args.batch_size} requests")
    print("-" * 70)

    try:
        import tracy as _tracy

        _signpost = _tracy.signpost
    except (ImportError, AttributeError):
        _signpost = lambda x: None

    _signpost("generate_0_start")
    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start
    _signpost("generate_0_end")

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    overall_tok_s = total_tokens / elapsed if elapsed > 0 else 0

    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Model:              {MODEL}")
    print(f"  Sampling:           {sampling_label}")
    print(f"  Temperature:        {args.temperature}")
    if args.temperature > 0:
        print(f"  Top-p:              {args.top_p}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Max tokens:         {args.max_tokens}")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Total time:         {elapsed:.2f}s")
    print(f"  Overall tok/s:      {overall_tok_s:.2f}")
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
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=f"KV cache memory fraction (default: {GPU_MEMORY_UTILIZATION}; use ~0.037 for batch>=16)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="(legacy) Number of sequential benchmark prompts",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (overrides built-in prompt)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"Use smaller max_model_len ({MAX_MODEL_LEN_FAST}) for faster compilation",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override max_model_len (overrides --fast)",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the warmup generation step",
    )
    parser.add_argument(
        "--skip-precompile",
        action="store_true",
        help="Skip precompilation (enforce_eager=True). Use for cleaner Tracy traces.",
    )
    args = parser.parse_args()

    llm = create_engine(
        cpu_sampling=args.cpu_sampling,
        fast=args.fast,
        skip_precompile=args.skip_precompile,
        max_model_len_override=args.max_model_len,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if not args.skip_warmup:
        warmup(llm, args.temperature)

    if args.benchmark:
        run_benchmark(llm, args)
    else:
        run_interactive(llm, args.temperature, args.max_tokens)


if __name__ == "__main__":
    main()
