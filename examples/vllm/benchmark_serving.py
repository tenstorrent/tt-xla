# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark token throughput for a vLLM server running on TT hardware.

Auto-detects the model being served via the /v1/models endpoint.

Usage:
    1. Start the vLLM server:
       $ bash examples/vllm/Llama-3.2-3B-Instruct/service.sh

    2. Run this benchmark:
       $ python examples/vllm/benchmark_serving.py --num-prompts 5 --max-tokens 128
"""

import argparse
import json
import time
from dataclasses import dataclass

import requests


@dataclass
class BenchmarkResult:
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token: float
    total_time: float
    tokens_per_sec: float


PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main differences between Python and Rust?",
    "Describe the process of photosynthesis step by step.",
    "What is the meaning of life according to different philosophies?",
    "Explain how a neural network learns from data.",
    "Write a poem about the ocean at night.",
    "What were the key events of the French Revolution?",
]


def benchmark_single_request(
    url: str, model: str, prompt: str, max_tokens: int
) -> BenchmarkResult:
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start_time = time.perf_counter()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0

    with requests.post(url, json=data, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event_data = line[len("data: ") :]
            if event_data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(event_data)
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    continue
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    completion_tokens += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

    total_time = time.perf_counter() - start_time
    tokens_per_sec = completion_tokens / total_time if total_time > 0 else 0

    return BenchmarkResult(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        time_to_first_token=ttft or total_time,
        total_time=total_time,
        tokens_per_sec=tokens_per_sec,
    )


def get_served_model(base_url: str) -> str:
    """Auto-detect the model being served via /v1/models."""
    models_url = base_url.rstrip("/") + "/v1/models"
    response = requests.get(models_url)
    response.raise_for_status()
    models = response.json()["data"]
    if not models:
        raise RuntimeError("No models being served")
    return models[0]["id"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM serving throughput")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (auto-detected from server if not provided)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="vLLM server base URL",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts to benchmark",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens per completion",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup requests (excluded from metrics)",
    )
    args = parser.parse_args()

    chat_url = args.base_url.rstrip("/") + "/v1/chat/completions"
    model = args.model or get_served_model(args.base_url)
    print(f"Using model: {model}")

    prompts = PROMPTS[: args.num_prompts]

    # Warmup
    print(f"Running {args.warmup} warmup request(s)...")
    for i in range(args.warmup):
        try:
            result = benchmark_single_request(
                chat_url, model, prompts[i % len(prompts)], args.max_tokens
            )
            print(
                f"  Warmup {i + 1}: {result.completion_tokens} tokens, {result.tokens_per_sec:.2f} tok/s"
            )
        except requests.exceptions.ConnectionError:
            print("  Server not ready, retrying...")
            time.sleep(5)

    # Benchmark
    print(f"\nBenchmarking {len(prompts)} prompts, max_tokens={args.max_tokens}...")
    print("-" * 80)

    results = []
    for i, prompt in enumerate(prompts):
        result = benchmark_single_request(chat_url, model, prompt, args.max_tokens)
        results.append(result)
        print(
            f"  [{i + 1}/{len(prompts)}] "
            f"TTFT: {result.time_to_first_token:.3f}s | "
            f"Tokens: {result.completion_tokens} | "
            f"Time: {result.total_time:.3f}s | "
            f"Throughput: {result.tokens_per_sec:.2f} tok/s"
        )

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    total_tokens = sum(r.completion_tokens for r in results)
    total_time = sum(r.total_time for r in results)
    avg_ttft = sum(r.time_to_first_token for r in results) / len(results)
    avg_tps = sum(r.tokens_per_sec for r in results) / len(results)
    total_prompt_tokens = sum(r.prompt_tokens for r in results)

    print(f"  Model:                 {model}")
    print(f"  Requests:              {len(results)}")
    print(f"  Total prompt tokens:   {total_prompt_tokens}")
    print(f"  Total output tokens:   {total_tokens}")
    print(f"  Total time:            {total_time:.3f}s")
    print(f"  Avg TTFT:              {avg_ttft:.3f}s")
    print(f"  Avg throughput:        {avg_tps:.2f} tok/s")
    print(f"  Overall throughput:    {total_tokens / total_time:.2f} tok/s")
    print("=" * 80)


if __name__ == "__main__":
    main()
