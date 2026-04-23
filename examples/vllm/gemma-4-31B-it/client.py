# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming chat client for a TT-backed vLLM server running Gemma-4 31B-it.

Usage:
    1. Start the vLLM server:
       $ bash examples/vllm/gemma-4-31B-it/service.sh

    2. Run this client (in a separate shell):
       $ python examples/vllm/gemma-4-31B-it/client.py [--max-tokens N]
"""

import argparse
import json
import time

import requests


def main():
    parser = argparse.ArgumentParser(
        description="Streaming chat client for Gemma-4 31B-it vLLM server"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    args = parser.parse_args()

    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    # Gemma-4 chat template does not accept the "system" role, so the
    # conversation starts empty and only user/assistant turns are appended.
    data = {
        "model": "google/gemma-4-31B-it",
        "messages": [],
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }

    while True:
        input_text = input("Enter a message ('c' to clear history, 'q' to quit): ")
        if input_text == "q":
            break
        if input_text == "c":
            data["messages"].clear()
            print("Conversation history cleared.")
            continue

        data["messages"].append({"role": "user", "content": input_text})

        try:
            full_response = ""
            token_count = 0
            ttft = None
            t_start = time.perf_counter()
            with requests.post(
                url, headers=headers, json=data, stream=True
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(
                    chunk_size=8192, decode_unicode=True
                ):
                    chunk = chunk.removeprefix("data: ")
                    if chunk:
                        if "[DONE]" in chunk:
                            break

                        token = json.loads(chunk)["choices"][0]["delta"].get(
                            "content", ""
                        )
                        if token:
                            if ttft is None:
                                ttft = time.perf_counter() - t_start
                            print(token, end="", flush=True)
                            full_response += token
                            token_count += 1

            elapsed = time.perf_counter() - t_start
            toks_per_sec = token_count / elapsed if elapsed > 0 else 0
            print(
                f"\n[TTFT: {ttft * 1000:.0f} ms | {token_count} tokens | {toks_per_sec:.1f} tok/s]"
            )
            data["messages"].append({"role": "assistant", "content": full_response})
        except Exception as e:
            if isinstance(e, requests.exceptions.ConnectionError):
                print(
                    "Server returned a connection error. This usually occurs when a request is made before the service is ready. Please wait for the service to be ready and try again."
                )
                data["messages"].pop()
            else:
                raise e


if __name__ == "__main__":
    main()
