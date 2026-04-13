# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive client for OPT-125M vLLM server with greedy device-side sampling."""

import argparse
import json
import time

import requests


def main():
    parser = argparse.ArgumentParser(description="OPT-125M completions client")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens to generate (default: 64)",
    )
    args = parser.parse_args()

    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}

    print(f"OPT-125M client (greedy, device sampling, max_tokens={args.max_tokens})")
    print("Enter a prompt to complete, or 'q' to quit.\n")

    while True:
        prompt = input("Prompt: ")
        if prompt.strip().lower() == "q":
            break

        data = {
            "model": "facebook/opt-125m",
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
        }

        try:
            token_count = 0
            t_start = time.perf_counter()
            t_first = None
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
                            t_end = time.perf_counter()
                            ttft = (t_first - t_start) * 1000 if t_first else 0
                            elapsed = t_end - t_start
                            tps = token_count / elapsed if elapsed > 0 else 0
                            print(
                                f"\n[{token_count} tokens, {tps:.1f} tok/s, "
                                f"TTFT {ttft:.0f}ms]\n"
                            )
                            break
                        token = json.loads(chunk)["choices"][0]["text"]
                        if t_first is None:
                            t_first = time.perf_counter()
                        token_count += 1
                        print(token, end="", flush=True)
        except requests.exceptions.ConnectionError:
            print(
                "Connection error. Is the server running? "
                "Start it with: ./service.sh"
            )
        except Exception as e:
            raise e


if __name__ == "__main__":
    main()
