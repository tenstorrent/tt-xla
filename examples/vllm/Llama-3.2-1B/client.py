# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive client for Llama-3.2-1B vLLM server with greedy device-side sampling."""

import argparse
import json

import requests


def main():
    parser = argparse.ArgumentParser(description="Llama-3.2-1B completions client")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens to generate (default: 64)",
    )
    args = parser.parse_args()

    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}

    print(
        f"Llama-3.2-1B client (greedy, device sampling, max_tokens={args.max_tokens})"
    )
    print("Enter a prompt to complete, or 'q' to quit.\n")

    while True:
        prompt = input("Prompt: ")
        if prompt.strip().lower() == "q":
            break

        data = {
            "model": "meta-llama/Llama-3.2-1B",
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
        }

        try:
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
                            print("\n")
                            break
                        token = json.loads(chunk)["choices"][0]["text"]
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
