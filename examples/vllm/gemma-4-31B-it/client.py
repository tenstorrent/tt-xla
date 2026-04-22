# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming chat client for a TT-backed vLLM server running Gemma-4 31B-it.

Usage:
    1. Start the vLLM server:
       $ bash examples/vllm/gemma-4-31B-it/service.sh

    2. Run this client (in a separate shell):
       $ python examples/vllm/gemma-4-31B-it/client.py
"""

import json

import requests


def main():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    # Gemma-4 chat template does not accept the "system" role, so the
    # conversation starts empty and only user/assistant turns are appended.
    data = {
        "model": "google/gemma-4-31B-it",
        "messages": [],
        "stream": True,
        "max_tokens": 256,
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
                            print("\nResponse completed.")
                            break

                        token = json.loads(chunk)["choices"][0]["delta"].get(
                            "content", ""
                        )
                        if token:
                            print(token, end="", flush=True)
                            full_response += token
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
