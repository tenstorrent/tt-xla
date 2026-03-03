# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Example client for the OpenAI /v1/responses API with a TT-backed vLLM server.

Usage:
    1. Start the vLLM server:
       $ bash examples/vllm/TinyLlama-1.1B-Chat-v1.0/service.sh

    2. Run this client:
       $ python examples/vllm/TinyLlama-1.1B-Chat-v1.0/responses_client.py
"""

import json

import requests


def main():
    url = "http://localhost:8000/v1/responses"
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    conversation = []

    while True:
        input_text = input("Enter a message ('c' to clear history, 'q' to quit): ")
        if input_text == "q":
            break
        if input_text == "c":
            conversation.clear()
            print("Conversation history cleared.")
            continue

        conversation.append({"role": "user", "content": input_text})

        data = {
            "model": model,
            "input": conversation,
            "stream": True,
        }

        try:
            full_response = ""
            with requests.post(url, json=data, stream=True) as response:
                if response.status_code != 200:
                    error = response.json().get("error", {})
                    print(f"Server error: {error.get('message', response.text)}")
                    conversation.pop()
                    continue
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    event_data = line[len("data: ") :]
                    if event_data.strip() == "[DONE]":
                        break
                    try:
                        event = json.loads(event_data)
                        if event.get("type") == "response.output_text.delta":
                            delta = event.get("delta", "")
                            print(delta, end="", flush=True)
                            full_response += delta
                    except json.JSONDecodeError:
                        pass

            print()
            conversation.append({"role": "assistant", "content": full_response})
        except requests.exceptions.ConnectionError:
            print(
                "Server returned a connection error. This usually occurs when a "
                "request is made before the service is ready. Please wait for the "
                "service to be ready and try again."
            )
            conversation.pop()


if __name__ == "__main__":
    main()
