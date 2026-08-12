# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Streaming chat client for the Devstral-2-123B galaxy DP+TP server
# (see service.sh for how to start it).
#
# vLLM keeps no conversation state, so we maintain the history client-side:
# each turn appends the user message, sends the WHOLE list, and appends the
# streamed reply so the next turn has context.

import json

import requests


def main():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "mistralai/Devstral-2-123B-Instruct-2512",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
        ],
        "stream": True,
        "temperature": 0,
    }

    while True:
        input_text = input("Enter a message (or 'q' to quit): ")
        if input_text == "q":
            break

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

                        token = json.loads(chunk)["choices"][0]["delta"]["content"]
                        print(token, end="", flush=True)
                        full_response += token
                data["messages"].append(
                    {"role": "assistant", "content": full_response}
                )
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
