# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Chat client for the Qwen3-0.6B codegen-load server (see service.sh for usage).
#
# vLLM keeps no conversation state, so we maintain the history client-side: each
# turn appends the user message, sends the WHOLE list, and appends the streamed
# reply so the next turn has context.

import json

import requests


def main():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [],
        "stream": True,
        "temperature": 0,
        # enable_thinking=False keeps Qwen3 answering directly (no <think> trace).
        "chat_template_kwargs": {"enable_thinking": False},
    }

    while True:
        input_text = input("Enter a message (or 'q' to quit): ")
        if input_text == "q":
            break
        if not input_text.strip():
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
                    if not chunk:
                        continue
                    if "[DONE]" in chunk:
                        print("\nResponse completed.")
                        break

                    token = json.loads(chunk)["choices"][0]["delta"].get("content")
                    if token:
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
