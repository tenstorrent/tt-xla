# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

import requests


def main():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ""},
        ],
        "stream": True,
    }

    while True:
        input_text = input("Enter a message (or 'q' to quit): ")
        if input_text == "q":
            break

        data["messages"][1]["content"] = input_text

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
                            print("\nResponse completed.")
                            break

                        token = json.loads(chunk)["choices"][0]["delta"]["content"]
                        print(token, end="", flush=True)
        except Exception as e:
            if isinstance(e, requests.exceptions.ConnectionError):
                print(
                    "Server returned a connection error. This usually occurs when a request is made before the service is ready. Please wait for the service to be ready and try again."
                )
            else:
                raise e


if __name__ == "__main__":
    main()
