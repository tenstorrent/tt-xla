# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import time

import requests


def main():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
        ],
        "stream": True,
        "max_tokens": 32,
    }

    while True:
        input_text = input("Enter a message (or 'q' to quit): ")
        if input_text == "q":
            break

        data["messages"].append({"role": "user", "content": input_text})

        try:
            full_response = ""
            token_count = 0
            start_time = time.time()
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
                            elapsed = time.time() - start_time
                            tps = token_count / elapsed if elapsed > 0 else 0
                            print(
                                f"\n({token_count} tokens in {elapsed:.2f}s — {tps:.1f} tok/s)"
                            )
                            break

                        token = json.loads(chunk)["choices"][0]["delta"]["content"]
                        print(token, end="", flush=True)
                        full_response += token
                        token_count += 1
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
