# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import requests


def main():
    url = "http://localhost:8000/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "BAAI/bge-m3",
        "input": "",
    }

    while True:
        input_text = input("Enter a text to embed (or 'q' to quit): ")
        if input_text == "q":
            break

        data["input"] = input_text

        try:
            response = requests.post(url, headers=headers, json=data)
            print(f"Embedding: {response.json()['data'][0]['embedding']}")
            print(f"Elapsed time: {response.elapsed.total_seconds()}")
        except Exception as e:
            if isinstance(e, requests.exceptions.ConnectionError):
                print(
                    "Server returned a connection error. This usually occurs when a request is made before the service is ready. Please wait for the service to be ready and try again."
                )
            else:
                raise e


if __name__ == "__main__":
    main()
