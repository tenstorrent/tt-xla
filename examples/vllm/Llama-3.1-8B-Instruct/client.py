# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass

import aiohttp
import requests


@dataclass
class UserResult:
    user_id: int
    token_count: int
    ttft: float
    elapsed: float
    tokens_per_sec: float
    text: str


def run_single_user(url, headers, data):
    """Original single-user interactive streaming (unchanged behavior)."""
    full_response = ""
    token_count = 0
    start_time = time.perf_counter()
    ttft = None
    with requests.post(url, headers=headers, json=data, stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
            chunk = chunk.removeprefix("data: ")
            if chunk:
                if "[DONE]" in chunk:
                    elapsed = time.perf_counter() - start_time
                    tok_s = token_count / elapsed if elapsed > 0 else 0
                    print(
                        f"\n[{token_count} tokens, TTFT: {ttft:.3f}s, {tok_s:.2f} tok/s]"
                    )
                    break

                token = json.loads(chunk)["choices"][0]["delta"]["content"]
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                token_count += 1
                print(token, end="", flush=True)
                full_response += token
    return full_response


class MultiUserDisplay:
    """Manages real-time display with each user on their own line."""

    def __init__(self, num_users):
        self.num_users = num_users
        self.texts = [""] * num_users
        self.lock = asyncio.Lock()
        self.started = False
        try:
            self.term_width = os.get_terminal_size().columns
        except OSError:
            self.term_width = 80

    async def update(self, user_id, token):
        async with self.lock:
            self.texts[user_id - 1] += token
            self._redraw()

    def _redraw(self):
        if self.started:
            print(f"\033[{self.num_users}A", end="")
        self.started = True
        for i, text in enumerate(self.texts):
            label = f"[User {i + 1}] "
            max_text = self.term_width - len(label) - 1
            display = text[-max_text:] if len(text) > max_text else text
            display = display.replace("\n", " ")
            line = f"{label}{display}"
            print(f"\033[2K{line}", flush=True)


async def stream_user(session, url, data, user_id, display):
    """Stream a single user's response asynchronously."""
    start_time = time.perf_counter()
    ttft = None
    token_count = 0
    text = ""

    async with session.post(url, json=data) as response:
        response.raise_for_status()
        buffer = ""
        async for chunk_bytes in response.content.iter_any():
            buffer += chunk_bytes.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                event_data = line[len("data: ") :]
                if event_data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(event_data)
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        if ttft is None:
                            ttft = time.perf_counter() - start_time
                        token_count += 1
                        token = delta["content"]
                        text += token
                        await display.update(user_id, token)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

    elapsed = time.perf_counter() - start_time
    tok_s = token_count / elapsed if elapsed > 0 else 0
    return UserResult(
        user_id=user_id,
        token_count=token_count,
        ttft=ttft or elapsed,
        elapsed=elapsed,
        tokens_per_sec=tok_s,
        text=text,
    )


async def run_multi_user(url, base_data, num_users):
    """Fire num_users concurrent requests and stream responses."""
    display = MultiUserDisplay(num_users)
    for _ in range(num_users - 1):
        print()
    async with aiohttp.ClientSession() as session:
        tasks = [
            stream_user(session, url, dict(base_data), i + 1, display)
            for i in range(num_users)
        ]
        return await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-users", type=int, default=1, help="Concurrent users (default: 1)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Max output tokens per response"
    )
    args = parser.parse_args()

    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
        ],
        "stream": True,
    }
    if args.max_tokens is not None:
        data["max_tokens"] = args.max_tokens

    while True:
        input_text = input("Enter a message (or 'q' to quit): ")
        if input_text == "q":
            break

        data["messages"].append({"role": "user", "content": input_text})

        try:
            if args.num_users == 1:
                full_response = run_single_user(url, headers, data)
                data["messages"].append({"role": "assistant", "content": full_response})
            else:
                wall_start = time.perf_counter()
                results = asyncio.run(run_multi_user(url, data, args.num_users))
                wall_time = time.perf_counter() - wall_start

                total_tokens = sum(r.token_count for r in results)
                print(f"{'='*60}")
                for r in results:
                    print(
                        f"  [User {r.user_id}] {r.token_count} tokens, "
                        f"TTFT: {r.ttft:.3f}s, {r.tokens_per_sec:.2f} tok/s"
                    )
                print(f"  Aggregate: {total_tokens / wall_time:.2f} tok/s")
                print(f"{'='*60}\n")

                # Use first user's response for conversation history
                data["messages"].append(
                    {"role": "assistant", "content": results[0].text}
                )
        except Exception as e:
            if isinstance(
                e, (requests.exceptions.ConnectionError, aiohttp.ClientError)
            ):
                print(
                    "Server returned a connection error. This usually occurs when a "
                    "request is made before the service is ready. Please wait for the "
                    "service to be ready and try again."
                )
                data["messages"].pop()
            else:
                raise e


if __name__ == "__main__":
    main()
