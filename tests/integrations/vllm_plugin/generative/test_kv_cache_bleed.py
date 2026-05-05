# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for KV cache bleed (#3899).

Two bugs caused cross-user response contamination:

1. logits_indices overflow (batched prefill): cumulative offsets overflow past
   each user's padded slot, reading from the wrong user's hidden states.

2. Prefix cache block overwrite (staggered prefill): paged_fill_cache writes
   suffix KV data to shared prefix blocks, corrupting cached data for
   concurrent requests.

Starts a vLLM server with TinyLlama-1.1B-Chat, sends concurrent and staggered
requests with distinct topics, and checks for cross-topic keyword contamination.
"""

import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

import pytest
import requests as http_requests

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SERVER_STARTUP_TIMEOUT = 600
REQUEST_TIMEOUT = 300

TOPICS = [
    ("Explain how penguins survive in Antarctica.", "penguin"),
    ("Describe how a submarine navigates underwater using sonar.", "submarine"),
    ("Tell me about how dinosaurs went extinct millions of years ago.", "dinosaur"),
    ("Explain the process of making chocolate from cacao beans.", "chocolate"),
]

SYSTEM_MSG = (
    "You are a helpful assistant. Always stay on topic. "
    "Only discuss what the user asks about. "
    "Do not mention unrelated subjects."
)

# Shared prefix (system message) must span at least one cache block (16 tokens)
# to trigger the prefix caching bug.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}System: {{ message['content'] }}\n{% endif %}"
    "{% if message['role'] == 'user' %}User: {{ message['content'] }}\nAssistant: {% endif %}"
    "{% endfor %}"
)

_session = http_requests.Session()
_session.trust_env = False


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _read_tail(path, chars=2000):
    try:
        with open(path) as f:
            return f.read()[-chars:]
    except OSError:
        return "<could not read log>"


def _send_request(url, prompt, idx, max_tokens=32):
    """Send a chat completion and return (idx, response_text)."""
    resp = _session.post(
        f"{url}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return idx, resp.json()["choices"][0]["message"]["content"]


def _check_bleed(texts, label):
    """Print responses and return failure messages for cross-topic contamination."""
    keywords = [kw for _, kw in TOPICS]
    failures = []
    for i, text in enumerate(texts):
        text_lower = text.lower()
        foreign = [kw for j, kw in enumerate(keywords) if j != i and kw in text_lower]
        status = f"BLEED {foreign}" if foreign else "ok"
        print(f"  {label} user {i} ({keywords[i]}): [{status}] {text[:80]}")
        if foreign:
            failures.append(
                f"{label} user {i} ({keywords[i]}): "
                f"foreign {foreign} in: {text[:80]}"
            )
    return failures


def _send_round(url, stagger_delay=0, max_tokens=32):
    """Send one round of requests. If stagger_delay > 0, stagger submissions."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(TOPICS)) as pool:
        futures = []
        for i, (prompt, _) in enumerate(TOPICS):
            futures.append(pool.submit(_send_request, url, prompt, i, max_tokens))
            if stagger_delay > 0:
                time.sleep(stagger_delay)
        texts = [""] * len(TOPICS)
        for f in concurrent.futures.as_completed(futures):
            idx, text = f.result()
            texts[idx] = text
    return texts


@pytest.fixture(scope="module")
def vllm_server():
    """Start a vLLM server with prefix caching enabled."""
    port = _find_free_port()
    base_url = f"http://localhost:{port}"

    template_fd, template_path = tempfile.mkstemp(suffix=".jinja")
    os.write(template_fd, CHAT_TEMPLATE.encode())
    os.close(template_fd)

    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="vllm_bleed_test_")
    log_file = os.fdopen(log_fd, "w")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--port",
        str(port),
        "--max-model-len",
        "192",
        "--max-num-batched-tokens",
        "768",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.001",
        "--enforce-eager",
        "--chat-template",
        template_path,
        "--additional-config",
        json.dumps({"enable_const_eval": False, "min_context_len": 32}),
    ]

    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

        deadline = time.time() + SERVER_STARTUP_TIMEOUT
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(
                    f"Server exited ({proc.returncode}).\n{_read_tail(log_path)}"
                )
            try:
                if _session.get(f"{base_url}/health", timeout=5).status_code == 200:
                    ready = True
                    break
            except http_requests.ConnectionError:
                pass
            time.sleep(2)

        if not ready:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)
            pytest.fail(f"Server not ready in {SERVER_STARTUP_TIMEOUT}s")

        yield base_url

        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()
        log_file.close()
        os.unlink(log_path)
        os.unlink(template_path)


@pytest.mark.nightly
@pytest.mark.single_device
def test_kv_cache_no_cross_user_bleed(vllm_server):
    """Concurrent and staggered requests must not leak data between users.

    Phase 1 (bug #1): all 4 requests sent at once — batched prefill.
    Phase 2 (bug #2): staggered overlapping requests — prefix cache reuse.
    """
    all_failures = []

    # Phase 1: concurrent (triggers batched prefill logits_indices overflow)
    print("\n--- Phase 1: concurrent requests (batched prefill) ---")
    for r in range(3):
        texts = _send_round(vllm_server)
        all_failures.extend(_check_bleed(texts, f"concurrent r{r + 1}"))

    # Phase 2: staggered with overlap (triggers prefix cache block overwrite)
    # Longer generation (96 tokens) ensures requests overlap so a prefilling
    # request can overwrite a prefix block while another is still decoding.
    print("\n--- Phase 2: staggered requests (prefix cache reuse) ---")
    for r in range(5):
        texts = _send_round(vllm_server, stagger_delay=0.8, max_tokens=96)
        all_failures.extend(_check_bleed(texts, f"staggered r{r + 1}"))

    assert (
        not all_failures
    ), f"KV cache bleed in {len(all_failures)} cases:\n" + "\n".join(all_failures[:10])
