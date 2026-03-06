# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the OpenAI /v1/responses API endpoint.

These tests start a vLLM server as a subprocess and exercise the /v1/responses
endpoint to validate that the TT backend works correctly with the Responses API.

The /v1/responses endpoint requires a chat template. Since facebook/opt-125m
does not include one, we provide a minimal template via --chat-template.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

import pytest
import requests

MODEL = "facebook/opt-125m"
SERVER_STARTUP_TIMEOUT = 600  # seconds (CI can be slow: model download + compilation)
REQUEST_TIMEOUT = 120  # seconds

# Minimal chat template for models that lack one (e.g. opt-125m).
CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


def get_output_text(result):
    """Extract generated text from a /v1/responses JSON result.

    Mirrors the OpenAI SDK Response.output_text property: walks the output
    list for message items and concatenates their output_text content blocks.
    """
    texts = []
    for item in result.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))
    return "".join(texts)


def _find_free_port():
    """Ask the OS for a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def vllm_server():
    """Start a vLLM OpenAI-compatible server and wait for it to be ready."""
    port = _find_free_port()
    base_url = f"http://localhost:{port}"

    # Write the chat template to a temp file (vLLM --chat-template accepts a path).
    template_fd, template_path = tempfile.mkstemp(suffix=".jinja")
    os.write(template_fd, CHAT_TEMPLATE.encode())
    os.close(template_fd)

    # Write server stdout to a log file instead of a pipe to avoid deadlock.
    # A pipe buffer (~64KB on Linux) can fill up during startup, blocking the
    # server before it passes the health check.
    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="vllm_server_")
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
        "128",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.001",
        "--chat-template",
        template_path,
        "--additional-config",
        json.dumps(
            {
                "enable_const_eval": False,
                "min_context_len": 32,
            }
        ),
    ]

    proc = None
    try:
        print(f"\nStarting vLLM server: {' '.join(cmd[:6])}... (port={port})")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        print(f"Server process started (pid={proc.pid})")

        health_url = f"{base_url}/health"
        start_time = time.time()
        deadline = start_time + SERVER_STARTUP_TIMEOUT
        ready = False
        print(
            f"\nWaiting for vLLM server on port {port} (timeout={SERVER_STARTUP_TIMEOUT}s)..."
        )
        while time.time() < deadline:
            elapsed = time.time() - start_time
            if proc.poll() is not None:
                log_tail = _read_tail(log_path, chars=8000)
                pytest.fail(
                    f"vLLM server exited with code {proc.returncode} after {elapsed:.0f}s.\n"
                    f"Output (last 8000 chars):\n{log_tail}"
                )
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    print(f"vLLM server ready after {elapsed:.0f}s")
                    ready = True
                    break
                else:
                    print(f"  [{elapsed:.0f}s] /health returned {resp.status_code}")
            except requests.ConnectionError:
                pass
            except Exception as e:
                print(f"  [{elapsed:.0f}s] health check error: {type(e).__name__}: {e}")
            time.sleep(2)

        if not ready:
            elapsed = time.time() - start_time
            print(f"Server NOT ready after {elapsed:.0f}s, sending SIGTERM...")
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)
            log_tail = _read_tail(log_path, chars=8000)
            pytest.fail(
                f"vLLM server did not become ready within {SERVER_STARTUP_TIMEOUT}s\n"
                f"Output (last 8000 chars):\n{log_tail}"
            )

        yield base_url

        print(f"\nTearing down vLLM server (pid={proc.pid})...")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
            print("Server shut down cleanly.")
        except subprocess.TimeoutExpired:
            print("Server didn't stop in 30s, killing...")
            proc.kill()
            proc.wait()
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()
        log_file.close()
        os.unlink(log_path)
        os.unlink(template_path)


def _read_tail(path, chars=2000):
    """Read the last ``chars`` characters from a file."""
    try:
        with open(path) as f:
            content = f.read()
        return content[-chars:]
    except OSError:
        return "<could not read log>"


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(
    "input_value",
    [
        "Hello, my name is",
        [{"role": "user", "content": "Once upon a time, there was a"}],
    ],
    ids=["string_input", "message_input"],
)
def test_responses_api_basic(vllm_server, input_value):
    """Test text generation via /v1/responses with string and message inputs."""
    url = f"{vllm_server}/v1/responses"
    data = {
        "model": MODEL,
        "input": input_value,
        "max_output_tokens": 32,
    }

    response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
    assert (
        response.status_code == 200
    ), f"Expected 200, got {response.status_code}: {response.text}"

    result = response.json()
    assert "id" in result, "Response should contain an 'id' field"
    assert "output" in result, "Response should contain an 'output' field"
    assert len(result["output"]) > 0, "Response should have at least one output item"

    output_text = get_output_text(result)
    assert len(output_text) > 0, "Expected non-empty output text"
    print(f"Input: {input_value!r}")
    print(f"Output: {output_text}")


@pytest.mark.nightly
@pytest.mark.single_device
def test_responses_api_streaming(vllm_server):
    """Test /v1/responses with streaming enabled.

    The responses API uses SSE with ``event: <type>`` + ``data: <json>``
    lines. The stream ends after a ``response.completed`` event.
    """
    url = f"{vllm_server}/v1/responses"
    data = {
        "model": MODEL,
        "input": "The capital of France is",
        "max_output_tokens": 32,
        "stream": True,
    }

    collected_events = []
    with requests.post(url, json=data, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        assert (
            resp.status_code == 200
        ), f"Expected 200, got {resp.status_code}: {resp.text}"
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            # SSE format: "event: <type>\ndata: <json>"
            if line.startswith("data: "):
                event_data = line[len("data: ") :]
                if event_data.strip() == "[DONE]":
                    break
                try:
                    collected_events.append(json.loads(event_data))
                except json.JSONDecodeError:
                    pass

    assert len(collected_events) > 0, "Expected at least one streaming event"

    event_types = [e.get("type") for e in collected_events]
    assert (
        "response.output_text.delta" in event_types
    ), f"Expected text delta events, got types: {event_types}"
    assert (
        "response.completed" in event_types
    ), f"Expected response.completed event, got types: {event_types}"

    streamed_text = "".join(
        e.get("delta", "")
        for e in collected_events
        if e.get("type") == "response.output_text.delta"
    )
    assert len(streamed_text) > 0, "Expected non-empty streamed text"

    print(f"Input: 'The capital of France is'")
    print(f"Received {len(collected_events)} streaming events")
    print(f"Streamed text: {streamed_text}")


@pytest.mark.nightly
@pytest.mark.single_device
def test_responses_api_deterministic(vllm_server):
    """Test deterministic output with temperature=0."""
    url = f"{vllm_server}/v1/responses"
    data = {
        "model": MODEL,
        "input": "Hello, my name is",
        "max_output_tokens": 16,
        "temperature": 0.0,
    }

    response1 = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
    assert response1.status_code == 200, f"Request 1 failed: {response1.text}"

    response2 = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
    assert response2.status_code == 200, f"Request 2 failed: {response2.text}"

    text1 = get_output_text(response1.json())
    text2 = get_output_text(response2.json())

    assert len(text1) > 0, "Expected non-empty output from request 1"
    assert len(text2) > 0, "Expected non-empty output from request 2"
    assert text1 == text2, (
        f"Expected deterministic output with temperature=0.\n"
        f"Response 1: {text1!r}\n"
        f"Response 2: {text2!r}"
    )
    print(f"Deterministic output: {text1!r}")


@pytest.mark.nightly
@pytest.mark.single_device
def test_responses_api_instructions(vllm_server):
    """Test /v1/responses with the instructions field (system prompt equivalent)."""
    url = f"{vllm_server}/v1/responses"
    data = {
        "model": MODEL,
        "input": "What should I do today?",
        "instructions": "You are a helpful travel guide who recommends activities.",
        "max_output_tokens": 32,
    }

    response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
    assert (
        response.status_code == 200
    ), f"Expected 200, got {response.status_code}: {response.text}"

    output_text = get_output_text(response.json())
    assert len(output_text) > 0, "Expected non-empty output text"
    print(f"Instructions: {data['instructions']!r}")
    print(f"Input: {data['input']!r}")
    print(f"Output: {output_text}")


@pytest.mark.nightly
@pytest.mark.single_device
def test_responses_api_top_logprobs(vllm_server):
    """Test /v1/responses with top_logprobs to verify logprob extraction works."""
    url = f"{vllm_server}/v1/responses"
    data = {
        "model": MODEL,
        "input": "The sky is",
        "max_output_tokens": 5,
        "temperature": 0.0,
        "top_logprobs": 3,
    }

    response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
    assert (
        response.status_code == 200
    ), f"Expected 200, got {response.status_code}: {response.text}"

    result = response.json()
    output_text = get_output_text(result)
    assert len(output_text) > 0, "Expected non-empty output text"

    # Verify logprobs are present in the output content
    for item in result.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    logprobs = content.get("logprobs", [])
                    assert len(logprobs) > 0, "Expected logprobs in output_text content"
                    for entry in logprobs:
                        assert "token" in entry, "Logprob entry should have 'token'"
                        assert "logprob" in entry, "Logprob entry should have 'logprob'"
                        top = entry.get("top_logprobs", [])
                        assert len(top) > 0, "Expected top_logprobs alternatives"
                    print(f"Input: 'The sky is'")
                    print(f"Output: {output_text}")
                    print(f"First token logprobs: {logprobs[0]}")
                    return

    pytest.fail("No output_text content found with logprobs")
