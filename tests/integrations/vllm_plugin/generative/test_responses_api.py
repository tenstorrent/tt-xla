# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the OpenAI /v1/responses API endpoint.

These tests start a vLLM server as a subprocess and exercise the /v1/responses
endpoint to validate that the TT backend works correctly with the Responses API.
"""

import json
import signal
import subprocess
import sys
import time

import pytest
import requests

MODEL = "facebook/opt-125m"
SERVER_PORT = 8321  # Non-default port to avoid conflicts
SERVER_HOST = "localhost"
SERVER_STARTUP_TIMEOUT = 300  # seconds
REQUEST_TIMEOUT = 120  # seconds


@pytest.fixture(scope="module")
def vllm_server():
    """Start a vLLM OpenAI-compatible server and wait for it to be ready."""
    base_url = f"http://{SERVER_HOST}:{SERVER_PORT}"

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--port",
        str(SERVER_PORT),
        "--max-model-len",
        "128",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.001",
        "--additional-config",
        json.dumps(
            {
                "enable_const_eval": False,
                "min_context_len": 32,
            }
        ),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    health_url = f"{base_url}/health"
    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            # Process exited prematurely — read whatever output we can.
            stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            pytest.fail(
                f"vLLM server exited with code {proc.returncode} before becoming ready.\n"
                f"Output:\n{stdout[-2000:]}"
            )
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                ready = True
                break
        except requests.ConnectionError:
            pass
        time.sleep(2)

    if not ready:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
        pytest.fail(
            f"vLLM server did not become ready within {SERVER_STARTUP_TIMEOUT}s"
        )

    yield base_url

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.mark.nightly
@pytest.mark.single_device
def test_responses_api_basic(vllm_server):
    """Test basic text generation via /v1/responses with a string input."""
    url = f"{vllm_server}/v1/responses"
    data = {
        "model": MODEL,
        "input": "Hello, my name is",
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
    assert (
        result.get("output_text") is not None
    ), "Response should contain 'output_text'"
    assert len(result["output_text"]) > 0, "Expected non-empty output text"
    print(f"Input: 'Hello, my name is'")
    print(f"Output: {result['output_text']}")


@pytest.mark.nightly
@pytest.mark.single_device
def test_responses_api_message_input(vllm_server):
    """Test /v1/responses with message-style input (list of role/content dicts)."""
    url = f"{vllm_server}/v1/responses"
    data = {
        "model": MODEL,
        "input": [
            {"role": "user", "content": "Once upon a time, there was a"},
        ],
        "max_output_tokens": 32,
    }

    response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
    assert (
        response.status_code == 200
    ), f"Expected 200, got {response.status_code}: {response.text}"

    result = response.json()
    assert "output" in result, "Response should contain an 'output' field"
    assert len(result["output"]) > 0, "Response should have at least one output item"

    # Verify the output contains a message item
    message_items = [item for item in result["output"] if item.get("type") == "message"]
    assert len(message_items) > 0, "Expected at least one message output item"
    print(f"Input: [user: 'Once upon a time, there was a']")
    print(f"Output: {result.get('output_text', '')}")


@pytest.mark.nightly
@pytest.mark.single_device
def test_responses_api_streaming(vllm_server):
    """Test /v1/responses with streaming enabled."""
    url = f"{vllm_server}/v1/responses"
    data = {
        "model": MODEL,
        "input": "The capital of France is",
        "max_output_tokens": 32,
        "stream": True,
    }

    collected_events = []
    got_done = False
    with requests.post(url, json=data, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        assert (
            resp.status_code == 200
        ), f"Expected 200, got {resp.status_code}: {resp.text}"
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            # SSE format: lines may be "event: <type>" or "data: <json>"
            if line.startswith("data: "):
                event_data = line[len("data: ") :]
                if event_data.strip() == "[DONE]":
                    got_done = True
                    break
                try:
                    collected_events.append(json.loads(event_data))
                except json.JSONDecodeError:
                    pass  # skip malformed events

    assert len(collected_events) > 0, "Expected at least one streaming event"
    assert got_done, "Expected [DONE] sentinel in stream"

    # The final event should contain the completed response
    last_event = collected_events[-1]
    assert last_event.get("type") in (
        "response.completed",
        "response.output_text.done",
        None,
    ) or "output_text" in str(
        last_event
    ), f"Unexpected final event: {last_event}"

    print(f"Input: 'The capital of France is'")
    print(f"Received {len(collected_events)} streaming events")


@pytest.mark.nightly
@pytest.mark.single_device
def test_responses_api_with_temperature(vllm_server):
    """Test /v1/responses with sampling parameters."""
    url = f"{vllm_server}/v1/responses"
    data = {
        "model": MODEL,
        "input": "Hello, my name is",
        "max_output_tokens": 16,
        "temperature": 0.0,
    }

    # Make two requests with temperature=0 and verify deterministic output
    response1 = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
    assert response1.status_code == 200, f"Request 1 failed: {response1.text}"

    response2 = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
    assert response2.status_code == 200, f"Request 2 failed: {response2.text}"

    text1 = response1.json().get("output_text", "")
    text2 = response2.json().get("output_text", "")

    assert len(text1) > 0, "Expected non-empty output from request 1"
    assert len(text2) > 0, "Expected non-empty output from request 2"
    assert text1 == text2, (
        f"Expected deterministic output with temperature=0.\n"
        f"Response 1: {text1!r}\n"
        f"Response 2: {text2!r}"
    )
    print(f"Deterministic output: {text1!r}")
