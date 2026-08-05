# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Sampling-parameter conformance over the OpenAI HTTP endpoint.

The rest of ``sampling/`` drives the in-process ``LLM`` API. These tests spawn a
real ``vllm.entrypoints.openai.api_server`` and hit ``/v1/chat/completions``,
which is the only way to exercise the request -> ``SamplingParams`` translation
that the OpenAI serving layer performs. Three gaps live only on that path and are
invisible to the offline tests:

  * ``logprobs: true`` with no ``top_logprobs`` maps to ``SamplingParams(logprobs=0)``.
    ``metadata.py`` treats a 0 as "no logprobs wanted", so no logprobs tensors are
    produced, and vLLM's serving layer then IndexErrors into a 500. The offline
    tests only ever pass ``logprobs>=1``.
  * under ``cpu_sampling``, any logprobs request kills EngineCore outright.
  * ``seed`` is not honored, and ``cpu_sampling: True`` does not fix it — contrary
    to the workaround documented on ``test_sampling_params.py::test_seed``.

``n`` is covered here too. It works, but nothing else in the tree tests it.

Uses opt-125m rather than a production model to keep the runtime down; the file
still takes ~12 min end to end, most of it the two server startups (device-
sampling and cpu_sampling fixtures) plus generation for the concurrency test.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

MODEL = "facebook/opt-125m"
SERVER_STARTUP_TIMEOUT = 600  # CI can be slow: model download + compilation
REQUEST_TIMEOUT = 120
CONCURRENT_REQUESTS = 16

# opt-125m ships no chat template; /v1/chat/completions requires one.
CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

# CI shared runners route local URLs through a proxy that answers 403.
_session = requests.Session()
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


def _start_server(extra_config=None):
    """Spawn an OpenAI api_server on a free port; yield its base URL."""
    port = _find_free_port()

    template_fd, template_path = tempfile.mkstemp(suffix=".jinja")
    os.write(template_fd, CHAT_TEMPLATE.encode())
    os.close(template_fd)

    # Log to a file, not a pipe: a full pipe buffer deadlocks startup.
    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="vllm_sampling_")
    log_file = os.fdopen(log_fd, "w")

    additional_config = {"min_context_len": 32}
    additional_config.update(extra_config or {})

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
        # The plugin asserts max_num_batched_tokens >= max_model_len * max_num_seqs,
        # so this cannot stay at 128 once max_num_seqs is raised for the
        # concurrent seeding test.
        "--max-num-batched-tokens",
        "4096",
        "--max-num-seqs",
        str(CONCURRENT_REQUESTS),
        "--gpu-memory-utilization",
        "0.001",
        "--chat-template",
        template_path,
        "--additional-config",
        json.dumps(additional_config),
    ]

    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

        deadline = time.time() + SERVER_STARTUP_TIMEOUT
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(
                    f"vLLM server exited with code {proc.returncode} before becoming "
                    f"ready.\nOutput:\n{_read_tail(log_path)}"
                )
            try:
                if (
                    _session.get(
                        f"http://localhost:{port}/health", timeout=5
                    ).status_code
                    == 200
                ):
                    break
            except requests.ConnectionError:
                pass
            time.sleep(2)
        else:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)
            pytest.fail(
                f"vLLM server did not become ready within {SERVER_STARTUP_TIMEOUT}s\n"
                f"Output:\n{_read_tail(log_path)}"
            )

        yield f"http://localhost:{port}"

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


@pytest.fixture(scope="module")
def server():
    """Server with the default on-device sampler."""
    yield from _start_server()


@pytest.fixture(scope="module")
def cpu_sampling_server():
    """Server with host-side sampling, the documented seed workaround."""
    yield from _start_server({"cpu_sampling": True})


def _chat(base_url, **overrides):
    """POST /v1/chat/completions; return (status_code, body)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Tell me a joke."}],
        "max_tokens": 16,
    }
    payload.update(overrides)
    resp = _session.post(
        f"{base_url}/v1/chat/completions", json=payload, timeout=REQUEST_TIMEOUT
    )
    return resp.status_code, resp.json()


def _content(body):
    return body["choices"][0]["message"]["content"].strip()


# --- n -------------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("n_val", [2, 3])
def test_chat_n(server, n_val):
    """`n` must return exactly n choices.

    No other test in the tree covers `n`. It passes today; this guards it.
    """
    status, body = _chat(server, n=n_val)
    assert status == 200, f"Expected 200, got {status}: {body}"
    assert (
        len(body["choices"]) == n_val
    ), f"n={n_val} should yield {n_val} choices, got {len(body['choices'])}"


# --- logprobs ------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.xfail(
    strict=True,
    reason=(
        "logprobs=true without top_logprobs maps to SamplingParams(logprobs=0). "
        "metadata.py:220 computes needs_logprobs as "
        "`max_num_logprobs > 0 if max_num_logprobs else False`, so a 0 is read as "
        "'no logprobs wanted' and none are produced; vLLM's serving layer then "
        "raises IndexError in _create_chat_logprobs and returns a 500. "
        "logprobs=0 is a valid request meaning 'the sampled token's logprob, no "
        "alternatives'."
    ),
)
def test_chat_logprobs_without_top_logprobs(server):
    """`logprobs: true` alone must return the sampled token's logprob."""
    status, body = _chat(server, logprobs=True)
    assert status == 200, f"Expected 200, got {status}: {body}"
    choice = body["choices"][0]
    assert choice.get("logprobs") is not None, f"logprobs missing from choice: {choice}"
    assert len(choice["logprobs"]["content"]) > 0


@pytest.mark.nightly
@pytest.mark.single_device
def test_chat_logprobs_with_top_logprobs(server):
    """`logprobs: true` + `top_logprobs: N` works — the contrast case.

    This is the shape the offline tests exercise (logprobs >= 1), which is why
    they never caught the logprobs=0 path above.
    """
    status, body = _chat(server, logprobs=True, top_logprobs=3)
    assert status == 200, f"Expected 200, got {status}: {body}"
    entries = body["choices"][0]["logprobs"]["content"]
    assert len(entries) > 0
    for entry in entries:
        assert "logprob" in entry and "token" in entry
        assert len(entry["top_logprobs"]) > 0


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Under cpu_sampling, gather_logprobs receives a host LongTensor for "
        "selected_token_ids and `logprobs.gather(-1, token_ids)` raises "
        "'Input tensor is not an XLA tensor' (sampler.py:253), killing EngineCore. "
        "Every subsequent request on that server then 500s."
    ),
)
def test_chat_logprobs_cpu_sampling(cpu_sampling_server):
    """Requesting logprobs under cpu_sampling must not kill the engine."""
    status, body = _chat(cpu_sampling_server, logprobs=True, top_logprobs=3)
    assert status == 200, f"Expected 200, got {status}: {body}"
    assert len(body["choices"][0]["logprobs"]["content"]) > 0


# --- seed ----------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.xfail(
    strict=True,
    reason=(
        "seed is not honored over HTTP. Device sampler side is tt-xla #4539 "
        "(tt::sampling uses one shared seed across cores and ignores per-row "
        "q_samples); see test_chat_seed_reproducibility_cpu_sampling for the "
        "host-sampling case."
    ),
)
def test_chat_seed_reproducibility(server):
    """Same seed + same prompt must produce the same completion."""
    payload = {"seed": 42, "temperature": 0.5}
    first = _content(_chat(server, **payload)[1])
    second = _content(_chat(server, **payload)[1])
    assert first == second, f"seed=42 not reproducible: {first!r} vs {second!r}"


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.xfail(
    strict=True,
    reason=(
        "The workaround documented on test_sampling_params.py::test_seed — "
        "additional_config={'cpu_sampling': True} — does not restore seeded "
        "determinism. Reproduced both over HTTP and via the in-process LLM API, "
        "so this is not an HTTP-path-only bug and #4539's kernel-level scope does "
        "not cover it."
    ),
)
def test_chat_seed_reproducibility_cpu_sampling(cpu_sampling_server):
    """The documented cpu_sampling workaround must make seeding deterministic."""
    payload = {"seed": 42, "temperature": 0.5}
    first = _content(_chat(cpu_sampling_server, **payload)[1])
    second = _content(_chat(cpu_sampling_server, **payload)[1])
    assert first == second, f"seed=42 not reproducible: {first!r} vs {second!r}"


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.xfail(
    strict=True,
    reason="seed is not honored; see test_chat_seed_reproducibility. tt-xla #4539.",
)
def test_chat_seed_under_concurrency(server):
    """Concurrent requests sharing one seed must all produce the same output.

    Mirrors the shape of tt-inference-server's VLLMParamConformanceTest
    test_non_uniform_seeding, which fires 32 same-seed requests at once. Batched
    execution is where a single shared device-side seed diverges from per-row
    seeding, so the single-request test above can pass by luck where this cannot.
    """

    def one(_):
        return _content(
            _chat(
                server,
                seed=0,
                temperature=0.9,
                max_tokens=24,
                messages=[
                    {"role": "user", "content": "Generate a list of 10 random colors."}
                ],
            )[1]
        )

    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
        outputs = list(pool.map(one, range(CONCURRENT_REQUESTS)))

    unique = set(outputs)
    assert len(unique) == 1, (
        f"seed=0 across {CONCURRENT_REQUESTS} concurrent requests should give 1 "
        f"unique output, got {len(unique)}"
    )
