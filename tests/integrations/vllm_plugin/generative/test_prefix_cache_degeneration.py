# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Reproducer for the TT cached-prefill greedy degeneration (tt-xla #5132).

A greedy (temperature=0) completion of the exact json-mode-eval[0] prompt (below)
generates normally on a prefix-cache MISS but collapses to empty output on a
subsequent cache HIT. Requires enable_trace=True and the exact prompt to trigger.
The test fails on the buggy build and passes once the cached-prefill path is fixed.
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
import requests

MODEL = "tiiuae/Falcon3-7B-Instruct"
SERVER_STARTUP_TIMEOUT = 2400  # trace + 16k is a heavy compile
REQUEST_TIMEOUT = 300

# EXACT NousResearch/json-mode-eval[0] prompt, Falcon3 chat template
# (add_generation_prompt=True). The degeneration is prompt-specific — it must
# be this exact prompt, not a paraphrase, to reproduce.
PROMPT = (
    "<|system|>\nYou are a helpful assistant that answers in JSON. Here's the "
    "json schema you must adhere to:\n<schema>\n{'title': 'WirelessAccessPoint', "
    "'type': 'object', 'properties': {'ssid': {'title': 'SSID', 'type': "
    "'string'}, 'securityProtocol': {'title': 'SecurityProtocol', 'type': "
    "'string'}, 'bandwidth': {'title': 'Bandwidth', 'type': 'string'}}, "
    "'required': ['ssid', 'securityProtocol', 'bandwidth']}\n</schema>\n\n"
    "<|user|>\nI'm currently configuring a wireless access point for our office "
    "network and I need to generate a JSON object that accurately represents "
    "its settings. The access point's SSID should be 'OfficeNetSecure', it uses "
    "WPA2-Enterprise as its security protocol, and it's capable of a bandwidth "
    "of up to 1300 Mbps on the 5 GHz band. This JSON object will be used to "
    "document our network configurations and to automate the setup process for "
    "additional access points in the future. Please provide a JSON object that "
    "includes these details.\n<|assistant|>\n"
)

_session = requests.Session()
_session.trust_env = False


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _read_tail(path, chars=20000):
    # 20k (was 2k): a server crash surfaces only the APIServer "Engine core
    # initialization failed. See root cause above." wrapper in the last ~2k
    # chars; the real EngineCore error (OOM / mask crash / compile) sits above
    # that window, so a small tail hides exactly the line we need.
    try:
        with open(path) as f:
            return f.read()[-chars:]
    except OSError:
        return "<could not read log>"


def _probe(port: int) -> dict:
    """Stream one greedy completion of PROMPT and return chunk counts.

    ``text_chunks`` counts SSE chunks whose ``text`` is non-empty (real output).
    ``choices_chunks`` counts chunks carrying a ``choices`` array at all,
    mirroring the vLLM benchmark client's ``first_chunk_received`` test
    (vllm/benchmarks/lib/endpoint_request_func.py) — an empty-text chunk still
    counts there, so reporting both lets us see how close we are to the bench's
    zero-``choices``-chunk failure condition.
    """
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "temperature": 0.0,
        "max_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    choices_chunks = 0
    text_chunks = 0
    with _session.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            raw = line[len("data: ") :]
            if raw.strip() == "[DONE]":
                continue
            ch = json.loads(raw).get("choices")
            if ch:  # bench counts this as a received chunk (text may be empty)
                choices_chunks += 1
                if ch[0].get("text"):
                    text_chunks += 1
    return {"choices_chunks": choices_chunks, "text_chunks": text_chunks}


@pytest.fixture(scope="module")
def vllm_server():
    port = _find_free_port()
    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="vllm_prefixcache_")
    log_file = os.fdopen(log_fd, "w")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--port",
        str(port),
        # forge serving config.
        "--max-model-len",
        "4096",
        "--max-num-batched-tokens",
        "16384",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.4",
        # the cache-hit cached-prefill path is what degenerates.
        "--enable-prefix-caching",
        # enable_trace=True is required to trigger.
        "--additional-config",
        json.dumps(
            {
                "enable_const_eval": True,
                "min_context_len": 32,
                "experimental_weight_dtype": "bfp_bf8",
                "cpu_sampling": False,
                "optimization_level": 0,
                "enable_trace": True,
            }
        ),
    ]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + SERVER_STARTUP_TIMEOUT
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(
                    f"server exited ({proc.returncode}).\n{_read_tail(log_path)}"
                )
            try:
                if (
                    _session.get(
                        f"http://127.0.0.1:{port}/health", timeout=5
                    ).status_code
                    == 200
                ):
                    ready = True
                    break
            except requests.ConnectionError:
                pass
            time.sleep(3)
        if not ready:
            pytest.fail(
                f"server not ready in {SERVER_STARTUP_TIMEOUT}s\n{_read_tail(log_path)}"
            )
        yield port
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        log_file.close()
        os.unlink(log_path)


@pytest.mark.nightly
@pytest.mark.single_device
def test_cached_prefill_warm_hit_not_degenerate(vllm_server):
    """A greedy completion of PROMPT must produce the same non-empty output on a
    prefix-cache HIT (warm) as on the MISS (cold).

    The first request is a cache miss and generates the full answer; the second
    identical request is a cache hit. Under the bug the warm request collapses
    to zero output text (only an empty-text stop chunk). Greedy decoding of an
    identical prompt must be deterministic, so the warm request must also be
    non-empty.
    """
    cold = _probe(vllm_server)
    warm = _probe(vllm_server)
    print(f"cold (cache miss): {cold}")
    print(f"warm (cache hit) : {warm}")

    # Sanity: the cold request must generate, else the prompt/config is wrong
    # (not the bug under test).
    assert cold["text_chunks"] > 0, (
        f"cold request produced no output ({cold}); cannot test the cache-hit "
        f"path — check prompt/server config"
    )

    # The bug: the warm cache-hit request degenerates to zero output text.
    assert warm["text_chunks"] > 0, (
        f"warm prefix-cache HIT produced zero output text (cold={cold}, "
        f"warm={warm}) — cached-prefill greedy degeneration"
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_cached_prefill_concurrent_batch_not_degenerate(vllm_server):
    """4 identical greedy requests in a concurrent batch must all generate.

    Under the original bug the signature was ``text_chunks=[51, 0, 0, 0]``:
    row 0 was a cache miss and generated normally, rows 1-3 hit the prefix
    that row 0 had just inserted and degenerated. With the always-mask SDPA
    fix, all four requests must produce non-empty output. By the time this
    test runs, the prior sequential test has already populated the prefix
    cache, so all four requests here hit the cache — even more conservative
    than the original [51, 0, 0, 0] scenario.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_probe, [vllm_server] * 4))

    text_counts = [r["text_chunks"] for r in results]
    print(f"concurrent batch (4 identical greedy): {text_counts}")
    assert all(c > 0 for c in text_counts), (
        f"concurrent prefix-cache batch produced empty rows ({text_counts}) "
        f"— cached-prefill greedy degeneration in the concurrent path"
    )
