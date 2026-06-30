# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Memory-leak probe client for a running vLLM (forge plugin) server.

Floods the server with many concurrent /v1/chat/completions requests while
sampling the VLLM::EngineCore process's host RSS from /proc. A leak shows up as
EngineCore RSS that climbs ~linearly with generated tokens and does NOT drop
back after all requests drain and the server goes idle (transient KV/activation
memory would be released on sequence completion).

Self-contained: stdlib only (urllib + threads + /proc), no external deps.

Pair with service.sh. Example:
    python3 leak_probe.py --port 8000 --num-prompts 200 --concurrency 32 \
        --max-tokens 1024 --sample-secs 15
"""

import argparse
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

_KIB = 1024


def _gb(kb):
    return kb / (_KIB * _KIB)


def _iter_pids():
    try:
        return [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return []


def _proc_name(pid):
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except OSError:
        return "?"


def _proc_rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _read_meminfo():
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    out[parts[0].rstrip(":")] = int(parts[1])
    except OSError:
        pass
    return out


def _host_used_pct():
    mem = _read_meminfo()
    total = mem.get("MemTotal", 0)
    if not total:
        return 0.0
    return 100.0 * (total - mem.get("MemAvailable", 0)) / total


def _find_engine_core_pid():
    """PID of the live VLLM::EngineCore (largest RSS; ignores defunct ones)."""
    best_pid, best_rss = None, -1
    for pid in _iter_pids():
        if _proc_name(pid).startswith("VLLM::EngineCor"):  # comm is truncated
            rss = _proc_rss_kb(pid)
            if rss > best_rss:
                best_pid, best_rss = pid, rss
    return best_pid


def _make_prompt(i):
    """Distinct, non-trivial prompt per request so prefix caching does not
    return cached KV/output (which would mask real per-request memory)."""
    return (
        f"Request #{i}. Without repeating yourself, write a detailed, original "
        f"technical explanation (unique to id {i}) of a randomly chosen systems "
        f"topic. Be thorough and specific; do not stop early."
    )


def _send_one(url, api_key, model, prompt, max_tokens):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False,
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=1200) as resp:
        data = json.loads(resp.read())
    usage = data.get("usage", {}) or {}
    return usage.get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--sample-secs", type=int, default=15)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument(
        "--engine-pid",
        type=int,
        default=None,
        help="VLLM::EngineCore pid (default: auto-detect highest-RSS one)",
    )
    a = ap.parse_args()

    url = f"http://{a.host}:{a.port}/v1/chat/completions"
    engine_pid = a.engine_pid or _find_engine_core_pid()
    base_rss = _proc_rss_kb(engine_pid) if engine_pid else 0
    peak_rss = base_rss
    completed = 0
    tokens = 0
    lock = threading.Lock()
    stop = threading.Event()

    def _sampler():
        nonlocal peak_rss
        t0 = time.time()
        while not stop.is_set():
            rss = _proc_rss_kb(engine_pid) if engine_pid else 0
            peak_rss = max(peak_rss, rss)
            with lock:
                done = completed
            print(
                f"[probe t={int(time.time() - t0):>5}s] done={done}/{a.num_prompts} "
                f"host_used={_host_used_pct():.1f}% "
                f"EngineCore={_gb(rss):.2f}GB (Δ{_gb(rss - base_rss):+.2f}GB from start)",
                flush=True,
            )
            stop.wait(a.sample_secs)

    print(
        f"Probing {url} model={a.model} num_prompts={a.num_prompts} "
        f"concurrency={a.concurrency} max_tokens={a.max_tokens}"
    )
    print(
        f"EngineCore pid={engine_pid} baseline RSS={_gb(base_rss):.2f}GB "
        f"host_used={_host_used_pct():.1f}%"
    )
    sampler = threading.Thread(target=_sampler, name="mem-sampler", daemon=True)
    sampler.start()

    t_start = time.time()
    errors = 0
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [
            ex.submit(_send_one, url, a.api_key, a.model, _make_prompt(i), a.max_tokens)
            for i in range(a.num_prompts)
        ]
        for f in as_completed(futs):
            try:
                ctoks = f.result()
                with lock:
                    completed += 1
                    tokens += ctoks
            except Exception as e:
                errors += 1
                print(f"  request error: {e!r}", flush=True)
    stop.set()
    sampler.join(timeout=2)

    dur = time.time() - t_start
    final_rss = _proc_rss_kb(engine_pid) if engine_pid else 0
    print("\n=== leak probe summary ===")
    print(f"requests: {completed} ok, {errors} errored, in {dur:.0f}s")
    print(
        f"output tokens: {tokens} ({tokens / dur:.1f} tok/s aggregate)" if dur else ""
    )
    if engine_pid:
        print(
            f"EngineCore RSS: start {_gb(base_rss):.2f}GB -> end {_gb(final_rss):.2f}GB "
            f"(peak {_gb(peak_rss):.2f}GB, Δ {_gb(final_rss - base_rss):+.2f}GB)"
        )
        print(
            "  Climbing Δ that does not flatten once the KV pool is full / does "
            "not release after drain => leak; a one-time rise that plateaus and "
            "releases => normal KV/activation warmup."
        )


if __name__ == "__main__":
    main()
