# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Client for the prefill last-token-select recompile repro (see
# recompile_repro_service.sh). Sends a SHORT request (warms + locks the engine on a
# small prefill bucket), then a LONGER request whose prompt lands in a new prefill
# bucket. On the affected path that second request JIT-compiles its select graph:
#   - with VLLM_XLA_CHECK_RECOMPILATION=1, the server raises
#     "Recompilation after warm up is detected" and the request fails;
#   - otherwise it just stalls (the compile latency) on first use.
#
# Usage: python recompile_repro_client.py [host:port]
import sys
import time

import requests

ADDR = sys.argv[1] if len(sys.argv) > 1 else "localhost:8000"
URL = f"http://{ADDR}/v1/completions"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def send(n_tokens, label):
    # prompt as a list of token ids -> exact prefill length (100 is a safe in-vocab id)
    body = {
        "model": MODEL,
        "prompt": [100] * n_tokens,
        "max_tokens": 8,
        "temperature": 0,
    }
    t0 = time.time()
    try:
        r = requests.post(URL, json=body, timeout=900)
        dt = time.time() - t0
        ok = r.status_code == 200
        print(
            f"[{label}] tokens={n_tokens:5d} -> HTTP {r.status_code} in {dt:6.1f}s"
            f"{'' if ok else '  BODY=' + r.text[:200]}"
        )
        return ok
    except Exception as e:
        print(
            f"[{label}] tokens={n_tokens:5d} -> EXCEPTION after {time.time()-t0:.1f}s: {type(e).__name__}: {e}"
        )
        return False


def wait_ready(timeout=2400):
    print(f"[client] waiting for server at {ADDR} ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.post(
                URL,
                json={
                    "model": MODEL,
                    "prompt": [100] * 8,
                    "max_tokens": 4,
                    "temperature": 0,
                },
                timeout=900,
            )
            if r.status_code == 200:
                print(f"[client] server ready ({time.time()-t0:.0f}s)", flush=True)
                return True
            print(f"[client] not ready (HTTP {r.status_code}), retrying...", flush=True)
        except Exception:
            print("[client] not ready (conn), retrying...", flush=True)
        time.sleep(10)
    return False


def main():
    if not wait_ready():
        print("[client] server never became ready")
        sys.exit(1)
    print("\n[client] === recompile probe ===")
    send(64, "short  ")  # 128 bucket — already warmed; locks the engine
    send(2000, "long   ")  # 2048 bucket — NEW; affected path recompiles here
    send(2000, "long-rpt")  # same bucket — should be cached (no recompile)


if __name__ == "__main__":
    main()
