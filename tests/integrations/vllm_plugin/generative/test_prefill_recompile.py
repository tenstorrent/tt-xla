# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test: no XLA graph recompiles after warmup across an ISL sweep.

Sweeps prompt lengths over every prefill token-bucket of a 4K context and asserts
the XLA cached-graph count does not grow. Chunked prefill is on, so ISLs above
the chunk split into continuation chunks routed through the chunked-SDPA op --
covering the cached-prefix warmup in _precompile_model_fused, whose graph must be
precompiled at init or the first continuation chunk recompiles mid-serving.

The graph counter is process-local, so the engine must run in-process
(VLLM_ENABLE_V1_MULTIPROCESSING=0), which inits the once-only XLA computation
cache; to avoid poisoning other vLLM tests, pytest re-execs this file as a
throwaway child worker.
"""
import os
import subprocess
import sys

import pytest

# Non-gated and small; only 1 layer is compiled below, so init stays fast.
MODEL = "Qwen/Qwen3-0.6B"
MAX_LEN = 4096
MAX_TOKENS = 4

# Per-seq chunk (< MAX_LEN enables chunked prefill / the chunked-SDPA path).
PREFILL_CHUNK = 512

# ISLs covering each prefill bucket, mixing aligned and padded-up lengths.
# Those above PREFILL_CHUNK split into continuation chunks.
SWEEP_ISLS = (128, 200, 512, 1000, 2048, 3000, 4000)

# No continuation chunk => chunked-prefix path untested (silent pass).
assert max(SWEEP_ISLS) > PREFILL_CHUNK

WORKER_TIMEOUT = 1800


@pytest.mark.push
@pytest.mark.single_device
def test_no_prefill_recompile():
    # Isolate the in-process engine in a fresh child (see module docstring). A
    # clean subprocess — not os.fork()/pytest-forked — is required: the parent
    # pytest worker has already loaded the PJRT plugin, so a forked child would
    # not guarantee the pristine XLA cache this test depends on.
    env = {**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
    try:
        proc = subprocess.run(
            [sys.executable, __file__],
            env=env,
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout, stderr, rc = e.stdout or "", e.stderr or "", None

    # Surface the worker's [recompile-test] trace for debugging either way.
    print(stdout, flush=True)
    if stderr:
        print(stderr, file=sys.stderr, flush=True)
    assert rc == 0, (
        f"prefill-recompile worker failed (exit={rc}) — prefill graphs recompiled "
        "at runtime or the worker timed out; see output above."
    )


def _run_sweep():
    """Worker body (runs in the isolated child process). Returns exit code."""
    import torch_xla.runtime as xr
    import vllm

    llm = vllm.LLM(
        model=MODEL,
        max_model_len=MAX_LEN,
        max_num_seqs=1,
        gpu_memory_utilization=0.1,
        enable_prefix_caching=False,
        # platform.py derives max_num_batched_tokens from prefill_chunk_size.
        additional_config={
            "num_hidden_layers": 1,
            "min_context_len": 128,
            "prefill_chunk_size": PREFILL_CHUNK,
        },
    )
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    def gen(n):
        llm.generate({"prompt_token_ids": [100] * n}, sp, use_tqdm=False)

    gen(64)  # one warmup request absorbs the one-time decode/sampling graphs
    baseline = xr.get_num_cached_compilation_graph()
    print(f"\n[recompile-test] warmup done; cached XLA graphs = {baseline}", flush=True)
    # If this is 0 the engine ran out-of-process and recompiles can't be seen.
    if baseline <= 0:
        print(
            "[recompile-test] FAIL: graph counter is 0 — engine did not run "
            "in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0 not honored).",
            flush=True,
        )
        return 1

    # Sweep each ISL and assert no new graphs were compiled.
    print(f"[recompile-test] sweeping {len(SWEEP_ISLS)} ISLs: {SWEEP_ISLS}", flush=True)
    prev = baseline
    for n in SWEEP_ISLS:
        gen(n)
        cur = xr.get_num_cached_compilation_graph()
        flag = "  <-- RECOMPILE" if cur > prev else ""
        print(
            f"[recompile-test] ISL={n:5d} -> +{cur - prev} graph(s){flag}", flush=True
        )
        prev = cur

    total = prev - baseline
    print(f"[recompile-test] total new graphs during sweep: {total}", flush=True)
    if total != 0:
        print(
            f"[recompile-test] FAIL: {total} XLA graph(s) compiled during the ISL "
            "sweep — prefill graphs are recompiling at runtime.",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_run_sweep())
