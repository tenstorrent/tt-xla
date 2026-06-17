# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test: no XLA graph recompiles after warmup across an ISL sweep.

Warms up, then sweeps prompt lengths across every prefill token-bucket of a 4K
context and asserts the XLA cached-graph count does not grow — i.e. all prefill
graphs were precompiled at init. A single transformer layer keeps compile fast.
"""
import os

import pytest
import torch_xla.runtime as xr
import vllm

# Run in-process so the (process-local) graph counter sees the engine's compiles
# (vLLM reads this lazily at engine creation, so setting it after import is fine).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# Non-gated and small; only 1 layer is compiled below, so init stays fast.
MODEL = "Qwen/Qwen3-0.6B"
MAX_LEN = 4096
MAX_TOKENS = 4

# ISLs covering each prefill bucket (128/256/512/1024/2048/4096), mixing power-of-
# two (aligned) and non-power-of-two (padded up) lengths:
#   128->128, 200->256, 512->512, 1000->1024, 2048->2048, 3000->4096, 4000->4096
SWEEP_ISLS = (128, 200, 512, 1000, 2048, 3000, 4000)


@pytest.mark.push
@pytest.mark.single_device
def test_no_prefill_recompile():
    llm = vllm.LLM(
        model=MODEL,
        max_model_len=MAX_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=MAX_LEN,
        gpu_memory_utilization=0.1,
        enable_prefix_caching=False,  # each ISL fully prefills its own bucket
        additional_config={"num_hidden_layers": 1, "min_context_len": 128},
    )
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    def gen(n):
        llm.generate({"prompt_token_ids": [100] * n}, sp, use_tqdm=False)

    gen(64)  # one warmup request absorbs the one-time decode/sampling graphs
    baseline = xr.get_num_cached_compilation_graph()
    print(f"\n[recompile-test] warmup done; cached XLA graphs = {baseline}", flush=True)
    # If this is 0 the engine ran out-of-process and recompiles can't be seen.
    assert baseline > 0, "graph counter is 0 — need VLLM_ENABLE_V1_MULTIPROCESSING=0"

    # Sweep each ISL and assert no new graphs were compiled
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
    assert total == 0, (
        f"{total} XLA graph(s) compiled during the ISL sweep — prefill graphs are "
        "recompiling at runtime instead of being precompiled at engine init."
    )
