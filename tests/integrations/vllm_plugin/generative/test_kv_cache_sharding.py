# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test: the KV cache is really head-sharded under DP+TP (#5796).

DP+TP used to leave the cache un-annotated, so SPMD handed the compiler the
full block pool with un-sharded num_kv_heads on every chip. The generation
tests cannot see that -- replication is functionally correct, just tp_size
times too big -- so assert it in the compiled IR: no graph may still take the
KV cache at its global [num_blocks, num_kv_heads, ...] shape, and at least one
must take it at [num_blocks, num_kv_heads/tp_size, ...].

Checking the live tensor's XLA sharding spec does NOT work: torch_xla's
propagation reports the heads dim as sharded either way, while the graph the
compiler actually receives still carries the full shape.

The runner's cache shapes and the IR dump are both process-local, so the
engine must run in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0); pytest re-execs
this file as a throwaway child so the once-only XLA init does not leak into
other vLLM tests.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

MODEL = "Qwen/Qwen3-0.6B"
WORKER_TIMEOUT = 1800


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
def test_kv_cache_head_sharded_under_dp_tp():
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

    print(stdout, flush=True)
    if stderr:
        print(stderr, file=sys.stderr, flush=True)
    assert rc == 0, (
        f"kv-sharding worker failed (exit={rc}) — the DP+TP KV cache reached "
        "the compiler un-sharded, or the worker timed out; see output above."
    )


def _find_model_runner(llm):
    core = llm.llm_engine.engine_core
    # InprocClient wraps the EngineCore; the MP client has no .engine_core.
    core = getattr(core, "engine_core", core)
    return core.model_executor.driver_worker.model_runner


def _check(export_dir: str) -> int:
    """Worker body (runs in the isolated child process). Returns exit code."""
    import vllm

    llm = vllm.LLM(
        model=MODEL,
        max_num_batched_tokens=64,
        max_num_seqs=2,
        max_model_len=32,
        gpu_memory_utilization=0.002,
        additional_config={
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "enable_data_parallel": True,
            "export_path": export_dir,
            "export_model_name": "kvshard",
        },
    )
    # Generate so the decode graphs are compiled and dumped too, and so a
    # failure here reads as "sharding broke generation".
    llm.generate(
        "Continue in English: I like taking walks in the",
        vllm.SamplingParams(temperature=0.0, max_tokens=4),
        use_tqdm=False,
    )

    runner = _find_model_runner(llm)
    mesh_shape = runner.mesh.shape()
    tp_size, dp_size = mesh_shape["model"], mesh_shape["batch"]
    print(f"[kv-sharding] mesh: dp={dp_size} tp={tp_size}", flush=True)
    if tp_size <= 1 or dp_size <= 1:
        print(
            f"[kv-sharding] FAIL: need a real DP+TP mesh, got dp={dp_size} "
            f"tp={tp_size} — this machine did not build a 2D mesh.",
            flush=True,
        )
        return 1

    # Global cache shape as the runner allocates it, before mark_sharding.
    entry = next(e for e in runner.kv_caches if isinstance(e, (list, tuple)))
    num_blocks, num_heads, block_size, head_size = entry[0].shape
    if num_heads % tp_size:
        print(
            f"[kv-sharding] FAIL: num_kv_heads {num_heads} not divisible by "
            f"tp_size {tp_size} — pick a model/mesh where it is.",
            flush=True,
        )
        return 1
    tail = f"x{block_size}x{head_size}x"
    replicated = f"tensor<{num_blocks}x{num_heads}{tail}"
    sharded = f"tensor<{num_blocks}x{num_heads // tp_size}{tail}"
    print(
        f"[kv-sharding] cache {tuple(entry[0].shape)}; want {sharded}, "
        f"reject {replicated}",
        flush=True,
    )

    graphs = sorted(Path(export_dir).glob("irs/ttir_*.mlir"))
    if not graphs:
        print(f"[kv-sharding] FAIL: no TTIR dumped under {export_dir}/irs.")
        return 1

    bad = [g.name for g in graphs if replicated in g.read_text()]
    good = [g.name for g in graphs if sharded in g.read_text()]
    print(f"[kv-sharding] {len(graphs)} graphs: {len(good)} sharded, {len(bad)} not")
    if bad:
        for name in bad[:8]:
            print(f"[kv-sharding] FAIL: {name} takes the cache at {replicated}")
        return 1
    if not good:
        print(f"[kv-sharding] FAIL: no graph takes the cache at {sharded}.")
        return 1
    print(f"[kv-sharding] PASS: heads sharded {tp_size}x on the model axis")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="kvshard-ir-") as d:
        sys.exit(_check(d))
