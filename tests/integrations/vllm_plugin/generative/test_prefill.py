# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill-graph tests + DP+TP chunked-prefill hang reproducer.

Companion to ``test_decode.py`` (which pins the decode-only graph). Here we
exercise the *prefill* graphs. Constructing ``vllm.LLM`` triggers warmup
(``capture_model``), which compiles + traces every prefill bucket, so a bad
prefill graph fails at construction before any ``generate``.

The DP+TP block reproduces a CCL hang seen on the BH galaxy (mesh [4, 8],
DP=4/TP=8) running Devstral-2-123B: with ``prefill_chunk_size`` set (chunked
prefill on -> the cached-prefix / chunked-SDPA path, ``_chunked_sdpa_active``),
warmup hangs in a collective op:

  - trace on:  hang at ``ttnn.end_trace_capture`` of the num_tokens=128
    cached-prefix prefill graph, just after ``reduce_scatter`` + ``all_gather``.
  - trace off: hang at ``ttnn.all_gather`` (cluster_axis=0, the DP/batch axis)
    in the num_tokens=64 bucket.

Both surface as the tt-metal 60s device timeout
("waiting for physical cores to finish: 15-3, 15-2"), which raises TT_THROW ->
the test fails (it does not hang forever). The same config *without*
``prefill_chunk_size`` (plain prefill, buckets up to max_model_len) runs fine,
so the parametrization below bisects chunked-on vs -off (and trace on vs off) to
pin the hang to the chunked-SDPA CCL path.
"""

import gc

import pytest
import vllm


def _shutdown(llm) -> None:
    """Release the TT device before the next engine is built.

    A bare ``del llm`` defers teardown to weakref finalize, so the EngineCore
    subprocess can still hold the device when the next ``vllm.LLM`` starts and
    deadlock it. Shut the engine core down explicitly instead. (Same idiom as
    ``test_chunked_prefill.py`` / ``sampling/conftest.py``.)
    """
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()


# A prompt comfortably longer than the chunk budget (128) so prefill is split
# into several chunks and the cached-prefix chunked-SDPA path (chunks 2..N) is
# actually exercised -- a one-chunk prompt would skip it entirely.
_PARA = (
    "The history of computing spans many centuries. Early mechanical "
    "calculators gave way to electromechanical machines, and then to the "
    "electronic digital computers that define the modern era. Each generation "
    "brought dramatic improvements in speed, reliability, and cost. "
)
_LONG_PROMPT = ("Summarize the following text.\n\n" + (_PARA * 12) + "\nSummary:")


@pytest.mark.nightly
@pytest.mark.single_device
def test_prefill_single_device():
    """Single-chip prefill sanity: compile + run a multi-token prefill graph.

    The decode counterpart is ``test_decode.py``. A multi-token prompt with
    ``max_tokens=1`` runs one prefill step (plus a single sampled token), so the
    prefill bucket graphs compile at warmup and execute once. No CCLs here (one
    device) -- this isolates a plain-prefill regression from the DP+TP CCL path.
    """
    llm_args = {
        "model": "facebook/opt-125m",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 64,
        "gpu_memory_utilization": 0.05,
        "additional_config": {
            "enable_const_eval": True,
            "min_context_len": 32,
            "num_hidden_layers": 1,
        },
    }
    llm = vllm.LLM(**llm_args)
    try:
        out = llm.generate(
            ["The capital of France is"],
            vllm.SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        )
        assert out[0].outputs[0].token_ids, "prefill produced no token"
    finally:
        _shutdown(llm)


@pytest.mark.nightly
@pytest.mark.single_device
def test_prefill_single_device_chunked():
    """Single-chip *chunked* prefill sanity (control for the DP+TP repro).

    Same chunked-SDPA path (``prefill_chunk_size`` < prompt length) but on one
    device, so no CCLs. This is expected to PASS -- if the DP+TP repro hangs but
    this passes, the fault is in the collective ops, not the chunked-SDPA op
    itself.
    """
    llm_args = {
        "model": "facebook/opt-125m",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.10,
        "enable_prefix_caching": False,
        "additional_config": {
            "enable_const_eval": True,
            "min_context_len": 32,
            "num_hidden_layers": 2,
            "prefill_chunk_size": 128,
        },
    }
    llm = vllm.LLM(**llm_args)
    try:
        out = llm.generate(
            [_LONG_PROMPT],
            vllm.SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        )
        assert out[0].outputs[0].token_ids, "chunked prefill produced no token"
    finally:
        _shutdown(llm)


# (model, mesh_shape) targets. Devstral 4x8 is the config that hangs; Qwen3-32B
# 8x4 is the sibling DP+TP config. num_hidden_layers=2 keeps compile fast -- the
# hang is in the CCL/chunked-SDPA path, not model depth.
_DPTP_TARGETS = [
    pytest.param(
        "mistralai/Devstral-2-123B-Instruct-2512",
        [4, 8],
        id="devstral-123b-4x8",
    ),
    pytest.param("Qwen/Qwen3-32B", [8, 4], id="qwen3-32b-8x4"),
]


@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.bh_galaxy
@pytest.mark.parametrize("model_name, mesh_shape", _DPTP_TARGETS)
@pytest.mark.parametrize("enable_trace", [False, True], ids=["trace-off", "trace-on"])
@pytest.mark.parametrize(
    "prefill_chunk_size",
    [0, 128],
    ids=["chunked-off", "chunked-on"],
)
def test_prefill_dptp_chunked_repro(
    model_name: str,
    mesh_shape: list[int],
    enable_trace: bool,
    prefill_chunk_size: int,
):
    """DP+TP prefill CCL-hang bisection (BH galaxy).

    4-cell matrix per model = {chunked-off, chunked-on} x {trace-off, trace-on}.
    Expectation from the field logs: ``chunked-off`` passes (plain prefill),
    ``chunked-on`` hangs in a collective (cached-prefix chunked-SDPA path). If
    that holds, the culprit is a CCL emitted only on the chunked path -- not the
    model, batch size, or trace alone.

    Constructing the engine runs warmup, which compiles + traces the prefill
    buckets; that is where the hang fires (before generate). A small generate
    follows to also exercise runtime prefill execution when warmup survives.
    """
    additional_config = {
        "enable_tensor_parallel": True,
        "enable_data_parallel": True,
        "shard_weights_on_batch_axis": False,
        "use_2d_mesh": True,
        "mesh_shape": mesh_shape,
        "experimental_weight_dtype": "bfp_bf8",
        "experimental_kv_cache_dtype": "bfp_bf8",
        "enable_const_eval": True,
        "optimization_level": 1,
        "enable_trace": enable_trace,
        # min_context_len=32 gives the [1, 32, 64, 128] prefill token ladder from
        # the failing run (the trace-off hang was in the num_tokens=64 bucket).
        "min_context_len": 32,
        "num_hidden_layers": 2,
        # cpu_sampling: device sampling on the 2D mesh is blocked (#4387/#4440),
        # and it is orthogonal to the prefill CCL hang under test.
        "cpu_sampling": True,
    }
    if prefill_chunk_size:
        additional_config["prefill_chunk_size"] = prefill_chunk_size

    # max_num_seqs kept modest (DP=first-dim -> 2 seqs/replica) so warmup is
    # quick; the DP all_gather and TP reduce_scatter/all_gather are present
    # regardless of batch size. max_num_batched_tokens is auto-derived under
    # chunked prefill; for the chunked-off cell it must cover max_model_len.
    max_num_seqs = 8
    max_model_len = 1024
    max_num_batched_tokens = (
        prefill_chunk_size * max_num_seqs
        if prefill_chunk_size
        else max_model_len * max_num_seqs
    )

    llm = vllm.LLM(
        model=model_name,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=0.3,
        enable_prefix_caching=False,
        additional_config=additional_config,
    )
    try:
        out = llm.generate(
            [_LONG_PROMPT] * max_num_seqs,
            vllm.SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        )
        assert len(out) == max_num_seqs
        for o in out:
            assert o.outputs[0].token_ids, "DP+TP prefill produced no token"
    finally:
        _shutdown(llm)


@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.parametrize("enable_trace", [False, True], ids=["trace-off", "trace-on"])
@pytest.mark.parametrize(
    "prefill_chunk_size",
    [0, 128],
    ids=["chunked-off", "chunked-on"],
)
def test_prefill_dptp_chunked_smallmesh(
    enable_trace: bool,
    prefill_chunk_size: int,
):
    """8-chip carve-out of the DP+TP chunked-prefill CCL hang (scale isolation).

    Companion to ``test_prefill_dptp_chunked_repro`` (galaxy 32-chip, mesh 4x8 /
    8x4) and to ``test_dptp_devstral``. Here the mesh is [2, 4] (DP=2, TP=4) on
    8 chips carved out of the BH galaxy via
    ``TT_VISIBLE_DEVICES=0,4,8,12,16,20,24,28``. Purpose: determine whether the
    ``test_dptp_devstral`` hang at ``ttnn.end_trace_capture`` of the fused TP
    all_reduce (cluster_axis=1) is specific to the 32-chip scale or reproduces
    on a smaller 2D mesh.

    Every compile/trace knob is copied verbatim from
    ``test_prefill_dptp_chunked_repro`` so the compiled graph -- and thus the
    cluster_axis=1 (TP) all_reduce from the RowParallel o_proj/down_proj
    reductions, plus the cluster_axis=0 (DP) collectives -- matches the galaxy
    graph. Only the model (Qwen3-0.6B, proven to shard on TP=4 by
    ``test_data_tensor_parallel_generation_push``) and the mesh ([2, 4]) change.
    The TP=4-vs-8 chip count IS the scale variable under test.

    NOT marked ``nightly`` on purpose: the chunked-on/trace-on cell is *designed*
    to hang, and the hang only converts into a fast failure when
    ``TT_METAL_OPERATION_TIMEOUT_SECONDS`` is set in the environment. Launch it
    explicitly by path/-k with that var set; it must not run in an untimed lane
    where it could wedge the machine.
    """
    additional_config = {
        "enable_tensor_parallel": True,
        "enable_data_parallel": True,
        "shard_weights_on_batch_axis": False,
        "use_2d_mesh": True,
        "mesh_shape": [2, 4],
        "experimental_weight_dtype": "bfp_bf8",
        "experimental_kv_cache_dtype": "bfp_bf8",
        "enable_const_eval": True,
        "optimization_level": 1,
        "enable_trace": enable_trace,
        # min_context_len=32 gives the [1, 32, 64, 128] prefill token ladder,
        # matching the failing galaxy run.
        "min_context_len": 32,
        "num_hidden_layers": 2,
        # cpu_sampling: device sampling on the 2D mesh is blocked (#4387/#4440)
        # and is orthogonal to the prefill CCL hang under test.
        "cpu_sampling": True,
    }
    if prefill_chunk_size:
        additional_config["prefill_chunk_size"] = prefill_chunk_size

    # DP=first-dim=2 -> 4 seqs/replica. The DP all_gather and TP
    # reduce_scatter/all_gather (or fused all_reduce) are present regardless of
    # batch size. max_num_batched_tokens is auto-derived under chunked prefill;
    # for the chunked-off cell it must cover max_model_len.
    max_num_seqs = 8
    max_model_len = 1024
    max_num_batched_tokens = (
        prefill_chunk_size * max_num_seqs
        if prefill_chunk_size
        else max_model_len * max_num_seqs
    )

    llm = vllm.LLM(
        model="Qwen/Qwen3-0.6B",
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=0.3,
        enable_prefix_caching=False,
        additional_config=additional_config,
    )
    try:
        out = llm.generate(
            [_LONG_PROMPT] * max_num_seqs,
            vllm.SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        )
        assert len(out) == max_num_seqs
        for o in out:
            assert o.outputs[0].token_ids, "DP+TP prefill produced no token"
    finally:
        _shutdown(llm)
