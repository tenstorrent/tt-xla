# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""``TTIndexer`` scoring, index selection, and indexer-K-cache management.

Exercises ``_forward_prefill`` / ``_forward_decode`` directly with pre-projected
q/k/gate tensors. ``TTIndexer.__init__`` builds vLLM parallel-linear submodules
that need a distributed environment, and ``_project`` is a faithful copy of
upstream's projection; the TT-specific logic worth pinning is everything *after*
it — the DSA op calls, the causal bound, the architecture-dependent index repair,
and the paged cache writes.

Selected indices are compared **as sets per row**, never positionally: bf16 scores
tie constantly, so index order is not reproducible between CPU and device.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr

# Registers the tt:: custom ops.
import tt_torch.custom_ops  # noqa: F401
from conftest import BLOCK_SIZE, DEEPSEEK_V32_CFG, gather_cache, maybe_mesh

from tests.utils import parametrize_arch

DTYPE = torch.bfloat16


def _make_indexer(topk_tokens, dsa_mode="auto", cfg=DEEPSEEK_V32_CFG):
    """A TTIndexer with only the attributes the tested methods read.

    Bypasses ``__init__`` (which constructs vLLM parallel linears requiring a
    distributed env) and sets the fields ``_forward_prefill`` / ``_forward_decode``
    actually depend on — so the test states those dependencies explicitly.
    """
    from tt_torch.custom_ops import dsa_kernels_available
    from vllm_tt.layers.dsa_indexer import TTIndexer, _TopKSlot

    ix = object.__new__(TTIndexer)
    ix.topk_tokens = topk_tokens
    ix.n_head = cfg["index_n_heads"]
    ix.head_dim = cfg["index_head_dim"]
    ix.rope_dim = cfg["qk_rope_head_dim"]
    ix.softmax_scale = cfg["index_head_dim"] ** -0.5
    ix._block_size = BLOCK_SIZE
    ix.dsa_mode = dsa_mode
    ix.k_chunk_size = TTIndexer._pick_k_chunk_size(topk_tokens)
    ix._slot = _TopKSlot()
    # Resolved at construction time in the real __init__; see the note there.
    ix._kernels_available = dsa_kernels_available()
    return ix


def _indexer_activations(users, seq_len, device, cfg=DEEPSEEK_V32_CFG, seed=0):
    """Pre-projected indexer q / k / gate weights in the DSA op layouts.

    ``k_op`` is assembled with an on-device ``cat``, mirroring ``_project``'s
    ``cat([k_pe, k_nope])``. That is not cosmetic: ``ttir.paged_update_cache`` fails
    to lower (TTIRToTTNNCommon, "Error code: 13") when its ``fill_value`` is a bare
    graph *input*, and compiles when it is a computed value. The production path
    always computes it (projection -> k_norm -> rope -> cat -> transpose), so
    building it the same way here keeps the test faithful instead of tripping a
    lowering quirk the real model never hits.
    """
    g = torch.Generator().manual_seed(seed)
    hi, d = cfg["index_n_heads"], cfg["index_head_dim"]
    rope_dim = cfg["qk_rope_head_dim"]
    q_op = torch.randn(users, hi, seq_len, d, dtype=DTYPE, generator=g).to(device)
    k_pe = torch.randn(users, 1, seq_len, rope_dim, dtype=DTYPE, generator=g).to(device)
    k_nope = torch.randn(users, 1, seq_len, d - rope_dim, dtype=DTYPE, generator=g).to(
        device
    )
    k_op = torch.cat([k_pe, k_nope], dim=-1)
    w_op = torch.randn(users, hi, seq_len, 1, dtype=DTYPE, generator=g).to(device)
    return q_op, k_op, w_op


def _metadata(users, cache_position, page_table):
    from vllm_tt.attention_impls.attention import TTMetadata

    return TTMetadata(
        cache_position=cache_position,
        attn_mask=None,
        page_table=page_table,
        is_causal=True,
        fill_page_table=page_table,
    )


def _row_sets(indices):
    """Per-row sets of valid (non-sentinel) indices."""
    signed = indices.cpu().to(torch.int32)
    flat = signed.reshape(-1, signed.shape[-1])
    return [{int(v) for v in row.tolist() if v >= 0} for row in flat]


# --------------------------------------------------------------------------- #
# Prefill
# --------------------------------------------------------------------------- #
@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize("users, seq_len, topk", [(1, 128, 128), (2, 64, 32)])
def test_indexer_prefill_selects_causal_keys(users, seq_len, topk, arch):
    """Selected keys must be causal, sentinel-tailed, and match a CPU golden."""
    xr.set_device_type("TT")
    maybe_mesh(arch, DEEPSEEK_V32_CFG["index_n_heads"])
    device = torch_xla.device()

    blocks_per_user = seq_len // BLOCK_SIZE
    num_blocks = users * blocks_per_user
    d = DEEPSEEK_V32_CFG["index_head_dim"]
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(
        users, blocks_per_user
    )

    def run(dev):
        ix = _make_indexer(topk)
        q_op, k_op, w_op = _indexer_activations(users, seq_len, dev)
        cache = torch.zeros(num_blocks, 1, BLOCK_SIZE, d, dtype=DTYPE, device=dev)
        md = _metadata(
            users,
            torch.full((users,), seq_len - 1, dtype=torch.int32, device=dev),
            page_table.to(dev),
        )
        idx = ix._forward_prefill(q_op, k_op, w_op, cache, md, users, seq_len)
        return idx, cache, k_op

    golden_idx, _, _ = run(torch.device("cpu"))
    device_idx, device_cache, device_k = run(device)
    torch_xla.sync()

    assert golden_idx is not None, "seq_len >= topk must take the sparse path"
    assert device_idx.shape == golden_idx.shape == (users, 1, seq_len, topk)

    # Causality + the contiguous-sentinel-tail contract sparse_sdpa requires.
    signed = device_idx.cpu().to(torch.int32)
    for u in range(users):
        for s in range(seq_len):
            row = signed[u, 0, s].tolist()
            assert all(
                v <= s for v in row if v >= 0
            ), f"user {u} row {s} selected a future key: {row}"
            sentinels = [i for i, v in enumerate(row) if v < 0]
            if sentinels:
                assert sentinels == list(
                    range(sentinels[0], topk)
                ), f"user {u} row {s} sentinels not a contiguous tail"
            assert any(v >= 0 for v in row), f"user {u} row {s} selected nothing"

    # Rows where every visible key fits in topk have a determined answer.
    dev_sets, gold_sets = _row_sets(device_idx), _row_sets(golden_idx)
    for i, (dev_set, gold_set) in enumerate(zip(dev_sets, gold_sets)):
        s = i % seq_len
        if s + 1 <= topk:
            assert dev_set == set(range(s + 1)) == gold_set, (
                f"row {s}: every visible key must be selected when it fits "
                f"(got {sorted(dev_set)})"
            )

    # The indexer K cache must hold this chunk's keys in logical order.
    expected = device_k.cpu().transpose(1, 2).reshape(users, seq_len, d)
    assert torch.allclose(
        gather_cache(device_cache.cpu(), page_table, seq_len), expected
    ), "indexer K cache filled incorrectly"


@pytest.mark.push
@pytest.mark.single_device
def test_indexer_prefill_falls_back_to_dense_below_topk():
    """``seq_len < topk`` must publish None (dense) but still fill the cache."""
    users, seq_len, topk = 1, 64, 2048
    d = DEEPSEEK_V32_CFG["index_head_dim"]
    blocks_per_user = seq_len // BLOCK_SIZE
    page_table = torch.arange(blocks_per_user, dtype=torch.int32).view(1, -1)
    q_op, k_op, w_op = _indexer_activations(users, seq_len, torch.device("cpu"))

    ix = _make_indexer(topk)
    cache = torch.zeros(blocks_per_user, 1, BLOCK_SIZE, d, dtype=DTYPE)
    md = _metadata(
        users, torch.full((users,), seq_len - 1, dtype=torch.int32), page_table
    )
    idx = ix._forward_prefill(q_op, k_op, w_op, cache, md, users, seq_len)

    assert idx is None, "seq_len < topk must fall back to dense attention"
    # The cache is still written -- later decodes depend on it.
    expected = k_op.transpose(1, 2).reshape(users, seq_len, d)
    assert torch.allclose(gather_cache(cache, page_table, seq_len), expected)


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #
@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize("users, cur_pos", [(1, 95), (2, 63)])
def test_indexer_decode_bounds_selection_by_cur_pos(users, cur_pos, arch):
    """Decode must respect the *runtime* per-user causal bound.

    ``tt.indexer_score_dsa``'s ``chunk_start_idx`` is a compile-time attribute, so
    with one query token the real bound (``cache_position``) is applied as an
    explicit additive mask. Positions past it hold prefill padding, so a selected
    index above ``cur_pos`` would mean attending to garbage.
    """
    xr.set_device_type("TT")
    maybe_mesh(arch, DEEPSEEK_V32_CFG["index_n_heads"])
    device = torch_xla.device()

    blocks_per_user = cur_pos // BLOCK_SIZE + 1
    num_blocks = users * blocks_per_user
    max_seq_len = blocks_per_user * BLOCK_SIZE
    topk = 32
    d = DEEPSEEK_V32_CFG["index_head_dim"]
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(
        users, blocks_per_user
    )
    # Pre-seed the whole cache, including the padding past cur_pos.
    seeded = torch.randn(num_blocks, 1, BLOCK_SIZE, d, dtype=DTYPE)

    def run(dev):
        ix = _make_indexer(topk)
        q_op, k_op, w_op = _indexer_activations(users, 1, dev)
        cache = seeded.clone().to(dev)
        md = _metadata(
            users,
            torch.full((users,), cur_pos, dtype=torch.int32, device=dev),
            page_table.to(dev),
        )
        idx = ix._forward_decode(q_op, k_op, w_op, cache, md, users)
        return idx, cache, k_op

    golden_idx, _, _ = run(torch.device("cpu"))
    device_idx, device_cache, device_k = run(device)
    torch_xla.sync()

    assert golden_idx is not None, "max_seq_len > topk must take the sparse path"
    assert device_idx.shape == golden_idx.shape == (users, 1, 1, topk)

    signed = device_idx.cpu().to(torch.int32)
    for u in range(users):
        row = signed[u, 0, 0].tolist()
        assert all(
            v <= cur_pos for v in row if v >= 0
        ), f"user {u} selected a padding/future key beyond cur_pos={cur_pos}: {row}"
        assert any(v >= 0 for v in row), f"user {u} selected nothing"
        # cur_pos+1 visible keys >= topk here, so no sentinels are expected.
        assert len({v for v in row if v >= 0}) == topk, "expected topk distinct keys"

    # The new token's indexer K lands at cur_pos; the prefix is untouched.
    after = gather_cache(device_cache.cpu(), page_table, cur_pos + 1)
    before = gather_cache(seeded, page_table, cur_pos + 1)
    assert torch.allclose(
        after[:, cur_pos, :], device_k.cpu().reshape(users, d)
    ), "decode did not write the indexer K at cache_position"
    assert torch.allclose(
        after[:, :cur_pos, :], before[:, :cur_pos, :]
    ), "decode clobbered the cached indexer prefix"


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "topk, dsa_mode, expect_sparse",
    [
        # Bucket cannot exceed topk -> dense decode is exactly equivalent, so no
        # indices are published even in the default mode.
        (2048, "auto", False),
        # Bucket exceeds topk -> the default must go sparse, so that only the
        # selected top-k participate in the main MLA attention.
        (32, "auto", True),
        # Explicit opt-out back to dense decode (perf A/B only).
        (32, "dense_decode", False),
    ],
)
def test_indexer_decode_mode_gating(topk, dsa_mode, expect_sparse):
    """Decode publishes indices whenever sparsity is *needed*, by default.

    The published indices are what restrict the main MLA attention to the top-k, so
    the default must not silently fall back to dense once the context can exceed
    index_topk -- that would let every cached entry participate.
    """
    users, cur_pos = 1, 95
    d = DEEPSEEK_V32_CFG["index_head_dim"]
    blocks_per_user = cur_pos // BLOCK_SIZE + 1
    page_table = torch.arange(blocks_per_user, dtype=torch.int32).view(1, -1)
    q_op, k_op, w_op = _indexer_activations(users, 1, torch.device("cpu"))

    ix = _make_indexer(topk, dsa_mode=dsa_mode)
    cache = torch.randn(blocks_per_user, 1, BLOCK_SIZE, d, dtype=DTYPE)
    md = _metadata(users, torch.full((users,), cur_pos, dtype=torch.int32), page_table)
    idx = ix._forward_decode(q_op, k_op, w_op, cache, md, users)

    assert (idx is not None) == expect_sparse
    # Either way the cache must carry the new token, or later steps read stale K.
    assert torch.allclose(
        gather_cache(cache, page_table, cur_pos + 1)[:, cur_pos, :],
        k_op.reshape(users, d),
    )
