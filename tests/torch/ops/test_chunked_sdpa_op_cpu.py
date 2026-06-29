# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for tt::chunked_scaled_dot_product_attention (tt-xla #4986).

Pin the op's CPU reference branch (the device test's oracle) against an
independent offset-causal SDPA over the dense gathered prefix.
"""

import pytest
import torch
import torch.nn.functional as F

# Registers tt:: custom ops (including chunked_scaled_dot_product_attention).
import tt_torch.custom_ops  # noqa: F401


def _make_paged_cache(num_blocks, n_kv, block_size, head, seed):
    g = torch.Generator().manual_seed(seed)
    key = torch.randn(num_blocks, n_kv, block_size, head, generator=g)
    value = torch.randn(num_blocks, n_kv, block_size, head, generator=g)
    return key, value


def _reference(query, key, value, page_table, chunk_start, scale):
    """Independent oracle: gather dense prefix+chunk, run offset-causal SDPA."""
    users, n_heads, chunk_len, head = query.shape
    n_kv, block_size = key.shape[1], key.shape[2]
    nbpu = page_table.shape[1]
    s_len = nbpu * block_size

    idx = page_table.reshape(-1)
    gk = (
        key.index_select(0, idx)
        .view(users, nbpu, n_kv, block_size, head)
        .permute(0, 2, 1, 3, 4)
        .reshape(users, n_kv, s_len, head)
    )
    gv = (
        value.index_select(0, idx)
        .view(users, nbpu, n_kv, block_size, head)
        .permute(0, 2, 1, 3, 4)
        .reshape(users, n_kv, s_len, head)
    )
    if n_kv != n_heads:  # GQA broadcast
        rep = n_heads // n_kv
        gk = gk.repeat_interleave(rep, dim=1)
        gv = gv.repeat_interleave(rep, dim=1)

    # query row i is absolute position chunk_start + i; attends key j iff
    # j <= chunk_start + i.
    q_pos = chunk_start + torch.arange(chunk_len).view(chunk_len, 1)
    k_pos = torch.arange(s_len).view(1, s_len)
    mask = (k_pos <= q_pos).view(1, 1, chunk_len, s_len)
    return F.scaled_dot_product_attention(query, gk, gv, attn_mask=mask, scale=scale)


@pytest.mark.push
@pytest.mark.cpu
@pytest.mark.parametrize(
    "n_heads,n_kv", [(8, 8), (8, 2), (32, 8)], ids=["mha", "gqa-4x", "gqa-4x-32h"]
)
@pytest.mark.parametrize(
    "chunk_start", [0, 64, 96], ids=["start0", "start64", "start96"]
)
def test_chunked_sdpa_cpu_matches_reference(n_heads, n_kv, chunk_start):
    """The op's CPU branch must match offset-causal SDPA over the gathered prefix.

    chunk_start=0 is the first/no-prefix chunk; chunk_start>0 is a continuation
    chunk attending over a cached prefix -- the path the device op exercises.
    """
    torch.manual_seed(0)
    head, block_size, nbpu = 64, 32, 4
    s_len = nbpu * block_size  # 128
    chunk_len = s_len - chunk_start  # chunk fills the rest of the context
    users = 2
    num_blocks = users * nbpu

    key, value = _make_paged_cache(num_blocks, n_kv, block_size, head, seed=1)
    # Distinct (non-overlapping) blocks per user.
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(users, nbpu)
    query = torch.randn(users, n_heads, chunk_len, head)
    scale = 1.0 / head**0.5

    start_t = torch.tensor([chunk_start], dtype=torch.int32)
    out = torch.ops.tt.chunked_scaled_dot_product_attention(
        query, key, value, page_table, start_t, scale=scale
    )
    ref = _reference(query, key, value, page_table, chunk_start, scale)

    assert out.shape == query.shape
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.push
@pytest.mark.cpu
def test_chunked_sdpa_cpu_single_token_remainder():
    """A 1-token chunk attending over a large cached prefix must be correct.

    Simulates the final step of a chunked prefill where ISL = chunk_size + 1:
    the first step processed chunk_size tokens (chunk_start=0, chunk_len=64),
    and this step processes the 1-token remainder (chunk_start=64, chunk_len=1).
    """
    torch.manual_seed(0)
    users, n_heads, n_kv, head, block_size, nbpu = 2, 8, 2, 64, 32, 4
    s_len = nbpu * block_size  # 128

    chunk_start = 64  # large prefix already cached
    chunk_len = 1  # single-token remainder

    key, value = _make_paged_cache(users * nbpu, n_kv, block_size, head, seed=3)
    page_table = torch.arange(users * nbpu, dtype=torch.int32).view(users, nbpu)
    query = torch.randn(users, n_heads, chunk_len, head)
    scale = 1.0 / head**0.5

    start_t = torch.tensor([chunk_start], dtype=torch.int32)
    out = torch.ops.tt.chunked_scaled_dot_product_attention(
        query, key, value, page_table, start_t, scale=scale
    )
    ref = _reference(query, key, value, page_table, chunk_start, scale)

    assert out.shape == query.shape
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.push
@pytest.mark.cpu
def test_chunked_sdpa_cpu_start0_is_plain_causal():
    """At chunk_start=0 the op must equal standard causal SDPA over the chunk."""
    torch.manual_seed(0)
    users, n_heads, head, block_size, nbpu = 1, 4, 64, 32, 2
    s_len = nbpu * block_size
    key, value = _make_paged_cache(users * nbpu, n_heads, block_size, head, seed=2)
    page_table = torch.arange(users * nbpu, dtype=torch.int32).view(users, nbpu)
    query = torch.randn(users, n_heads, s_len, head)  # full chunk, start=0
    scale = 1.0 / head**0.5

    out = torch.ops.tt.chunked_scaled_dot_product_attention(
        query, key, value, page_table, torch.zeros(1, dtype=torch.int32), scale=scale
    )
    dense_k = key.permute(1, 0, 2, 3).reshape(1, n_heads, s_len, head)
    dense_v = value.permute(1, 0, 2, 3).reshape(1, n_heads, s_len, head)
    ref = F.scaled_dot_product_attention(
        query, dense_k, dense_v, is_causal=True, scale=scale
    )

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
