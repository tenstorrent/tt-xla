# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla.core.xla_model as xm
from infra.utilities.types import Framework

from tests.infra.testers.single_chip.op.op_tester import OpTester, run_op_test

# TODO: Record superset properties for these tests.


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("num_heads", [12, 16])
@pytest.mark.parametrize("max_seq_len", [64, 128])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("seq_len_to_fill", [32, 64])
def test_fill_cache(num_heads, max_seq_len, head_size, seq_len_to_fill):

    cache = torch.zeros(1, num_heads, max_seq_len, head_size, dtype=torch.bfloat16)
    fill_value = torch.randn(
        1, num_heads, seq_len_to_fill, head_size, dtype=torch.bfloat16
    )

    run_op_test(
        torch.ops.tt.fill_cache, [cache, fill_value, 0], framework=Framework.TORCH
    )


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("num_heads", [12, 16])
@pytest.mark.parametrize("max_seq_len", [64, 128])
@pytest.mark.parametrize("head_size", [64, 128])
def test_update_cache(num_heads, max_seq_len, head_size):

    cache = torch.zeros(1, num_heads, max_seq_len, head_size, dtype=torch.bfloat16)
    fill_value = torch.randn(1, num_heads, 1, head_size, dtype=torch.bfloat16)

    cache_position = torch.tensor([10], dtype=torch.int32)

    run_op_test(
        torch.ops.tt.update_cache,
        [cache, fill_value, cache_position, 0],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_size, num_kv_heads, max_seq_len, is_causal, scale, sliding_window_size",
    [
        (1, 12, 32, 128, 12, 32, True, 1.0, 16),
        (1, 12, 32, 128, 12, 128, False, 1.0, 16),
        (8, 12, 32, 128, 12, 32, True, 1.0, 32),
        (8, 12, 32, 128, 12, 128, False, 1.0, 32),
        (1, 12, 32, 128, 4, 32, True, 1.0, 16),
        (1, 12, 32, 128, 4, 128, False, 1.0, 16),
        (8, 12, 32, 128, 4, 32, True, 1.0, 32),
        (8, 12, 32, 128, 4, 128, False, 1.0, 32),
    ],
)
def test_scaled_dot_product_attention(
    batch_size,
    num_heads,
    seq_len,
    head_size,
    num_kv_heads,
    max_seq_len,
    is_causal,
    scale,
    sliding_window_size,
):

    query = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=torch.bfloat16)
    key = torch.randn(
        batch_size, num_kv_heads, max_seq_len, head_size, dtype=torch.bfloat16
    )
    value = torch.randn(
        batch_size, num_kv_heads, max_seq_len, head_size, dtype=torch.bfloat16
    )
    attn_mask = (
        torch.randn(batch_size, 1, seq_len, max_seq_len, dtype=torch.bfloat16)
        if not is_causal
        else None
    )

    run_op_test(
        torch.ops.tt.scaled_dot_product_attention,
        [query, key, value, attn_mask, is_causal, scale, sliding_window_size],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "batch_size, num_heads, head_size, num_kv_heads, max_seq_len, is_causal, scale",
    [
        (1, 12, 128, 12, 32, True, 1.0),
        (1, 12, 128, 12, 32, False, 1.0),
        (1, 12, 128, 12, 128, False, 1.0),
        (8, 12, 128, 12, 32, True, 1.0),
        (8, 12, 128, 12, 128, False, 1.0),
        (1, 12, 128, 4, 32, True, 1.0),
        (1, 12, 128, 4, 32, False, 1.0),
        (1, 12, 128, 4, 128, False, 1.0),
        (8, 12, 128, 4, 32, True, 1.0),
        (8, 12, 128, 4, 128, False, 1.0),
    ],
)
def test_scaled_dot_product_attention_decode(
    batch_size, num_heads, head_size, num_kv_heads, max_seq_len, is_causal, scale
):

    query = torch.randn(1, batch_size, num_heads, head_size, dtype=torch.bfloat16)
    key = torch.randn(
        batch_size, num_kv_heads, max_seq_len, head_size, dtype=torch.bfloat16
    )
    value = torch.randn(
        batch_size, num_kv_heads, max_seq_len, head_size, dtype=torch.bfloat16
    )
    cur_pos_tensor = torch.arange(batch_size, dtype=torch.int32)
    attn_mask = (
        torch.randn(batch_size, 1, num_heads, max_seq_len, dtype=torch.bfloat16)
        if not is_causal
        else None
    )

    run_op_test(
        torch.ops.tt.scaled_dot_product_attention_decode,
        [query, key, value, cur_pos_tensor, attn_mask, None, is_causal, scale],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("num_users", [8, 16, 24, 32])
@pytest.mark.parametrize("max_num_blocks_per_seq", [16, 32])
@pytest.mark.parametrize("num_heads", [1, 8, 32])
@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_paged_update_cache(
    num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim
):
    max_num_blocks = max_num_blocks_per_seq * num_users
    max_seq_len = max_num_blocks_per_seq * block_size

    cache = torch.zeros(
        max_num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    fill_value = torch.randn(1, num_users, num_heads, head_dim, dtype=torch.bfloat16)

    # Create arbitrary update indices
    cache_idxs = torch.randperm(max_seq_len)[:num_users]
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(num_users, max_num_blocks_per_seq).to(
        torch.int32
    )

    run_op_test(
        torch.ops.tt.paged_update_cache,
        [cache, fill_value, cache_idxs, page_table, False],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("num_users", [8, 16])
@pytest.mark.parametrize("max_num_blocks_per_seq", [16, 32])
@pytest.mark.parametrize("num_heads", [1, 8, 32])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("seq_len_to_fill", [10, 20, 32, 50, 70])
def test_paged_fill_cache(
    num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim, seq_len_to_fill
):
    max_num_blocks = max_num_blocks_per_seq * num_users

    cache = torch.zeros(
        max_num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    fill_value = torch.randn(
        1, num_heads, seq_len_to_fill, head_dim, dtype=torch.bfloat16
    )

    # Create arbitrary page table
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(num_users, max_num_blocks_per_seq).to(
        torch.int32
    )

    batch_idx = torch.randint(0, num_users, (1,), dtype=torch.int32)

    run_op_test(
        torch.ops.tt.paged_fill_cache,
        [cache, fill_value, page_table, batch_idx],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("num_users", [8])
@pytest.mark.parametrize("max_num_blocks_per_seq", [16, 32])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("head_dim", [128])
def test_paged_scaled_dot_product_attention_decode(
    num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim
):
    max_num_blocks = max_num_blocks_per_seq * num_users

    query = torch.randn(1, num_users, num_heads, head_dim, dtype=torch.bfloat16)
    key = torch.randn(
        max_num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    value = torch.randn(
        max_num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    page_table = torch.ones(num_users, max_num_blocks_per_seq).to(torch.int32)
    cur_pos_tensor = torch.ones(num_users).to(torch.int32)

    run_op_test(
        torch.ops.tt.paged_scaled_dot_product_attention_decode,
        [query, key, value, page_table, True, None, cur_pos_tensor],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_size, num_kv_heads, head_dim_v, has_value, is_causal, scale",
    [
        # MLA-from-latent (value=None, head_dim_v < head_size, d_rope=64)
        (1, 16, 64, 192, 1, 128, False, True, 1.0),
        (1, 16, 64, 192, 1, 128, False, False, 1.0),
        (2, 16, 128, 192, 1, 128, False, True, 1.0),
        (2, 16, 128, 192, 1, 128, False, False, 1.0),
        # MLA-from-latent with d_rope=0 (head_dim_v == head_size)
        (1, 32, 64, 128, 1, 128, False, True, 1.0),
        (1, 32, 64, 128, 1, 128, False, False, 1.0),
        # Explicit value tensor path
        (1, 16, 64, 128, 1, 64, True, True, 1.0),
        (2, 16, 128, 128, 1, 64, True, False, 1.0),
    ],
)
def test_flash_mla_prefill(
    batch_size,
    num_heads,
    seq_len,
    head_size,
    num_kv_heads,
    head_dim_v,
    has_value,
    is_causal,
    scale,
):

    query = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=torch.bfloat16)
    key = torch.randn(
        batch_size, num_kv_heads, seq_len, head_size, dtype=torch.bfloat16
    )
    value = (
        torch.randn(batch_size, num_kv_heads, seq_len, head_dim_v, dtype=torch.bfloat16)
        if has_value
        else None
    )
    attn_mask = (
        torch.randn(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)
        if not is_causal
        else None
    )

    run_op_test(
        torch.ops.tt.flash_mla_prefill,
        [query, key, head_dim_v, value, attn_mask, is_causal, scale],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "num_users, num_heads, num_kv_heads, head_size, head_dim_v, has_value, block_size, max_num_blocks_per_seq, is_causal, scale",
    [
        # MLA-from-latent (value=None): single shared latent KV head, the Q/K
        # head dim carries the rope tail (dh_qk = head_dim_v + d_rope = 128 + 64).
        (8, 16, 1, 192, 128, False, 32, 16, True, 1.0),
        (8, 16, 1, 192, 128, False, 64, 16, True, 1.0),
        # MLA-from-latent with d_rope=0 (dh_qk == head_dim_v).
        (8, 32, 1, 128, 128, False, 32, 16, True, 1.0),
        # Explicit value tensor path (separate V cache) with a single latent KV
        # head broadcast across the query heads.
        (8, 16, 1, 128, 64, True, 32, 16, True, 1.0),
        # GQA latent: a few KV heads, each shared by a group of query heads.
        (8, 16, 4, 192, 128, False, 32, 16, True, 1.0),
        # MHA sanity (num_kv_heads == num_heads), explicit value.
        (8, 8, 8, 128, 128, True, 32, 16, True, 1.0),
    ],
)
def test_paged_flash_mla_decode(
    num_users,
    num_heads,
    num_kv_heads,
    head_size,
    head_dim_v,
    has_value,
    block_size,
    max_num_blocks_per_seq,
    is_causal,
    scale,
):
    max_num_blocks = max_num_blocks_per_seq * num_users
    max_seq_len = max_num_blocks_per_seq * block_size

    query = torch.randn(1, num_users, num_heads, head_size, dtype=torch.bfloat16)
    key = torch.randn(
        max_num_blocks, num_kv_heads, block_size, head_size, dtype=torch.bfloat16
    )
    value = (
        torch.randn(
            max_num_blocks, num_kv_heads, block_size, head_dim_v, dtype=torch.bfloat16
        )
        if has_value
        else None
    )

    # A valid page table: distinct, non-overlapping physical blocks per user.
    page_table = (
        torch.randperm(max_num_blocks)
        .reshape(num_users, max_num_blocks_per_seq)
        .to(torch.int32)
    )
    # Current decode position per user, kept within the cached range.
    cur_pos_tensor = torch.randint(0, max_seq_len, (num_users,), dtype=torch.int32)

    # args = [query, key, head_dim_v, page_table, value, is_causal,
    #         attn_mask, cur_pos_tensor, attention_sink, scale]
    run_op_test(
        torch.ops.tt.paged_flash_mla_decode,
        [
            query,
            key,
            head_dim_v,
            page_table,
            value,
            is_causal,
            None,
            cur_pos_tensor,
            None,
            scale,
        ],
        framework=Framework.TORCH,
    )
