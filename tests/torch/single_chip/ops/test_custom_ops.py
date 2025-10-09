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


@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_size, num_kv_heads, max_seq_len, is_causal, scale",
    [
        (1, 12, 32, 128, 12, 32, True, 1.0),
        (1, 12, 32, 128, 12, 128, False, 1.0),
        (8, 12, 32, 128, 12, 32, True, 1.0),
        (8, 12, 32, 128, 12, 128, False, 1.0),
        (1, 12, 32, 128, 4, 32, True, 1.0),
        (1, 12, 32, 128, 4, 128, False, 1.0),
        (8, 12, 32, 128, 4, 32, True, 1.0),
        (8, 12, 32, 128, 4, 128, False, 1.0),
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
        [query, key, value, attn_mask, is_causal, scale],
        framework=Framework.TORCH,
    )


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
