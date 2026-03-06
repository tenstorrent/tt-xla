# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
def test_split_query_key_value_and_split_heads_mha_matmul(request):
    """MHA pattern: equal Q/K/V weight shapes, matmul variant."""
    batch, seq_len, hidden_dim = 1, 32, 512
    num_heads, head_size = 8, 64

    def mha_qkv_matmul(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
    ) -> tuple:
        # Reshape input: [batch, seq_len, hidden_dim] -> [batch*seq_len, hidden_dim]
        x_2d = x.reshape(batch * seq_len, hidden_dim)
        # Project Q, K, V
        q = (
            torch.matmul(x_2d, wq)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        k = (
            torch.matmul(x_2d, wk)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        v = (
            torch.matmul(x_2d, wv)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        return q, k, v

    run_graph_test_with_random_inputs(
        mha_qkv_matmul,
        [
            (batch, seq_len, hidden_dim),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
        ],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
def test_split_query_key_value_and_split_heads_mha_matmul_with_bias(request):
    """MHA pattern: equal Q/K/V weight shapes, matmul + bias addition variant."""
    batch, seq_len, hidden_dim = 1, 32, 512
    num_heads, head_size = 8, 64

    def mha_qkv_linear(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
    ) -> tuple:
        x_2d = x.reshape(batch * seq_len, hidden_dim)
        q = (
            (torch.matmul(x_2d, wq) + bq)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        k = (
            (torch.matmul(x_2d, wk) + bk)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        v = (
            (torch.matmul(x_2d, wv) + bv)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        return q, k, v

    run_graph_test_with_random_inputs(
        mha_qkv_linear,
        [
            (batch, seq_len, hidden_dim),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
            (num_heads * head_size,),
            (num_heads * head_size,),
            (num_heads * head_size,),
        ],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
@pytest.mark.xfail(reason="This seems to not trigger the pattern?")
def test_split_query_key_value_and_split_heads_mha_linear(request):
    """MHA pattern: equal Q/K/V weight shapes, F.linear (with bias) variant."""
    batch, seq_len, hidden_dim = 1, 32, 512
    num_heads, head_size = 8, 64

    def mha_qkv_linear(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
    ) -> tuple:
        x_2d = x.reshape(batch * seq_len, hidden_dim)
        q = (
            torch.nn.functional.linear(x_2d, wq, bq)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        k = (
            torch.nn.functional.linear(x_2d, wk, bk)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        v = (
            torch.nn.functional.linear(x_2d, wv, bv)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        return q, k, v

    run_graph_test_with_random_inputs(
        mha_qkv_linear,
        [
            (batch, seq_len, hidden_dim),
            (num_heads * head_size, hidden_dim),
            (num_heads * head_size, hidden_dim),
            (num_heads * head_size, hidden_dim),
            (num_heads * head_size,),
            (num_heads * head_size,),
            (num_heads * head_size,),
        ],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
def test_split_query_key_value_and_split_heads_mha_transposed_key(request):
    """MHA pattern: transposed key variant (key permute is [0, 2, 3, 1])."""
    batch, seq_len, hidden_dim = 1, 32, 512
    num_heads, head_size = 8, 64

    def mha_qkv_transposed_key(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
    ) -> tuple:
        x_2d = x.reshape(batch * seq_len, hidden_dim)
        q = (
            torch.matmul(x_2d, wq)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        k = (
            torch.matmul(x_2d, wk)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 3, 1)
        )
        v = (
            torch.matmul(x_2d, wv)
            .reshape(batch, seq_len, num_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        return q, k, v

    run_graph_test_with_random_inputs(
        mha_qkv_transposed_key,
        [
            (batch, seq_len, hidden_dim),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
        ],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
def test_split_query_key_value_and_split_heads_gqa_matmul(request):
    """GQA pattern: query has more heads than key/value, matmul variant."""
    batch, seq_len, hidden_dim = 1, 32, 512
    num_query_heads, num_kv_heads, head_size = 8, 2, 64

    def gqa_qkv_matmul(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
    ) -> tuple:
        x_2d = x.reshape(batch * seq_len, hidden_dim)
        q = (
            torch.matmul(x_2d, wq)
            .reshape(batch, seq_len, num_query_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        k = (
            torch.matmul(x_2d, wk)
            .reshape(batch, seq_len, num_kv_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        v = (
            torch.matmul(x_2d, wv)
            .reshape(batch, seq_len, num_kv_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        return q, k, v

    run_graph_test_with_random_inputs(
        gqa_qkv_matmul,
        [
            (batch, seq_len, hidden_dim),
            (hidden_dim, num_query_heads * head_size),
            (hidden_dim, num_kv_heads * head_size),
            (hidden_dim, num_kv_heads * head_size),
        ],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
